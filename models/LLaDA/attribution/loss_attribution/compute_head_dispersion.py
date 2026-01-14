#!/usr/bin/env python3
"""
Compute per-layer, per-head *attention dispersion* metrics for LLaDA.

Motivation
----------
Adaptive block-sparse attention selects KV *blocks* per head based on attention patterns.
For designing per-head keep_ratio, a head's "dispersion" (how spread its attention mass is across blocks)
is often more directly relevant than "head importance" computed by gating the head output.

This script computes dispersion on the *block-level attention mass distribution*:
  - Compute attention weights: softmax(q @ k^T / sqrt(d))
  - Aggregate over key positions into blocks of size `block_size` by summing probabilities
  - Average block distributions over selected query positions and samples
  - Compute metrics per head:
      * entropy (nats)
      * normalized entropy in [0,1] (divide by log(#blocks))
      * effective_blocks = exp(entropy)      (aka perplexity)
      * topk_mass (sum of largest K block masses; K is a fraction of #blocks)

Output
------
Writes `head_dispersion.pt`:
  {
    "dispersion": {
      "entropy": {layer_idx: tensor[n_heads]},
      "entropy_norm": {layer_idx: tensor[n_heads]},
      "effective_blocks": {layer_idx: tensor[n_heads]},
      "topk_mass": {layer_idx: tensor[n_heads]},
    },
    "metadata": {...}
  }
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Ensure repo root is on sys.path (so `import models.*` works when running directly)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import AutoTokenizer
from datasets import load_dataset

from models.LLaDA.core.modeling import LLaDAModelLM
from models.LLaDA.core.configuration import ActivationCheckpointingStrategy

# Reuse attribution helpers for dataset/tokenization logic (keep behavior consistent).
from models.LLaDA.attribution.loss_attribution.compute_loss_attribution import (  # noqa: E402
    _find_layers,
    _build_gsm8k_prompt_and_answer,
    _build_nemotron_prompt_and_completion,
    _tokenize_pair,
    _get_mask_token_id,
    _stable_int_seed,
)


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # p: (..., K) where K is #blocks, sum over last dim ~= 1
    p = torch.clamp(p, min=eps)
    return -(p * torch.log(p)).sum(dim=-1)


def _pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -1, value: float = 0.0) -> torch.Tensor:
    n = int(x.size(dim))
    if multiple <= 0:
        return x
    r = n % multiple
    if r == 0:
        return x
    pad = multiple - r
    pad_cfg = [0, 0] * x.dim()
    # PyTorch pad uses reverse order (last dim first)
    pad_cfg[2 * (x.dim() - 1 - (dim % x.dim())) + 1] = pad
    return F.pad(x, pad_cfg, mode="constant", value=value)


def _aggregate_key_blocks(attn_probs: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Aggregate token-level attention probs over key positions into key-block probs.

    attn_probs: (B, n_heads, Q, K)
    Returns:    (B, n_heads, Q, n_k_blocks)
    """
    if block_size <= 1:
        return attn_probs
    B, H, Q, K = attn_probs.shape
    x = _pad_to_multiple(attn_probs, block_size, dim=-1, value=0.0)
    Kp = int(x.size(-1))
    n_blocks = Kp // block_size
    x = x.view(B, H, Q, n_blocks, block_size).sum(dim=-1)
    # Numerical safety: renormalize (padding adds 0 mass so sum should already be 1)
    x = x / (x.sum(dim=-1, keepdim=True) + 1e-12)
    return x


def _qk_block_mean(attn_probs: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Match `create_attention_block_mask`'s block importance reduction:
      blocks = attn.unfold(q, bs).unfold(k, bs)
      block_importance = blocks.abs().mean(dim=(-1, -2))  # (B, H, q_blocks, k_blocks)

    attn_probs: (B, H, Q, K), non-negative and sums to 1 over K for each (B,H,Q).
    returns:    (B, H, q_blocks, k_blocks) in float32.
    """
    if block_size <= 1:
        # Treat each token as its own "block"
        return attn_probs
    bsz, num_heads, q_len, kv_len = attn_probs.shape
    num_blocks_q = (q_len + block_size - 1) // block_size
    num_blocks_kv = (kv_len + block_size - 1) // block_size
    pad_q = (num_blocks_q * block_size - q_len) % block_size
    pad_kv = (num_blocks_kv * block_size - kv_len) % block_size
    padded = F.pad(attn_probs, (0, pad_kv, 0, pad_q))
    blocks = padded.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    # attn_probs >= 0 so abs() is a no-op; keep it for conceptual alignment
    return blocks.abs().mean(dim=(-1, -2))  # (B, H, q_blocks, k_blocks)


def _select_query_positions(
    q_len: int,
    *,
    completion_start: int,
    mode: str,
    last_n: int,
) -> torch.Tensor:
    """
    Returns a 1D LongTensor of query positions to evaluate.
    """
    mode = str(mode)
    if mode == "all":
        return torch.arange(q_len, dtype=torch.long)
    if mode == "completion":
        start = int(max(0, min(int(completion_start), q_len)))
        return torch.arange(start, q_len, dtype=torch.long)
    if mode == "last_n":
        n = int(max(1, last_n))
        start = max(0, q_len - n)
        return torch.arange(start, q_len, dtype=torch.long)
    raise ValueError(f"Unsupported query_span={mode!r}. Expected one of: all, completion, last_n")


@torch.no_grad()
def compute_head_dispersion(
    model: torch.nn.Module,
    layers: List[torch.nn.Module],
    layer_indices: List[int],
    tokenizer,
    dataset_rows: List[Dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    dataset_name: str,
    block_size: int,
    query_span: str,
    last_n: int,
    topk_frac: float,
    use_amp_bf16: bool,
    show_progress: bool,
    progress_update_every: int,
    causal_mask: bool,
    aggregation: str,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Returns dict with keys: entropy, entropy_norm, effective_blocks, topk_mass, topk_dispersion.
    Each value is a dict[layer_idx] = tensor[n_heads] (float32 on CPU).
    """
    if len(layers) != len(layer_indices):
        raise ValueError("layers and layer_indices must have same length.")
    if len(layers) == 0:
        raise ValueError("No layers selected.")

    # We'll use model config for head counts and kv-heads.
    cfg = getattr(model, "config", None)
    n_heads = int(getattr(cfg, "n_heads", 0) or getattr(cfg, "num_attention_heads", 0) or 0)
    n_kv_heads = int(getattr(cfg, "effective_n_kv_heads", 0) or getattr(cfg, "num_key_value_heads", 0) or n_heads)
    if n_heads <= 0:
        raise ValueError("Cannot infer n_heads from model.config")
    if n_kv_heads <= 0:
        n_kv_heads = n_heads

    # Accumulators (on CPU for simplicity)
    entropy_sum: Dict[int, torch.Tensor] = {int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices}
    ent_norm_sum: Dict[int, torch.Tensor] = {int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices}
    effblk_sum: Dict[int, torch.Tensor] = {int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices}
    topk_sum: Dict[int, torch.Tensor] = {int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices}
    topk_disp_sum: Dict[int, torch.Tensor] = {
        int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices
    }
    count = 0
    total_rows_seen = 0

    # Precompute which block type we have (llama vs sequential)
    # We'll compute Q/K from each block's modules without relying on internal attention() hooks.
    def _compute_qk_for_block(block: torch.nn.Module, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_in: (B, T, d_model) - input to the block
        # We need "pre-attn normalized" input for projections.
        if not hasattr(block, "attn_norm"):
            raise AttributeError("Block has no attn_norm; unsupported for dispersion computation.")
        x_norm = block.attn_norm(x_in)

        # LLaDA llama blocks: q_proj/k_proj exist. Sequential blocks: att_proj exists.
        if hasattr(block, "q_proj") and hasattr(block, "k_proj"):
            q_lin = block.q_proj(x_norm)
            k_lin = block.k_proj(x_norm)
        elif hasattr(block, "att_proj") and hasattr(block, "fused_dims"):
            fused = block.att_proj(x_norm)
            # fused dims: (d_model, n_kv*head_dim, n_kv*head_dim)
            d_q, d_k, _d_v = block.fused_dims
            q_lin = fused[..., :d_q]
            k_lin = fused[..., d_q : d_q + d_k]
        else:
            raise AttributeError("Cannot find q_proj/k_proj or att_proj on this block.")

        return q_lin, k_lin

    iterator = enumerate(dataset_rows)
    if show_progress and tqdm is not None:
        iterator = tqdm(  # type: ignore[assignment]
            iterator,
            total=len(dataset_rows),
            desc="head_dispersion",
            dynamic_ncols=True,
            leave=False,
        )

    mask_token_id = _get_mask_token_id(model, tokenizer)

    # Register forward-pre-hooks on selected blocks to compute dispersion using the block input x.
    # We compute and accumulate immediately inside the hook (no need to keep all intermediates).
    handles: List[Any] = []
    li_to_block: Dict[int, torch.nn.Module] = {int(li): blk for li, blk in zip(layer_indices, layers)}

    rope_warned = False
    aggregation = str(aggregation)
    if aggregation not in ("key_block_mass", "qk_block_mean"):
        raise ValueError(f"Invalid aggregation={aggregation!r}. Expected 'key_block_mass' or 'qk_block_mean'.")

    # Per-forward storage for completion_start and q_len; set by outer loop before calling model
    current_info: Dict[str, Any] = {}

    def _make_block_hook(layer_idx: int):
        def _hook(module, inputs):
            # inputs[0] is x: (B, T, C)
            x_in = inputs[0]
            if not torch.is_tensor(x_in):
                return
            completion_start = int(current_info.get("completion_start", 0))
            # Compute q/k (linear projected), then reshape and apply RoPE consistently with model.
            q_lin, k_lin = _compute_qk_for_block(module, x_in)

            B, T, Cq = q_lin.shape
            head_dim = Cq // n_heads
            q = q_lin.view(B, T, n_heads, head_dim).transpose(1, 2)  # (B, n_heads, T, d)
            k = k_lin.view(B, T, n_kv_heads, head_dim).transpose(1, 2)  # (B, n_kv, T, d)

            # Apply RoPE if present on module
            if hasattr(module, "rotary_emb") and callable(getattr(module, "rotary_emb")):
                try:
                    q, k = module.rotary_emb(q, k)
                except Exception:
                    # If RoPE signature differs, skip but warn once (otherwise this becomes a silent footgun).
                    nonlocal rope_warned
                    if not rope_warned:
                        rope_warned = True
                        print(
                            "[warn] RoPE application failed in compute_head_dispersion (LLaDA). "
                            "Proceeding WITHOUT RoPE for dispersion. If you care about absolute correctness, "
                            "please validate/adjust RoPE application for this model."
                        )

            # Repeat KV heads to Q-heads for GQA if needed
            if k.size(1) != q.size(1):
                assert q.size(1) % k.size(1) == 0
                rep = q.size(1) // k.size(1)
                k_for_q = k.repeat_interleave(rep, dim=1, output_size=q.size(1))
            else:
                k_for_q = k

            q_len = int(q.size(-2))
            pos = _select_query_positions(q_len, completion_start=completion_start, mode=query_span, last_n=last_n)
            if pos.numel() == 0:
                return

            # Move indices to device and slice queries
            pos_d = pos.to(device=q.device)
            q_sel = q.index_select(dim=2, index=pos_d)  # (B, n_heads, Q, d)
            # Compute attention probs in float32 for stability
            logits = torch.matmul(q_sel.to(torch.float32), k_for_q.transpose(2, 3).to(torch.float32)) / float(
                head_dim**0.5
            )
            if bool(causal_mask):
                # Apply causal mask (optional). Note: LLaDA/Dream in this repo typically use NON-causal attention,
                # so this should generally be left OFF unless you explicitly want causal-style dispersion.
                key_pos = torch.arange(logits.size(-1), device=logits.device).view(1, 1, 1, -1)
                # pos_d holds absolute query positions (within original sequence)
                qpos = pos_d.view(1, 1, -1, 1)
                logits = logits.masked_fill(key_pos > qpos, float("-inf"))
            attn = torch.softmax(logits, dim=-1)  # (B, n_heads, Q, K)
            if aggregation == "key_block_mass":
                block_p = _aggregate_key_blocks(attn, block_size=block_size)  # (B, H, Q, k_blocks)
                # Average over B and Q to get per-head distribution over key blocks
                p = block_p.mean(dim=(0, 2))  # (H, k_blocks)
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
            else:
                # Align with create_attention_block_mask's block importance reduction over (q_block, k_block).
                bi = _qk_block_mean(attn, block_size=block_size)  # (B, H, q_blocks, k_blocks)
                p = bi.mean(dim=(0, 2))  # (H, k_blocks) - not a probability; normalize for entropy/topk metrics
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)

            n_blocks = int(p.size(-1))
            ent = _entropy_from_probs(p)  # (n_heads,)
            ent_norm = ent / float(max(1, torch.log(torch.tensor(float(n_blocks))).item()))
            eff = torch.exp(ent)

            k_top = max(1, int(torch.ceil(torch.tensor(float(n_blocks) * float(topk_frac))).item()))
            k_top = min(k_top, n_blocks)
            topk_mass = torch.topk(p, k=k_top, dim=-1).values.sum(dim=-1)
            topk_disp = 1.0 - topk_mass

            # Accumulate on CPU (float64 for numeric stability)
            entropy_sum[layer_idx] += ent.detach().to("cpu", dtype=torch.float64)
            ent_norm_sum[layer_idx] += ent_norm.detach().to("cpu", dtype=torch.float64)
            effblk_sum[layer_idx] += eff.detach().to("cpu", dtype=torch.float64)
            topk_sum[layer_idx] += topk_mass.detach().to("cpu", dtype=torch.float64)
            topk_disp_sum[layer_idx] += topk_disp.detach().to("cpu", dtype=torch.float64)

        return _hook

    for li, blk in li_to_block.items():
        h = blk.register_forward_pre_hook(_make_block_hook(int(li)))
        handles.append(h)

    try:
        for row_idx, row in iterator:
            total_rows_seen += 1
            if dataset_name == "gsm8k":
                prompt, completion = _build_gsm8k_prompt_and_answer(row["question"], row["answer"])
            elif dataset_name == "nemotron":
                prompt, completion = _build_nemotron_prompt_and_completion(row)
            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name}")

            input_ids, attention_mask, completion_start = _tokenize_pair(
                tokenizer,
                prompt,
                completion,
                device=device,
                max_length=max_length,
                mask_token_id=mask_token_id,
            )

            current_info["completion_start"] = int(completion_start)

            if use_amp_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(input_ids, attention_mask=attention_mask).logits
            else:
                _ = model(input_ids, attention_mask=attention_mask).logits

            count += 1
            if show_progress and tqdm is None and progress_update_every > 0:
                if (count % int(progress_update_every)) == 0:
                    print(f"[progress] processed={count}/{len(dataset_rows)}")

        if count <= 0:
            raise RuntimeError("No samples processed.")

        # Mean and cast to float32
        out = {
            "entropy": {li: (entropy_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
            "entropy_norm": {li: (ent_norm_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
            "effective_blocks": {li: (effblk_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
            "topk_mass": {li: (topk_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
            "topk_dispersion": {li: (topk_disp_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
        }
        compute_head_dispersion._diag = {  # type: ignore[attr-defined]
            "total_rows_seen": int(total_rows_seen),
            "total_items_processed": int(count),
        }
        return out
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description="Compute per-head attention dispersion metrics for LLaDA (block-level).")
    p.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "nemotron"])
    p.add_argument("--dataset_config", type=str, default="main")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--dataset_shuffle", action="store_true", default=False)
    # Nemotron sampling knobs (for dataset=nemotron)
    p.add_argument("--samples_per_category", type=int, default=10)
    p.add_argument("--nemotron_pool_per_category", type=int, default=1000)
    p.add_argument("--nemotron_categories", type=str, default="code,math,science,chat,safety")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--data_seed",
        type=int,
        default=-1,
        help="Seed for dataset subsampling/shuffle (esp. nemotron). -1 means use --seed.",
    )
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--layer_start", type=int, default=0)
    p.add_argument("--layer_end", type=int, default=-1, help="Inclusive. -1 means last layer.")
    p.add_argument("--block_size", type=int, default=128, help="Key block size for block-level dispersion.")
    p.add_argument("--query_span", type=str, default="completion", choices=["all", "completion", "last_n"])
    p.add_argument("--last_n", type=int, default=256, help="Only used when query_span=last_n")
    p.add_argument("--topk_frac", type=float, default=0.1, help="Top-k mass fraction over key blocks (e.g. 0.1).")
    p.add_argument(
        "--aggregation",
        type=str,
        default="key_block_mass",
        choices=["key_block_mass", "qk_block_mean"],
        help=(
            "How to aggregate attention into key-block distributions. "
            "key_block_mass: sum probs within each key block (default). "
            "qk_block_mean: match create_attention_block_mask's unfold+mean over (q_block,k_block)."
        ),
    )
    p.add_argument(
        "--causal_mask",
        action="store_true",
        default=False,
        help="If set, apply a causal mask (prevent attending to future keys). Default off (LLaDA here is non-causal).",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp_bf16", action="store_true", default=True)
    p.add_argument(
        "--activation_checkpointing",
        type=str,
        default="none",
        choices=["none", "whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"],
    )
    p.add_argument("--no_progress", action="store_true", default=False)
    p.add_argument("--progress_update_every", type=int, default=10)
    args = p.parse_args()

    base_seed = int(args.seed)
    data_seed = base_seed if int(args.data_seed) < 0 else int(args.data_seed)
    torch.manual_seed(base_seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model = LLaDAModelLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    if str(args.activation_checkpointing) != "none":
        strat = ActivationCheckpointingStrategy[str(args.activation_checkpointing)]
        if hasattr(model, "model") and hasattr(model.model, "set_activation_checkpointing"):
            model.model.set_activation_checkpointing(strat)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load dataset rows (mirror attribution behavior lightly)
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", args.dataset_config, split=args.split)
        if bool(args.dataset_shuffle):
            ds = ds.shuffle(seed=int(data_seed))
        rows = [ds[i] for i in range(min(int(args.max_samples), len(ds)))]
    else:
        # Nemotron streaming with deterministic sampling by category (to make data_seed meaningful).
        # Keep interface similar to loss attribution runners.
        cats = [c.strip() for c in str(getattr(args, "nemotron_categories", "code,math,science,chat,safety")).split(",") if c.strip()]
        samples_per_category = int(getattr(args, "samples_per_category", 10))
        pool_per_category = int(getattr(args, "nemotron_pool_per_category", 1000))
        rows = []
        per_cat_counts: Dict[str, int] = {}
        for cat_idx, cat in enumerate(cats):
            stream = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=cat, streaming=True)
            buf = []
            for i, sample in enumerate(stream):
                buf.append(sample)
                if len(buf) >= pool_per_category:
                    break
            if len(buf) > 1:
                g = torch.Generator()
                g.manual_seed(_stable_int_seed(int(data_seed), int(cat_idx)))
                idx = torch.randperm(len(buf), generator=g).tolist()
                buf = [buf[j] for j in idx]
            take_n = min(samples_per_category, len(buf))
            rows.extend(buf[:take_n])
            per_cat_counts[str(cat)] = int(take_n)
        # IMPORTANT: apply dataset_shuffle globally for nemotron too (avoid always taking early categories).
        if bool(args.dataset_shuffle) and len(rows) > 1:
            g_all = torch.Generator()
            g_all.manual_seed(_stable_int_seed(int(data_seed), 999_002))
            perm = torch.randperm(len(rows), generator=g_all).tolist()
            rows = [rows[i] for i in perm]
        if len(rows) > int(args.max_samples):
            rows = rows[: int(args.max_samples)]

    # Data diagnostics
    print("-" * 80)
    print("[data] sampling summary")
    print(f"[data] dataset={args.dataset} max_samples={int(args.max_samples)} dataset_shuffle={bool(args.dataset_shuffle)}")
    if args.dataset == \"nemotron\":
        print(f\"[data] nemotron_categories={getattr(args, 'nemotron_categories', '')}\")
        print(f\"[data] samples_per_category={int(getattr(args,'samples_per_category',0))} nemotron_pool_per_category={int(getattr(args,'nemotron_pool_per_category',0))} pool_per_category_used={int(pool_per_category)}\")
        print(f\"[data] per_category_counts={per_cat_counts}\")
    print(f\"[data] rows_loaded={len(rows)}\")
    print(\"-\" * 80)

    layers_all = _find_layers(model)
    n_layers = len(layers_all)
    layer_start = max(0, int(args.layer_start))
    layer_end = int(args.layer_end)
    if layer_end < 0:
        layer_end = n_layers - 1
    layer_end = min(layer_end, n_layers - 1)
    if layer_start > layer_end:
        raise ValueError(f"Invalid layer range: {layer_start}..{layer_end} (n_layers={n_layers})")

    selected_layer_indices = list(range(layer_start, layer_end + 1))
    selected_layers = [layers_all[i] for i in selected_layer_indices]

    disp = compute_head_dispersion(
        model=model,
        layers=selected_layers,
        layer_indices=selected_layer_indices,
        tokenizer=tokenizer,
        dataset_rows=rows,
        device=device,
        max_length=int(args.max_length),
        dataset_name=str(args.dataset),
        block_size=int(args.block_size),
        query_span=str(args.query_span),
        last_n=int(args.last_n),
        topk_frac=float(args.topk_frac),
        use_amp_bf16=bool(args.use_amp_bf16 and device.type == "cuda"),
        show_progress=(not bool(args.no_progress)),
        progress_update_every=int(args.progress_update_every),
        causal_mask=bool(args.causal_mask),
        aggregation=str(args.aggregation),
    )

    out = {
        "dispersion": {k: {int(li): v.detach().cpu() for li, v in d.items()} for k, d in disp.items()},
        "metadata": {
            "method": "block_level_attention_dispersion",
            "model_path": args.model_path,
            "dataset": str(args.dataset),
            "split": str(args.split),
            "max_samples": int(args.max_samples),
            "rows_loaded": int(len(rows)),
            "seed": int(args.seed),
            "data_seed": int(data_seed),
            "max_length": int(args.max_length),
            "layer_range": [int(layer_start), int(layer_end)],
            "block_size": int(args.block_size),
            "query_span": str(args.query_span),
            "last_n": int(args.last_n),
            "topk_frac": float(args.topk_frac),
            "aggregation": str(args.aggregation),
            "causal_mask": bool(args.causal_mask),
            "generated_at": datetime.now().isoformat(),
        },
    }

    if args.dataset == "nemotron":
        out["metadata"]["samples_per_category"] = int(getattr(args, "samples_per_category", 0))
        out["metadata"]["nemotron_pool_per_category"] = int(getattr(args, "nemotron_pool_per_category", 0))
        out["metadata"]["nemotron_per_category_counts"] = per_cat_counts

    diag = getattr(compute_head_dispersion, "_diag", None)
    if isinstance(diag, dict):
        out["metadata"]["total_rows_seen"] = int(diag.get("total_rows_seen", 0))
        out["metadata"]["total_items_processed"] = int(diag.get("total_items_processed", 0))
        print(
            f\"[data] processed summary: rows_seen={out['metadata']['total_rows_seen']} \"\n+            f\"items_used={out['metadata']['total_items_processed']}\"\n+        )

    out_path = os.path.join(args.output_dir, "head_dispersion.pt")
    torch.save(out, out_path)
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


