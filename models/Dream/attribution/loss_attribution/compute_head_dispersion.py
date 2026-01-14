#!/usr/bin/env python3
"""
Compute per-layer, per-head *attention dispersion* metrics for Dream (block-level).

We compute dispersion on the key-block probability distribution per head:
  - Compute token-level attention probs: softmax(q @ k^T / sqrt(d))
  - Aggregate over key positions into blocks of size `block_size` by summing probs
  - Average over selected query positions and samples
  - Metrics per head:
      * entropy (nats)
      * normalized entropy (divide by log(#blocks), so ~[0,1])
      * effective_blocks = exp(entropy)
      * topk_mass (sum of largest K block masses; K is a fraction of #blocks)

This aligns with the block-sparse mask selection logic used by Dream's sparse/adaptive attention.

Output: `head_dispersion.pt`
  {
    "dispersion": {...},
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

from models.Dream.core.modeling_dream import DreamModel

# Reuse helper utilities from the Dream loss-attribution script to keep prompt/tokenization consistent.
# IMPORTANT: We load the module by file path to avoid optional imports from models.Dream.attribution.__init__.
import importlib.util

_BASE_PATH = os.path.join(os.path.dirname(__file__), "compute_loss_attribution.py")
_spec = importlib.util.spec_from_file_location("_dream_loss_attr_layerwise", _BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load base module spec from: {_BASE_PATH}")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)  # type: ignore[arg-type]

_find_layers = _base._find_layers
_get_mask_token_id = _base._get_mask_token_id
_build_gsm8k_prompt_and_completion = _base._build_gsm8k_prompt_and_completion
_build_nemotron_prompt_and_completion = _base._build_nemotron_prompt_and_completion
_tokenize_pair = _base._tokenize_pair
_stable_int_seed = _base._stable_int_seed


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
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
    pad_cfg[2 * (x.dim() - 1 - (dim % x.dim())) + 1] = pad
    return F.pad(x, pad_cfg, mode="constant", value=value)


def _aggregate_key_blocks(attn_probs: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    attn_probs: (B, n_heads, Q, K)
    returns:    (B, n_heads, Q, n_k_blocks)
    """
    if block_size <= 1:
        return attn_probs
    B, H, Q, K = attn_probs.shape
    x = _pad_to_multiple(attn_probs, block_size, dim=-1, value=0.0)
    Kp = int(x.size(-1))
    n_blocks = Kp // block_size
    x = x.view(B, H, Q, n_blocks, block_size).sum(dim=-1)
    x = x / (x.sum(dim=-1, keepdim=True) + 1e-12)
    return x


def _qk_block_mean(attn_probs: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Match `create_attention_block_mask`'s block importance reduction:
      blocks = attn.unfold(q, bs).unfold(k, bs)
      block_importance = blocks.abs().mean(dim=(-1, -2))  # (B, H, q_blocks, k_blocks)

    attn_probs: (B, H, Q, K)
    returns:    (B, H, q_blocks, k_blocks)
    """
    if block_size <= 1:
        return attn_probs
    bsz, num_heads, q_len, kv_len = attn_probs.shape
    num_blocks_q = (q_len + block_size - 1) // block_size
    num_blocks_kv = (kv_len + block_size - 1) // block_size
    pad_q = (num_blocks_q * block_size - q_len) % block_size
    pad_kv = (num_blocks_kv * block_size - kv_len) % block_size
    padded = F.pad(attn_probs, (0, pad_kv, 0, pad_q))
    blocks = padded.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    return blocks.abs().mean(dim=(-1, -2))  # (B, H, q_blocks, k_blocks)


def _select_query_positions(
    q_len: int,
    *,
    completion_start: int,
    mode: str,
    last_n: int,
) -> torch.Tensor:
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
    dataset_use_chat_template: bool,
    gsm8k_completion_mode: str,
    block_size: int,
    query_span: str,
    last_n: int,
    topk_frac: float,
    use_amp_bf16: bool,
    show_progress: bool,
    progress_update_every: int,
    causal_mask: bool,
    aggregation: str,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Any]]:
    if len(layers) != len(layer_indices):
        raise ValueError("layers and layer_indices must have same length.")
    if len(layers) == 0:
        raise ValueError("No layers selected.")

    cfg = getattr(model, "config", None)
    n_heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_heads", 0) or 0)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or n_heads)
    if n_heads <= 0:
        raise ValueError("Cannot infer num_attention_heads from model.config")
    if n_kv_heads <= 0:
        n_kv_heads = n_heads

    entropy_sum: Dict[int, torch.Tensor] = {int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices}
    ent_norm_sum: Dict[int, torch.Tensor] = {int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices}
    effblk_sum: Dict[int, torch.Tensor] = {int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices}
    topk_sum: Dict[int, torch.Tensor] = {int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices}
    topk_disp_sum: Dict[int, torch.Tensor] = {
        int(li): torch.zeros(n_heads, dtype=torch.float64) for li in layer_indices
    }
    hook_calls: Dict[int, int] = {int(li): 0 for li in layer_indices}
    count = 0
    total_rows_seen = 0

    # Per-forward info passed to hooks
    current_info: Dict[str, Any] = {}
    rope_warned = False
    aggregation = str(aggregation)
    if aggregation not in ("key_block_mass", "qk_block_mean"):
        raise ValueError(f"Invalid aggregation={aggregation!r}. Expected 'key_block_mass' or 'qk_block_mean'.")

    # Hook each selected attention module (layer.self_attn) so we see the normalized hidden_states and position_ids.
    handles: List[Any] = []
    li_to_layer: Dict[int, torch.nn.Module] = {int(li): lyr for li, lyr in zip(layer_indices, layers)}

    def _make_attn_pre_hook(layer_idx: int):
        # IMPORTANT (why Dream was all-zeros before):
        # DreamDecoderLayer calls self_attn with **kwargs**:
        #   self.self_attn(hidden_states=..., attention_mask=..., position_ids=..., ...)
        # In PyTorch, forward_pre_hook only receives positional `inputs` unless `with_kwargs=True`.
        # Without kwargs, `inputs` is empty -> hook does nothing -> all metrics stay 0.
        def _hook(module, inputs, kwargs):
            hook_calls[layer_idx] += 1
            hidden_states = None
            if inputs and torch.is_tensor(inputs[0]):
                hidden_states = inputs[0]
            elif isinstance(kwargs, dict) and torch.is_tensor(kwargs.get("hidden_states", None)):
                hidden_states = kwargs["hidden_states"]
            if hidden_states is None:
                return

            # Try to extract position_ids (may be None; we fall back to arange).
            position_ids = None
            if isinstance(kwargs, dict) and torch.is_tensor(kwargs.get("position_ids", None)):
                position_ids = kwargs["position_ids"]

            completion_start = int(current_info.get("completion_start", 0))

            bsz, q_len, _ = hidden_states.size()
            # Projections
            q_lin = module.q_proj(hidden_states)
            k_lin = module.k_proj(hidden_states)
            v_lin = module.v_proj(hidden_states)

            head_dim = int(getattr(module, "head_dim", 0) or (q_lin.size(-1) // n_heads))
            q = q_lin.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
            k = k_lin.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
            v = v_lin.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

            # RoPE (match model behavior as much as possible)
            if hasattr(module, "rotary_emb"):
                try:
                    if position_ids is None:
                        # Create simple position_ids: (1, q_len)
                        position_ids_ = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
                    else:
                        position_ids_ = position_ids
                    cos, sin = module.rotary_emb(v, position_ids_)
                    if hasattr(module, "_apply_rotary_pos_emb"):
                        q2, k2 = module._apply_rotary_pos_emb(q, k, cos, sin)
                        q, k = q2, k2
                except Exception:
                    nonlocal rope_warned
                    if not rope_warned:
                        rope_warned = True
                        print(
                            "[warn] RoPE application failed in compute_head_dispersion (Dream). "
                            "Proceeding WITHOUT RoPE for dispersion. If you care about absolute correctness, "
                            "please validate/adjust RoPE application for this model."
                        )

            # Repeat KV heads to Q heads for computing per-Q-head attention weights
            if k.size(1) != q.size(1):
                assert q.size(1) % k.size(1) == 0
                rep = q.size(1) // k.size(1)
                k_for_q = k.repeat_interleave(rep, dim=1)[:, : q.size(1), :, :]
            else:
                k_for_q = k

            pos = _select_query_positions(q_len, completion_start=completion_start, mode=query_span, last_n=last_n)
            if pos.numel() == 0:
                return
            pos_d = pos.to(device=hidden_states.device)
            q_sel = q.index_select(dim=2, index=pos_d)

            logits = torch.matmul(q_sel.to(torch.float32), k_for_q.transpose(2, 3).to(torch.float32)) / float(
                head_dim**0.5
            )
            if bool(causal_mask):
                # Optional causal mask (default off). Dream in this repo is non-causal for diffusion generation.
                key_pos = torch.arange(logits.size(-1), device=logits.device).view(1, 1, 1, -1)
                qpos = pos_d.view(1, 1, -1, 1)
                logits = logits.masked_fill(key_pos > qpos, float("-inf"))
            attn = torch.softmax(logits, dim=-1)
            if aggregation == "key_block_mass":
                block_p = _aggregate_key_blocks(attn, block_size=block_size)
                p = block_p.mean(dim=(0, 2))  # (H, k_blocks)
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
            else:
                bi = _qk_block_mean(attn, block_size=block_size)  # (B,H,q_blocks,k_blocks)
                p = bi.mean(dim=(0, 2))  # (H,k_blocks)
                p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
            n_blocks = int(p.size(-1))
            ent = _entropy_from_probs(p)
            ent_norm = ent / float(max(1, torch.log(torch.tensor(float(n_blocks))).item()))
            eff = torch.exp(ent)

            k_top = max(1, int(torch.ceil(torch.tensor(float(n_blocks) * float(topk_frac))).item()))
            k_top = min(k_top, n_blocks)
            topk_mass = torch.topk(p, k=k_top, dim=-1).values.sum(dim=-1)
            topk_disp = 1.0 - topk_mass

            entropy_sum[layer_idx] += ent.detach().to("cpu", dtype=torch.float64)
            ent_norm_sum[layer_idx] += ent_norm.detach().to("cpu", dtype=torch.float64)
            effblk_sum[layer_idx] += eff.detach().to("cpu", dtype=torch.float64)
            topk_sum[layer_idx] += topk_mass.detach().to("cpu", dtype=torch.float64)
            topk_disp_sum[layer_idx] += topk_disp.detach().to("cpu", dtype=torch.float64)

        return _hook

    # Register on each layer's self_attn
    for li, layer in li_to_layer.items():
        if not hasattr(layer, "self_attn"):
            raise AttributeError(f"Dream layer {li} has no self_attn")
        attn = getattr(layer, "self_attn")
        # Use with_kwargs=True to capture keyword arguments (Dream passes hidden_states via kwargs).
        try:
            h = attn.register_forward_pre_hook(_make_attn_pre_hook(int(li)), with_kwargs=True)  # type: ignore[arg-type]
        except TypeError:
            # Fallback for older PyTorch: use a forward hook with kwargs support.
            h = attn.register_forward_hook(lambda m, inp, out, li=int(li): _make_attn_pre_hook(li)(m, inp, {}))  # type: ignore[arg-type]
        handles.append(h)

    iterator = enumerate(dataset_rows)
    # Keep Dream's behavior: avoid tqdm in nohup/non-tty logs
    use_tqdm = bool(show_progress and (tqdm is not None) and sys.stderr.isatty())
    if use_tqdm:
        iterator = tqdm(  # type: ignore[assignment]
            iterator,
            total=len(dataset_rows),
            desc="head_dispersion",
            dynamic_ncols=True,
            leave=False,
        )

    try:
        for row_idx, row in iterator:
            total_rows_seen += 1
            if dataset_name == "gsm8k":
                prompt, completion = _build_gsm8k_prompt_and_completion(
                    row["question"],
                    row["answer"],
                    tokenizer=tokenizer,
                    use_chat_template=bool(dataset_use_chat_template),
                    completion_mode=str(gsm8k_completion_mode),
                )
            elif dataset_name == "nemotron":
                prompt, completion = _build_nemotron_prompt_and_completion(
                    row,
                    tokenizer=tokenizer,
                    use_chat_template=bool(dataset_use_chat_template),
                )
            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name}")

            input_ids, attention_mask, completion_start = _tokenize_pair(
                tokenizer,
                prompt,
                completion,
                device=device,
                max_length=max_length,
            )
            current_info["completion_start"] = int(completion_start)

            if use_amp_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask).logits

            count += 1
            if show_progress and (not use_tqdm) and progress_update_every > 0:
                if (count % int(progress_update_every)) == 0:
                    print(f"[progress] processed={count}/{len(dataset_rows)}")

        if count <= 0:
            raise RuntimeError("No samples processed.")

        out: Dict[str, Dict[int, torch.Tensor]] = {
            "entropy": {li: (entropy_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
            "entropy_norm": {li: (ent_norm_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
            "effective_blocks": {li: (effblk_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
            "topk_mass": {li: (topk_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
            "topk_dispersion": {li: (topk_disp_sum[li] / float(count)).to(torch.float32) for li in layer_indices},
        }
        debug = {
            "n_samples": int(count),
            "total_rows_seen": int(total_rows_seen),
            "hook_calls": {int(k): int(v) for k, v in hook_calls.items()},
        }
        return out, debug
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


def main() -> None:
    p = argparse.ArgumentParser(description="Compute per-head attention dispersion metrics for Dream (block-level).")
    p.add_argument("--model_path", type=str, required=True)
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
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--query_span", type=str, default="completion", choices=["all", "completion", "last_n"])
    p.add_argument("--last_n", type=int, default=256)
    p.add_argument("--topk_frac", type=float, default=0.1)
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
        help="If set, apply a causal mask (prevent attending to future keys). Default off (Dream here is non-causal).",
    )
    p.add_argument("--dataset_use_chat_template", action="store_true", default=False)
    p.add_argument("--gsm8k_completion_mode", type=str, default="final", choices=["final", "full"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp_bf16", action="store_true", default=True)
    p.add_argument("--no_progress", action="store_true", default=False)
    p.add_argument("--progress_update_every", type=int, default=10)
    args = p.parse_args()

    base_seed = int(args.seed)
    data_seed = base_seed if int(args.data_seed) < 0 else int(args.data_seed)
    torch.manual_seed(base_seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load dataset rows
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", args.dataset_config, split=args.split)
        if bool(args.dataset_shuffle):
            ds = ds.shuffle(seed=int(data_seed))
        rows = [ds[i] for i in range(min(int(args.max_samples), len(ds)))]
    else:
        cats = [c.strip() for c in str(args.nemotron_categories).split(",") if c.strip()]
        rows = []
        pool_per_category = max(int(args.samples_per_category), int(args.nemotron_pool_per_category))
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
            take_n = min(int(args.samples_per_category), len(buf))
            rows.extend(buf[:take_n])
            per_cat_counts[str(cat)] = int(take_n)
        # IMPORTANT: apply dataset_shuffle globally for nemotron too.
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
    if args.dataset == "nemotron":
        print(f"[data] nemotron_categories={str(args.nemotron_categories)}")
        print(
            f"[data] samples_per_category={int(args.samples_per_category)} "
            f"nemotron_pool_per_category={int(args.nemotron_pool_per_category)} "
            f"pool_per_category_used={int(pool_per_category)}"
        )
        print(f"[data] per_category_counts={per_cat_counts}")
    print(f"[data] rows_loaded={len(rows)}")
    print("-" * 80)

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

    disp, debug = compute_head_dispersion(
        model=model,
        layers=selected_layers,
        layer_indices=selected_layer_indices,
        tokenizer=tokenizer,
        dataset_rows=rows,
        device=device,
        max_length=int(args.max_length),
        dataset_name=str(args.dataset),
        dataset_use_chat_template=bool(args.dataset_use_chat_template),
        gsm8k_completion_mode=str(args.gsm8k_completion_mode),
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
            "dataset_use_chat_template": bool(args.dataset_use_chat_template),
            "gsm8k_completion_mode": str(args.gsm8k_completion_mode),
            "generated_at": datetime.now().isoformat(),
            "debug": debug,
        },
    }
    if args.dataset == "nemotron":
        out["metadata"]["samples_per_category"] = int(args.samples_per_category)
        out["metadata"]["nemotron_pool_per_category"] = int(args.nemotron_pool_per_category)
        out["metadata"]["nemotron_per_category_counts"] = per_cat_counts
    # Mirror the summary into logs for easy grepping
    if isinstance(debug, dict):
        print(
            f"[data] processed summary: rows_seen={int(debug.get('total_rows_seen', 0))} "
            f"items_used={int(debug.get('n_samples', 0))}"
        )

    out_path = os.path.join(args.output_dir, "head_dispersion.pt")
    torch.save(out, out_path)
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


