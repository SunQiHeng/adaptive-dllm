#!/usr/bin/env python3
"""
Loss-based head attribution for LLaDA (teacher forcing NLL, answer-only).

This is the **all-heads jointly** version of `compute_loss_attribution.py`.

Difference vs the original:
- Original computes IG **per-layer** by attaching a head-gating vector α_{h} to one layer at a time.
- This script attaches head-gating vectors to **all selected layers at once**, using one flattened gate
  vector α_flat that covers every (layer, head). IG is then computed jointly in one pass.

Everything else (dataset construction, diffusion-style masking, CE objective, IG path, output format)
is kept the same to make results comparable and to stay compatible with the evaluation pipeline.

Output:
- head_importance.pt
  {
    "importance_scores": {layer_idx: tensor[n_heads]},
    "metadata": {...}
  }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

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

# Reuse helper utilities from the layer-wise implementation to keep logic identical.
from models.LLaDA.attribution.loss_attribution.compute_loss_attribution import (  # noqa: E402
    _find_layers,
    _find_attn_and_oproj,
    _get_num_heads,
    _dry_run_check_o_proj_shape,
    _build_gsm8k_prompt_and_answer,
    _build_mmlu_prompt_and_answer,
    _build_humaneval_prompt_and_completion,
    _build_nemotron_prompt_and_completion,
    _tokenize_pair,
    _get_mask_token_id,
    _build_labels_and_masked_inputs_for_completion_span,
    _stable_int_seed,
    _masked_ce_answer_only_batch,
)


class _MultiOProjHeadGate:
    """
    Attach forward_pre_hook to multiple o_proj modules.
    Each hook scales its o_proj input per head by a slice of `alpha_flat`.

    alpha_flat layout:
      concat over selected layers: [layer0_heads..., layer1_heads..., ...]
    """

    def __init__(self, specs: List[Dict[str, Any]]):
        # specs: list of dicts with keys:
        #   o_proj, n_heads, head_dim, offset
        self.specs = specs
        self.alpha_flat: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

    def install(self) -> None:
        if self._handles:
            raise RuntimeError("Gate hooks already installed.")

        for spec in self.specs:
            o_proj = spec["o_proj"]
            n_heads = int(spec["n_heads"])
            head_dim = int(spec["head_dim"])
            offset = int(spec["offset"])

            def _make_pre_hook(o_proj_module, n_heads_, head_dim_, offset_):
                def _pre_hook(module, inputs):
                    if module is not o_proj_module:
                        return inputs
                    alpha_flat = self.alpha_flat
                    if alpha_flat is None:
                        return inputs
                    x = inputs[0]  # (B, T, hidden_size)
                    b, t, hs = x.shape
                    # (B, T, n_heads, head_dim)
                    x_ = x.view(b, t, n_heads_, head_dim_)
                    a = alpha_flat[offset_ : offset_ + n_heads_].view(1, 1, n_heads_, 1)
                    x_ = x_ * a
                    x_ = x_.view(b, t, hs)
                    return (x_,) + tuple(inputs[1:])

                return _pre_hook

            h = o_proj.register_forward_pre_hook(_make_pre_hook(o_proj, n_heads, head_dim, offset))
            self._handles.append(h)

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


def compute_all_heads_joint_ig(
    model: torch.nn.Module,
    layers: List[torch.nn.Module],
    layer_indices: List[int],
    tokenizer,
    dataset_rows: List[Dict[str, Any]],
    *,
    device: torch.device,
    ig_steps: int,
    baseline_value: float,
    max_length: int,
    num_heads_from_config: int,
    use_amp_bf16: bool,
    dataset_name: str,
    mask_probs: List[float],
    mask_samples_per_prob: int,
    loss_normalize: str,
    seed: int,
    ig_postprocess: str,
    mask_batch_size: int,
    show_progress: bool,
    progress_update_every: int,
    path_mode: str,
    path_samples: int,
    path_seed: int,
    min_completion_tokens: int,
    debug_gate: bool = False,
    debug_save_per_sample: int = 0,
    gsm8k_answer_mode: str = "final",
    gsm8k_fewshot_prefix: str = "",
) -> Dict[int, torch.Tensor]:
    """
    Joint IG over all (layer, head) gates at once.

    Returns:
      importance_scores: dict[layer_idx] = tensor[n_heads]  (float32 on device)
    """
    if len(layers) != len(layer_indices):
        raise ValueError("layers and layer_indices must have same length.")
    if len(layers) == 0:
        raise ValueError("No layers selected for attribution.")

    # Build hook specs and flat indexing
    specs: List[Dict[str, Any]] = []
    offsets: Dict[int, Tuple[int, int]] = {}  # layer_idx -> (offset, n_heads)
    total_heads = 0

    for li, layer in zip(layer_indices, layers):
        attn, o_proj = _find_attn_and_oproj(layer)
        n_heads = _get_num_heads(attn, fallback_from_config=num_heads_from_config)
        if not hasattr(o_proj, "in_features"):
            raise AttributeError("o_proj has no in_features; cannot infer head_dim safely.")
        hidden_size = int(o_proj.in_features)
        if hidden_size % n_heads != 0:
            raise ValueError(f"Layer {li}: hidden_size={hidden_size} not divisible by n_heads={n_heads}")
        head_dim = hidden_size // n_heads
        _dry_run_check_o_proj_shape(o_proj, hidden_size)

        offsets[int(li)] = (int(total_heads), int(n_heads))
        specs.append(
            {
                "o_proj": o_proj,
                "n_heads": int(n_heads),
                "head_dim": int(head_dim),
                "offset": int(total_heads),
            }
        )
        total_heads += int(n_heads)

    gate = _MultiOProjHeadGate(specs)
    gate.install()

    try:
        ig_sum_flat = torch.zeros(total_heads, device=device, dtype=torch.float32)
        total_items = 0
        total_rows_seen = 0
        total_rows_skipped_no_variants = 0
        per_sample_ig: List[torch.Tensor] = []
        completion_lens: List[int] = []

        mask_token_id = _get_mask_token_id(model, tokenizer)

        did_debug_check = False

        iterator = enumerate(dataset_rows)
        if show_progress and tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(dataset_rows),
                desc="all_heads_joint",
                dynamic_ncols=True,
                leave=False,
            )

        for row_idx, row in iterator:
            total_rows_seen += 1
            if dataset_name == "gsm8k":
                prompt, completion = _build_gsm8k_prompt_and_answer(
                    row["question"],
                    row["answer"],
                    answer_mode=str(gsm8k_answer_mode),
                    fewshot_prefix=str(gsm8k_fewshot_prefix),
                )
            elif dataset_name == "nemotron":
                prompt, completion = _build_nemotron_prompt_and_completion(row)
            elif dataset_name == "mmlu":
                prompt, completion = _build_mmlu_prompt_and_answer(row)
            elif dataset_name == "humaneval":
                prompt, completion = _build_humaneval_prompt_and_completion(row)
            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name}")

            full_input_ids, attention_mask, completion_start = _tokenize_pair(
                tokenizer,
                prompt,
                completion,
                device=device,
                max_length=max_length,
                mask_token_id=mask_token_id,
                min_completion_tokens=int(min_completion_tokens),
            )
            # diagnostics: how many completion tokens are actually present after truncation
            try:
                completion_lens.append(int(full_input_ids.size(1) - int(completion_start)))
            except Exception:
                pass

            # Build masked variants (same logic; seed no longer depends on layer_idx since we compute jointly)
            masked_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for prob_idx, prob in enumerate(mask_probs):
                for s in range(int(mask_samples_per_prob)):
                    gen = torch.Generator(device=device)
                    gen.manual_seed(_stable_int_seed(int(seed), int(row_idx), int(prob_idx), int(s)))
                    input_ids_masked, labels = _build_labels_and_masked_inputs_for_completion_span(
                        full_input_ids=full_input_ids,
                        completion_start=completion_start,
                        mask_token_id=mask_token_id,
                        mask_prob=float(prob),
                        generator=gen,
                    )
                    if (labels != -100).sum().item() <= 0:
                        continue
                    masked_batches.append((input_ids_masked, attention_mask, labels))

            if len(masked_batches) == 0:
                total_rows_skipped_no_variants += 1
                continue

            if debug_gate and (not did_debug_check):
                with torch.no_grad():
                    gate.alpha_flat = torch.ones(total_heads, device=device, dtype=torch.float32)
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits1 = model(masked_batches[0][0], attention_mask=masked_batches[0][1]).logits
                    else:
                        logits1 = model(masked_batches[0][0], attention_mask=masked_batches[0][1]).logits

                    alpha2 = torch.ones(total_heads, device=device, dtype=torch.float32)
                    alpha2[0] = 0.0
                    gate.alpha_flat = alpha2
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits2 = model(masked_batches[0][0], attention_mask=masked_batches[0][1]).logits
                    else:
                        logits2 = model(masked_batches[0][0], attention_mask=masked_batches[0][1]).logits

                    delta = (logits1.to(torch.float32) - logits2.to(torch.float32)).abs().mean().item()
                    if delta <= 0.0:
                        raise RuntimeError(
                            "Debug gate check failed: changing alpha did not change logits. "
                            "Hook may not be attached to the active o_proj modules."
                        )
                    print(f"[debug_gate] all_heads_joint mean|Δlogits|={delta:.6g}")
                    did_debug_check = True

            # Stack variants
            all_input_ids = torch.cat([x for (x, _, _) in masked_batches], dim=0)
            all_attn = torch.cat([m for (_, m, _) in masked_batches], dim=0)
            all_labels = torch.cat([y for (_, _, y) in masked_batches], dim=0)
            n_variants = int(all_input_ids.size(0))

            chunk = int(mask_batch_size)
            if chunk <= 0:
                chunk = n_variants

            # Random-path Joint IG (design fix):
            # We integrate gradients along a *vector-valued* path alpha(t) where each head can have a different schedule.
            # Correct discrete approximation: sum_k grad(alpha_k) * (alpha_k - alpha_{k-1}) elementwise.
            #
            # path_mode:
            # - diagonal: alpha_h(t) are all identical (backward compatible with old behavior)
            # - random_threshold: each head has a random threshold u_h; it stays near baseline before u_h then ramps to 1
            path_mode_ = str(path_mode)
            if path_mode_ not in ("random_threshold", "diagonal"):
                raise ValueError(f"Unsupported path_mode={path_mode_!r}. Expected 'random_threshold' or 'diagonal'.")

            ps = int(max(1, path_samples))
            ig_row_total = torch.zeros(total_heads, device=device, dtype=torch.float32)

            # For deterministic randomness, prefer the provided path_seed; fall back to `seed` if negative.
            base_path_seed = int(seed) if int(path_seed) < 0 else int(path_seed)

            for path_i in range(ps):
                # Sample u on CPU for deterministic behavior, then move to device.
                if path_mode_ == "random_threshold":
                    g_u = torch.Generator()
                    g_u.manual_seed(_stable_int_seed(int(base_path_seed), int(row_idx), int(path_i)))
                    u = torch.rand((total_heads,), generator=g_u, dtype=torch.float32).to(device)
                    # avoid division by zero when u==1
                    denom = torch.clamp(1.0 - u, min=1e-6)
                else:
                    u = None
                    denom = None

                ig_accum = torch.zeros(total_heads, device=device, dtype=torch.float32)
                alpha_prev = torch.full((total_heads,), fill_value=float(baseline_value), device=device, dtype=torch.float32)

                for k in range(1, ig_steps + 1):
                    t = float(k) / float(ig_steps)
                    if path_mode_ == "diagonal":
                        alpha_vals = float(baseline_value) + float(t) * float(1.0 - baseline_value)
                        alpha_now = torch.full(
                            (total_heads,), fill_value=float(alpha_vals), device=device, dtype=torch.float32
                        )
                    else:
                        # ramp in [0,1]: (t - u)/(1-u) clipped
                        t_t = torch.full((total_heads,), fill_value=float(t), device=device, dtype=torch.float32)
                        ramp = torch.clamp((t_t - u) / denom, min=0.0, max=1.0)
                        alpha_now = float(baseline_value) + ramp * float(1.0 - baseline_value)

                    delta_alpha = (alpha_now - alpha_prev).to(torch.float32)
                    alpha_prev = alpha_now

                    # Make alpha a leaf requiring grad (so .grad is populated)
                    alpha_flat = alpha_now.detach().clone().requires_grad_(True)
                    gate.alpha_flat = alpha_flat

                    model.zero_grad(set_to_none=True)

                    loss_weighted_sum = None
                    total_variants = 0
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            for start in range(0, n_variants, chunk):
                                end = min(start + chunk, n_variants)
                                logits = model(all_input_ids[start:end], attention_mask=all_attn[start:end]).logits
                                l = _masked_ce_answer_only_batch(logits, all_labels[start:end], normalize=loss_normalize)
                                bs = int(end - start)
                                total_variants += bs
                                lw = l * float(bs)
                                loss_weighted_sum = lw if loss_weighted_sum is None else (loss_weighted_sum + lw)
                    else:
                        for start in range(0, n_variants, chunk):
                            end = min(start + chunk, n_variants)
                            logits = model(all_input_ids[start:end], attention_mask=all_attn[start:end]).logits
                            l = _masked_ce_answer_only_batch(logits, all_labels[start:end], normalize=loss_normalize)
                            bs = int(end - start)
                            total_variants += bs
                            lw = l * float(bs)
                            loss_weighted_sum = lw if loss_weighted_sum is None else (loss_weighted_sum + lw)

                    if loss_weighted_sum is None or total_variants <= 0:
                        continue
                    loss = loss_weighted_sum / float(total_variants)

                    loss.backward()
                    if alpha_flat.grad is None:
                        raise RuntimeError("alpha_flat.grad is None; hook may not be applied correctly.")
                    ig_accum += alpha_flat.grad.detach().to(torch.float32) * delta_alpha

                ig_row_total += ig_accum / float(ps)

            ig_row = ig_row_total
            if int(debug_save_per_sample) > 0 and len(per_sample_ig) < int(debug_save_per_sample):
                # Save the raw per-sample IG contribution before postprocess aggregation.
                per_sample_ig.append(ig_row.detach().to(torch.float32).cpu().clone())
            if ig_postprocess == "abs":
                ig_sum_flat += ig_row.abs()
            elif ig_postprocess == "signed":
                ig_sum_flat += ig_row
            elif ig_postprocess == "relu":
                ig_sum_flat += torch.clamp(ig_row, min=0.0)
            else:
                raise ValueError(f"Unsupported ig_postprocess: {ig_postprocess}")

            total_items += 1
            if show_progress and tqdm is None and progress_update_every > 0:
                if (total_items % int(progress_update_every)) == 0:
                    print(f"[progress] all_heads_joint processed={total_items}/{len(dataset_rows)}")

        if total_items == 0:
            raise RuntimeError("No valid samples were processed; cannot compute attribution.")

        ig_mean_flat = ig_sum_flat / float(total_items)

        # Split back into per-layer tensors (keep on device, float32)
        out: Dict[int, torch.Tensor] = {}
        for li in layer_indices:
            off, nh = offsets[int(li)]
            out[int(li)] = ig_mean_flat[off : off + nh].clone()
        # Attach counters for caller-side diagnostics (via attributes).
        # (We keep return type unchanged; caller can read these via closure vars if needed.)
        compute_all_heads_joint_ig._diag = {  # type: ignore[attr-defined]
            "total_rows_seen": int(total_rows_seen),
            "total_items_processed": int(total_items),
            "total_rows_skipped_no_variants": int(total_rows_skipped_no_variants),
            "per_sample_ig": per_sample_ig,
            "completion_lens": completion_lens,
        }
        return out
    finally:
        gate.remove()


def _row_fingerprint(dataset: str, row: Dict[str, Any]) -> str:
    """
    Stable-ish fingerprint for a sampled row to detect duplicates / unchanged sample sets.
    """
    if dataset == "gsm8k":
        payload = {"q": row.get("question", ""), "a": row.get("answer", "")}
    else:
        payload = {
            "system_prompt": row.get("system_prompt", ""),
            "input": row.get("input", ""),
            "output": row.get("output", ""),
            "__nemotron_cat__": row.get("__nemotron_cat__", None),
        }
    s = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "nemotron", "mmlu", "humaneval"])
    p.add_argument("--dataset_config", type=str, default="main", help="gsm8k config or mmlu subject (e.g., 'all').")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--dataset_shuffle", action="store_true", default=False)
    p.add_argument("--samples_per_category", type=int, default=50)
    p.add_argument("--nemotron_pool_per_category", type=int, default=1000)
    p.add_argument("--nemotron_categories", type=str, default="code,math,science,chat,safety")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--data_seed", type=int, default=-1)
    p.add_argument("--mask_seed", type=int, default=-1)
    p.add_argument("--ig_steps", type=int, default=8)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--mask_probs", type=str, default="1.0")
    p.add_argument("--mask_samples_per_prob", type=int, default=1)
    p.add_argument("--loss_normalize", type=str, default="mean_masked", choices=["sum", "mean_masked"])
    p.add_argument("--ig_postprocess", type=str, default="abs", choices=["abs", "signed", "relu"])
    p.add_argument("--mask_batch_size", type=int, default=1)
    p.add_argument(
        "--min_completion_tokens",
        type=int,
        default=0,
        help=(
            "If >0, keep at least this many completion tokens by truncating prompt from the left. "
            "Recommended for nemotron where outputs can be extremely long."
        ),
    )
    p.add_argument(
        "--path_mode",
        type=str,
        default="random_threshold",
        choices=["random_threshold", "diagonal"],
        help="Integrated path mode for joint IG. random_threshold is a Shapley-like randomized path.",
    )
    p.add_argument(
        "--path_samples",
        type=int,
        default=4,
        help="Number of random paths to average per sample when path_mode=random_threshold.",
    )
    p.add_argument(
        "--path_seed",
        type=int,
        default=-1,
        help="Seed for random path generation. -1 means use mask_seed.",
    )
    p.add_argument(
        "--activation_checkpointing",
        type=str,
        default="none",
        choices=["none", "whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"],
    )
    p.add_argument("--no_progress", action="store_true", default=False)
    p.add_argument("--progress_update_every", type=int, default=10)
    p.add_argument("--baseline", type=str, default="zero", choices=["zero", "scalar"])
    p.add_argument("--baseline_scalar", type=float, default=0.3)
    p.add_argument(
        "--gsm8k_answer_mode",
        type=str,
        default="final",
        choices=["final", "final_hash", "full"],
        help="For GSM8K: supervision target for attribution (final / '#### <final>' / full answer+rationale).",
    )
    p.add_argument("--num_fewshot", type=int, default=0, help="For GSM8K: number of few-shot examples to prepend.")
    p.add_argument("--layer_start", type=int, default=0)
    p.add_argument("--layer_end", type=int, default=-1, help="Inclusive. -1 means last layer.")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp_bf16", action="store_true", default=True)
    p.add_argument("--debug_gate", action="store_true", default=False)
    p.add_argument("--debug_dump_samples", type=int, default=0, help="Print first N sampled rows' fingerprints.")
    p.add_argument(
        "--debug_save_per_sample",
        type=int,
        default=0,
        help="Save per-sample IG vectors for first N processed samples to output_dir/per_sample_ig.pt.",
    )
    args = p.parse_args()

    base_seed = int(args.seed)
    data_seed = base_seed if int(args.data_seed) < 0 else int(args.data_seed)
    mask_seed = base_seed if int(args.mask_seed) < 0 else int(args.mask_seed)

    torch.manual_seed(base_seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("LLaDA Loss-based Head Attribution (All-heads Joint IG)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}/{args.dataset_config} split={args.split} max_samples={args.max_samples}")
    print(f"Seeds: base={base_seed} data_seed={data_seed} mask_seed={mask_seed}")
    print(f"IG steps: {args.ig_steps}")
    print(f"Baseline: {args.baseline} (scalar={args.baseline_scalar})")
    print(f"Mask probs: {args.mask_probs} (samples/prob={args.mask_samples_per_prob}, loss_normalize={args.loss_normalize})")
    print(f"IG postprocess: {args.ig_postprocess} | mask_batch_size: {args.mask_batch_size}")
    print(f"Path: mode={args.path_mode} samples={args.path_samples} seed={args.path_seed}")
    print(f"Tokenization: min_completion_tokens={int(args.min_completion_tokens)} max_length={int(args.max_length)}")
    print(f"Activation checkpointing: {args.activation_checkpointing}")
    print(f"Progress: {'disabled' if bool(args.no_progress) else ('tqdm' if tqdm is not None else 'print')}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

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
        else:
            print("[warn] activation checkpointing requested but model.model.set_activation_checkpointing not found; ignoring.")

    for p_ in model.parameters():
        p_.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    gsm8k_fewshot_prefix = ""
    if args.dataset == "gsm8k":
        ds = load_dataset("gsm8k", args.dataset_config, split=args.split)
        if bool(args.dataset_shuffle):
            ds = ds.shuffle(seed=int(data_seed))
        rows = [ds[i] for i in range(min(args.max_samples, len(ds)))]
        if int(args.num_fewshot) > 0:
            print(f"Building {int(args.num_fewshot)}-shot prefix for GSM8K...")
            ds_train = load_dataset("gsm8k", args.dataset_config, split="train")
            g = torch.Generator().manual_seed(int(data_seed))
            idx = torch.randperm(len(ds_train), generator=g)[: int(args.num_fewshot)].tolist()
            parts = []
            for i in idx:
                r = ds_train[int(i)]
                parts.append(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n")
            gsm8k_fewshot_prefix = "".join(parts)
    elif args.dataset == "nemotron":
        cats = [c.strip() for c in args.nemotron_categories.split(",") if c.strip()]
        rows = []
        per_cat_counts: Dict[str, int] = {}
        pool_per_category = max(int(args.samples_per_category), int(args.nemotron_pool_per_category))
        for cat_idx, cat in enumerate(cats):
            print(f"Loading Nemotron split={cat} (streaming)...")
            stream = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=cat, streaming=True)
            buf = []
            for i, sample in enumerate(stream):
                # annotate category for downstream debugging
                if isinstance(sample, dict):
                    s2 = dict(sample)
                    s2["__nemotron_cat__"] = str(cat)
                    buf.append(s2)
                else:
                    buf.append(sample)
                if len(buf) >= int(pool_per_category):
                    break
            if len(buf) > 1:
                g = torch.Generator()
                g.manual_seed(_stable_int_seed(int(data_seed), int(cat_idx)))
                idx = torch.randperm(len(buf), generator=g).tolist()
                buf = [buf[j] for j in idx]
            take_n = min(int(args.samples_per_category), len(buf))
            rows.extend(buf[:take_n])
            per_cat_counts[str(cat)] = int(take_n)
        # IMPORTANT:
        # `--dataset_shuffle` should affect nemotron too; otherwise small max_samples will
        # always pick from the first category due to fixed concatenation order.
        if bool(args.dataset_shuffle) and len(rows) > 1:
            g_all = torch.Generator()
            # Mix in a constant to avoid accidental collisions with per-category shuffles.
            g_all.manual_seed(_stable_int_seed(int(data_seed), 999_001))
            perm = torch.randperm(len(rows), generator=g_all).tolist()
            rows = [rows[i] for i in perm]
        # Global cap after (optional) shuffle, consistent with gsm8k behavior.
        if len(rows) > int(args.max_samples):
            rows = rows[: int(args.max_samples)]
    elif args.dataset == "mmlu":
        subject = args.dataset_config if args.dataset_config != "main" else "all"
        print(f"Loading MMLU subject={subject}...")
        ds = load_dataset("cais/mmlu", subject, split=args.split)
        if bool(args.dataset_shuffle):
            ds = ds.shuffle(seed=int(data_seed))
        rows = [ds[i] for i in range(min(args.max_samples, len(ds)))]
    elif args.dataset == "humaneval":
        print("Loading HumanEval...")
        ds = load_dataset("openai_humaneval", split="test")
        if bool(args.dataset_shuffle):
            ds = ds.shuffle(seed=int(data_seed))
        rows = [ds[i] for i in range(min(args.max_samples, len(ds)))]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Data diagnostics (crucial for debugging "1 sample vs many samples" behavior)
    print("-" * 80)
    print("[data] sampling summary")
    print(f"[data] dataset={args.dataset} max_samples={int(args.max_samples)} dataset_shuffle={bool(args.dataset_shuffle)}")
    if args.dataset == "nemotron":
        print(f"[data] nemotron_categories={args.nemotron_categories}")
        print(f"[data] samples_per_category={int(args.samples_per_category)} nemotron_pool_per_category={int(args.nemotron_pool_per_category)} pool_per_category_used={int(pool_per_category)}")
        print(f"[data] per_category_counts={per_cat_counts}")
    print(f"[data] rows_loaded={len(rows)}")
    # Optional: print sample fingerprints for quick sanity checks (dup/unchanged set).
    if int(args.debug_dump_samples) > 0:
        n = min(int(args.debug_dump_samples), len(rows))
        print(f"[data] debug_dump_samples (first {n}):")
        for i in range(n):
            r = rows[i]
            if isinstance(r, dict):
                fp = _row_fingerprint(str(args.dataset), r)
                cat = r.get("__nemotron_cat__", None)
                in_len = len(str(r.get("input", "")))
                out_len = len(str(r.get("output", "")))
                print(f"  i={i:03d} fp={fp} cat={cat} input_len={in_len} output_len={out_len}")
            else:
                print(f"  i={i:03d} (non-dict row type={type(r)})")
    print("-" * 80)

    layers_all = _find_layers(model)
    n_layers = len(layers_all)
    n_heads_cfg = int(getattr(model.config, "n_heads", 0) or getattr(model.config, "num_attention_heads", 0) or 0)
    if n_heads_cfg <= 0:
        attn0, _ = _find_attn_and_oproj(layers_all[0])
        n_heads_cfg = _get_num_heads(attn0)

    layer_start = max(0, int(args.layer_start))
    layer_end = int(args.layer_end)
    if layer_end < 0:
        layer_end = n_layers - 1
    layer_end = min(layer_end, n_layers - 1)
    if layer_start > layer_end:
        raise ValueError(f"Invalid layer range: {layer_start}..{layer_end} (n_layers={n_layers})")

    baseline_value = 0.0 if args.baseline == "zero" else float(args.baseline_scalar)
    if not (0.0 <= baseline_value <= 1.0):
        raise ValueError(f"baseline_value must be in [0,1]. Got {baseline_value}")

    mask_probs = [float(x.strip()) for x in str(args.mask_probs).split(",") if x.strip()]
    if len(mask_probs) == 0:
        raise ValueError("--mask_probs cannot be empty.")
    for mp in mask_probs:
        if not (0.0 <= float(mp) <= 1.0):
            raise ValueError(f"mask_prob must be in [0,1]. Got {mp}")
    if int(args.mask_samples_per_prob) <= 0:
        raise ValueError("--mask_samples_per_prob must be > 0.")

    selected_layer_indices = list(range(layer_start, layer_end + 1))
    selected_layers = [layers_all[i] for i in selected_layer_indices]

    print(f"Selected layers: {layer_start}..{layer_end} (count={len(selected_layer_indices)})")

    importance_scores_device = compute_all_heads_joint_ig(
        model=model,
        layers=selected_layers,
        layer_indices=selected_layer_indices,
        tokenizer=tokenizer,
        dataset_rows=rows,
        device=device,
        ig_steps=int(args.ig_steps),
        baseline_value=float(baseline_value),
        max_length=int(args.max_length),
        num_heads_from_config=int(n_heads_cfg),
        use_amp_bf16=bool(args.use_amp_bf16 and device.type == "cuda"),
        dataset_name=str(args.dataset),
        mask_probs=mask_probs,
        mask_samples_per_prob=int(args.mask_samples_per_prob),
        loss_normalize=str(args.loss_normalize),
        seed=int(mask_seed),
        ig_postprocess=str(args.ig_postprocess),
        mask_batch_size=int(args.mask_batch_size),
        show_progress=(not bool(args.no_progress)),
        progress_update_every=int(args.progress_update_every),
        path_mode=str(args.path_mode),
        path_samples=int(args.path_samples),
        path_seed=int(args.path_seed),
        min_completion_tokens=int(args.min_completion_tokens),
        debug_gate=bool(args.debug_gate),
        debug_save_per_sample=int(args.debug_save_per_sample),
        gsm8k_answer_mode=str(args.gsm8k_answer_mode),
        gsm8k_fewshot_prefix=str(gsm8k_fewshot_prefix),
    )

    importance_scores: Dict[int, torch.Tensor] = {
        int(k): v.detach().to(torch.float32).cpu() for k, v in importance_scores_device.items()
    }

    all_vals = torch.cat([importance_scores[k] for k in sorted(importance_scores.keys())]).to(torch.float32)
    print(
        f"Joint head_scores: mean={all_vals.mean().item():.6f}, std={all_vals.std().item():.6f}, "
        f"min={all_vals.min().item():.6f}, max={all_vals.max().item():.6f}"
    )

    out = {
        "importance_scores": importance_scores,
        "metadata": {
            "method": "all_heads_joint_ig_diffusion_masked_ce_answer_only_multit",
            "model_path": args.model_path,
            "dataset": (
                f"gsm8k/{args.dataset_config}"
                if args.dataset == "gsm8k"
                else (f"mmlu/{args.dataset_config}" if args.dataset == "mmlu" else f"{args.dataset}")
            ),
            "split": args.split,
            "max_samples": int(args.max_samples),
            "rows_loaded": int(len(rows)),
            "seed": int(base_seed),
            "data_seed": int(data_seed),
            "mask_seed": int(mask_seed),
            "dataset_shuffle": bool(args.dataset_shuffle),
            "samples_per_category": int(args.samples_per_category) if args.dataset == "nemotron" else None,
            "nemotron_pool_per_category": int(args.nemotron_pool_per_category) if args.dataset == "nemotron" else None,
            "nemotron_per_category_counts": per_cat_counts if args.dataset == "nemotron" else None,
            "gsm8k_answer_mode": str(args.gsm8k_answer_mode) if args.dataset == "gsm8k" else None,
            "gsm8k_num_fewshot": int(args.num_fewshot) if args.dataset == "gsm8k" else 0,
            "ig_steps": int(args.ig_steps),
            "path_mode": str(args.path_mode),
            "path_samples": int(args.path_samples),
            "path_seed": int(mask_seed if int(args.path_seed) < 0 else int(args.path_seed)),
            "min_completion_tokens": int(args.min_completion_tokens),
            "mask_probs": mask_probs,
            "mask_samples_per_prob": int(args.mask_samples_per_prob),
            "loss_normalize": str(args.loss_normalize),
            "ig_postprocess": str(args.ig_postprocess),
            "mask_batch_size": int(args.mask_batch_size),
            "baseline": args.baseline,
            "baseline_value": float(baseline_value),
            "layer_range": [int(layer_start), int(layer_end)],
            "generated_at": datetime.now().isoformat(),
            "note": (
                "Joint IG on head gates α inserted at attention o_proj input for all selected layers at once. "
                "Objective matches LLaDA's mask-predictor behavior: within the completion/answer span, randomly mask a subset "
                "of tokens with mask_token_id at multiple mask probabilities (diffusion time steps) and average the loss. "
                "Cross-entropy is computed ONLY on masked completion/answer positions (labels != -100). "
                "For GSM8K, completion is the final answer after '####'. For Nemotron, completion is sample['output']."
            ),
        },
    }

    # Processing diagnostics (how many rows actually contributed to the mean)
    diag = getattr(compute_all_heads_joint_ig, "_diag", None)
    if isinstance(diag, dict):
        out["metadata"]["total_rows_seen"] = int(diag.get("total_rows_seen", 0))
        out["metadata"]["total_items_processed"] = int(diag.get("total_items_processed", 0))
        out["metadata"]["total_rows_skipped_no_variants"] = int(diag.get("total_rows_skipped_no_variants", 0))
        print(
            f"[data] processed summary: rows_seen={out['metadata']['total_rows_seen']} "
            f"items_used={out['metadata']['total_items_processed']} "
            f"skipped_no_variants={out['metadata']['total_rows_skipped_no_variants']}"
        )
        cl = diag.get("completion_lens", None)
        if isinstance(cl, list) and len(cl) > 0:
            cl_sorted = sorted(int(x) for x in cl)
            out["metadata"]["completion_len_tokens_min"] = int(cl_sorted[0])
            out["metadata"]["completion_len_tokens_med"] = int(cl_sorted[len(cl_sorted) // 2])
            out["metadata"]["completion_len_tokens_max"] = int(cl_sorted[-1])
            print(
                f"[data] completion_len_tokens: min={out['metadata']['completion_len_tokens_min']} "
                f"med={out['metadata']['completion_len_tokens_med']} "
                f"max={out['metadata']['completion_len_tokens_max']}"
            )
        # Save per-sample IG vectors if requested.
        if int(args.debug_save_per_sample) > 0:
            per = diag.get("per_sample_ig", None)
            if isinstance(per, list) and len(per) > 0:
                per_t = torch.stack(per, dim=0)  # (N, total_heads)
                save_path = os.path.join(args.output_dir, "per_sample_ig.pt")
                torch.save(
                    {
                        "per_sample_ig": per_t,
                        "note": "Per-sample IG vectors (pre postprocess aggregation; signed values).",
                    },
                    save_path,
                )
                print(f"[debug] Saved per-sample IG vectors to: {save_path} (shape={tuple(per_t.shape)})")

    out_path = os.path.join(args.output_dir, "head_importance.pt")
    torch.save(out, out_path)
    print(f"\n✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


