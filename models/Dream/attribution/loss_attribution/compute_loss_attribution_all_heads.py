#!/usr/bin/env python3
"""
Loss-based head attribution for Dream (ALL-HEADS JOINT Integrated Gradients over per-head gates).

This mirrors `compute_loss_attribution.py` in this directory, but changes ONLY ONE thing:

- Original: compute IG **layer-wise** (one layer at a time), producing per-layer head scores.
- This script: compute IG **jointly across all selected layers at once**, by attaching head gates
  to every selected layer's attention `o_proj` input, and using one flattened gate vector
  `alpha_flat` that covers all (layer, head).

Everything else (dataset prompt/completion building, diffusion-style masking, CE objective,
`num_logits_to_keep` memory optimization, IG path, postprocess, output format) is kept the same.

Output:
  head_importance.pt
    {
      "importance_scores": {layer_idx: tensor[n_heads]},
      "metadata": {...}
    }
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

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

# -----------------------------------------------------------------------------
# IMPORTANT:
# We intentionally DO NOT import via `models.Dream.attribution.*` because that package's
# `__init__.py` imports optional modules that may not exist in this repo snapshot,
# which would break running this script (even for --help).
#
# Instead, we load the layer-wise script in THIS directory directly by file path,
# and reuse its helper functions to keep logic identical.
# -----------------------------------------------------------------------------
import importlib.util

_BASE_PATH = os.path.join(os.path.dirname(__file__), "compute_loss_attribution.py")
_spec = importlib.util.spec_from_file_location("_dream_loss_attr_layerwise", _BASE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load base module spec from: {_BASE_PATH}")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)  # type: ignore[arg-type]

_stable_int_seed = _base._stable_int_seed
_find_layers = _base._find_layers
_find_attn_and_oproj = _base._find_attn_and_oproj
_get_num_heads = _base._get_num_heads
_get_mask_token_id = _base._get_mask_token_id
_dry_run_check_o_proj_shape = _base._dry_run_check_o_proj_shape
_build_gsm8k_prompt_and_completion = _base._build_gsm8k_prompt_and_completion
_build_nemotron_prompt_and_completion = _base._build_nemotron_prompt_and_completion
_build_mmlu_prompt_and_answer = _base._build_mmlu_prompt_and_answer
_build_humaneval_prompt_and_completion = _base._build_humaneval_prompt_and_completion
_tokenize_pair = _base._tokenize_pair
_build_labels_and_masked_inputs_for_completion_span = _base._build_labels_and_masked_inputs_for_completion_span
_masked_ce_answer_only_batch = _base._masked_ce_answer_only_batch


class _MultiOProjHeadGate:
    """
    Register forward_pre_hook on multiple Dream attention o_proj modules.
    Each hook applies per-head scaling on the o_proj input:
      x[..., h, :] <- alpha_flat[offset + h] * x[..., h, :]
    """

    def __init__(self, specs: List[Dict[str, Any]]):
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
    dataset_use_chat_template: bool,
    gsm8k_answer_mode: str,
    mask_probs: List[float],
    mask_samples_per_prob: int,
    loss_normalize: str,
    seed: int,
    ig_postprocess: str,
    mask_batch_size: int,
    show_progress: bool,
    progress_update_every: int,
    path_mode: str = "diagonal",
    path_samples: int = 1,
    path_seed: int = -1,
    debug_gate: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Joint IG over all (layer, head) gates at once (Dream).
    Returns dict[layer_idx] = tensor[n_heads] (float32 on device).
    """
    if len(layers) != len(layer_indices):
        raise ValueError("layers and layer_indices must have same length.")
    if len(layers) == 0:
        raise ValueError("No layers selected for attribution.")

    # Build hook specs and flattened indexing
    specs: List[Dict[str, Any]] = []
    offsets: Dict[int, Tuple[int, int]] = {}
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
        scale = float(1.0 - baseline_value)

        mask_token_id = _get_mask_token_id(model, tokenizer)
        did_debug_check = False

        iterator = enumerate(dataset_rows)
        # Keep Dream's behavior: avoid tqdm in nohup/non-tty logs
        use_tqdm = bool(show_progress and (tqdm is not None) and sys.stderr.isatty())
        if use_tqdm:
            iterator = tqdm(  # type: ignore[assignment]
                iterator,
                total=len(dataset_rows),
                desc="all_heads_joint",
                dynamic_ncols=True,
                leave=False,
            )

        for row_idx, row in iterator:
            if dataset_name == "gsm8k":
                prompt, completion = _build_gsm8k_prompt_and_completion(
                    row["question"],
                    row["answer"],
                    tokenizer=tokenizer,
                    use_chat_template=dataset_use_chat_template,
                    answer_mode=gsm8k_answer_mode,
                )
            elif dataset_name == "nemotron":
                prompt, completion = _build_nemotron_prompt_and_completion(
                    row, tokenizer=tokenizer, use_chat_template=dataset_use_chat_template
                )
            elif dataset_name == "mmlu":
                prompt, completion = _build_mmlu_prompt_and_answer(
                    row, tokenizer=tokenizer, use_chat_template=dataset_use_chat_template
                )
            elif dataset_name == "humaneval":
                prompt, completion = _build_humaneval_prompt_and_completion(
                    row, tokenizer=tokenizer, use_chat_template=dataset_use_chat_template
                )
            else:
                raise ValueError(f"Unsupported dataset_name: {dataset_name}")

            full_input_ids, attention_mask, completion_start = _tokenize_pair(
                tokenizer,
                prompt,
                completion,
                device=device,
                max_length=max_length,
            )

            # Preserve Dream optimization: only compute logits for completion span (at the end)
            num_logits_to_keep = int(full_input_ids.size(1) - int(completion_start))
            if num_logits_to_keep <= 0:
                continue

            masked_batches: List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]] = []
            for prob_idx, prob in enumerate(mask_probs):
                for s in range(int(mask_samples_per_prob)):
                    # In the layer-wise version, mask seeding included layer_idx.
                    # For joint attribution, we still want determinism; we drop layer_idx but keep other parts.
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
                continue

            if debug_gate and (not did_debug_check):
                with torch.no_grad():
                    gate.alpha_flat = torch.ones(total_heads, device=device, dtype=torch.float32)
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits1 = model(masked_batches[0][0], num_logits_to_keep=num_logits_to_keep).logits
                    else:
                        logits1 = model(masked_batches[0][0], num_logits_to_keep=num_logits_to_keep).logits

                    alpha2 = torch.ones(total_heads, device=device, dtype=torch.float32)
                    alpha2[0] = 0.0
                    gate.alpha_flat = alpha2
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits2 = model(masked_batches[0][0], num_logits_to_keep=num_logits_to_keep).logits
                    else:
                        logits2 = model(masked_batches[0][0], num_logits_to_keep=num_logits_to_keep).logits

                    delta = (logits1.to(torch.float32) - logits2.to(torch.float32)).abs().mean().item()
                    if delta <= 0.0:
                        raise RuntimeError(
                            "Debug gate check failed: changing alpha did not change logits. "
                            "Hook may not be attached to the active o_proj modules."
                        )
                    print(f"[debug_gate] all_heads_joint mean|Δlogits|={delta:.6g}")
                    did_debug_check = True

            # Pre-stack variants for efficiency
            all_input_ids = torch.cat([x for (x, _, _) in masked_batches], dim=0)
            all_labels = torch.cat([y for (_, _, y) in masked_batches], dim=0)
            n_variants = int(all_input_ids.size(0))
            # Align labels with logits when we use `num_logits_to_keep` (DreamModel returns last-K logits).
            all_labels_tail = all_labels[:, -num_logits_to_keep:]

            chunk = int(mask_batch_size)
            if chunk <= 0:
                chunk = n_variants

            path_mode_ = str(path_mode)
            ps = int(max(1, path_samples))
            ig_row_total = torch.zeros(total_heads, device=device, dtype=torch.float32)
            base_path_seed = int(seed) if int(path_seed) < 0 else int(path_seed)

            for path_i in range(ps):
                if path_mode_ == "random_threshold":
                    g_u = torch.Generator()
                    g_u.manual_seed(_stable_int_seed(int(base_path_seed), int(row_idx), int(path_i)))
                    u = torch.rand((total_heads,), generator=g_u, dtype=torch.float32).to(device)
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
                        alpha_now = torch.full((total_heads,), fill_value=float(alpha_vals), device=device, dtype=torch.float32)
                    else:
                        t_t = torch.full((total_heads,), fill_value=float(t), device=device, dtype=torch.float32)
                        ramp = torch.clamp((t_t - u) / denom, min=0.0, max=1.0)
                        alpha_now = float(baseline_value) + ramp * float(1.0 - baseline_value)

                    delta_alpha = (alpha_now - alpha_prev).to(torch.float32)
                    alpha_prev = alpha_now

                    alpha_flat = alpha_now.detach().clone().requires_grad_(True)
                    gate.alpha_flat = alpha_flat

                    model.zero_grad(set_to_none=True)
                    loss_weighted_sum = None
                    total_variants = 0
                    if use_amp_bf16:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            for start in range(0, n_variants, chunk):
                                end = min(start + chunk, n_variants)
                                logits = model(all_input_ids[start:end], num_logits_to_keep=num_logits_to_keep).logits
                                l = _masked_ce_answer_only_batch(logits, all_labels_tail[start:end], normalize=loss_normalize)
                                bs = int(end - start)
                                total_variants += bs
                                lw = l * float(bs)
                                loss_weighted_sum = lw if loss_weighted_sum is None else (loss_weighted_sum + lw)
                    else:
                        for start in range(0, n_variants, chunk):
                            end = min(start + chunk, n_variants)
                            logits = model(all_input_ids[start:end], num_logits_to_keep=num_logits_to_keep).logits
                            l = _masked_ce_answer_only_batch(logits, all_labels_tail[start:end], normalize=loss_normalize)
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
            if ig_postprocess == "abs":
                ig_sum_flat += ig_row.abs()
            elif ig_postprocess == "signed":
                ig_sum_flat += ig_row
            elif ig_postprocess == "relu":
                ig_sum_flat += torch.clamp(ig_row, min=0.0)
            else:
                raise ValueError(f"Unsupported ig_postprocess: {ig_postprocess}")

            total_items += 1
            if show_progress and (not use_tqdm) and progress_update_every > 0:
                if (total_items % int(progress_update_every)) == 0:
                    print(f"[progress] all_heads_joint processed={total_items}/{len(dataset_rows)}")

        if total_items == 0:
            raise RuntimeError("No valid samples were processed; cannot compute attribution.")

        ig_mean_flat = ig_sum_flat / float(total_items)
        out: Dict[int, torch.Tensor] = {}
        for li in layer_indices:
            off, nh = offsets[int(li)]
            out[int(li)] = ig_mean_flat[off : off + nh].clone()
        return out
    finally:
        gate.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="nemotron", choices=["gsm8k", "nemotron", "mmlu", "humaneval"])
    parser.add_argument("--dataset_config", type=str, default="main", help="gsm8k config or mmlu subject (e.g., 'all').")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--samples_per_category", type=int, default=50)
    parser.add_argument("--nemotron_pool_per_category", type=int, default=1000)
    parser.add_argument("--nemotron_categories", type=str, default="code,math,science,chat,safety")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument(
        "--gsm8k_answer_mode",
        type=str,
        default="final",
        choices=["final", "final_hash", "full"],
        help="gsm8k only: 'final' uses answer after ####, 'final_hash' uses '#### <final>', 'full' uses full text.",
    )

    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--mask_seed", type=int, default=None)
    parser.add_argument("--ig_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--mask_probs", type=str, default="1.0")
    parser.add_argument("--mask_samples_per_prob", type=int, default=1)
    parser.add_argument("--loss_normalize", type=str, default="mean_masked", choices=["sum", "mean_masked"])
    parser.add_argument("--ig_postprocess", type=str, default="abs", choices=["abs", "signed", "relu"])
    parser.add_argument("--mask_batch_size", type=int, default=1)
    parser.add_argument(
        "--path_mode",
        type=str,
        default="diagonal",
        choices=["random_threshold", "diagonal"],
        help="Integrated path mode for joint IG. random_threshold is a Shapley-like randomized path.",
    )
    parser.add_argument(
        "--path_samples",
        type=int,
        default=1,
        help="Number of random paths to average per sample when path_mode=random_threshold.",
    )
    parser.add_argument(
        "--path_seed",
        type=int,
        default=-1,
        help="Seed for random path generation. -1 means use mask_seed.",
    )

    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--progress_update_every", type=int, default=20)
    parser.add_argument("--debug_gate", action="store_true")

    # Keep arg for compatibility with runner; joint version still benefits from model checkpointing
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable Dream gradient checkpointing (will switch model to train() during attribution to activate).",
    )

    parser.add_argument("--baseline", type=str, default="zero", choices=["zero", "scalar"])
    parser.add_argument("--baseline_scalar", type=float, default=0.3)
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_amp_bf16", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seed = int(args.seed)
    data_seed = int(args.data_seed) if args.data_seed is not None else seed
    mask_seed = int(args.mask_seed) if args.mask_seed is not None else seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========================================================")
    print("Dream Loss Attribution (All-heads Joint Gate IG)")
    print("========================================================")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"device={device}")
    print(f"model_path={args.model_path}")
    print(f"dataset={args.dataset} split={args.split} max_samples={args.max_samples}")
    print(f"seed={seed} data_seed={data_seed} mask_seed={mask_seed}")
    print(f"ig_steps={args.ig_steps} max_length={args.max_length}")
    print(f"mask_probs={args.mask_probs} mask_samples_per_prob={args.mask_samples_per_prob} loss_normalize={args.loss_normalize}")
    print(f"ig_postprocess={args.ig_postprocess} mask_batch_size={args.mask_batch_size}")
    print(f"use_chat_template={bool(args.use_chat_template)}")
    if str(args.dataset) == "gsm8k":
        print(f"gsm8k_answer_mode={str(args.gsm8k_answer_mode)}")
    print(f"gradient_checkpointing={bool(args.gradient_checkpointing)}")
    print("========================================================")

    # Load model
    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Activate checkpointing.
    #
    # IMPORTANT:
    # DreamBaseModel only applies checkpointing when `self.gradient_checkpointing and self.training` is True.
    # We enable it and switch to train() so the checkpoint path is used.
    #
    # To avoid the "inputs must require grad" pitfall (common when model params are frozen and we only
    # backprop through gate tensors), we force non-reentrant checkpointing (use_reentrant=False),
    # matching the layer-wise attribution script.
    if bool(args.gradient_checkpointing):
        attn_do = float(getattr(getattr(model, "config", None), "attention_dropout", 0.0) or 0.0)
        preserve_rng_state = bool(attn_do == 0.0)

        # Try HF helper first (newer transformers supports gradient_checkpointing_kwargs).
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # Older transformers: enable default and then override the checkpoint function on the base model.
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        # Ensure base model uses non-reentrant checkpoint.
        if hasattr(model, "model") and hasattr(model.model, "_gradient_checkpointing_func"):
            model.model._gradient_checkpointing_func = partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=preserve_rng_state,
            )
        if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing"):
            model.model.gradient_checkpointing = True

        if attn_do > 0.0:
            print(
                f"[warn] gradient_checkpointing requires train() in Dream; attention_dropout={attn_do} > 0 "
                f"may introduce stochasticity. Consider setting attention_dropout=0 for determinism."
            )

        model.train()
    else:
        model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load dataset (same as original script)
    if str(args.dataset) == "gsm8k":
        ds = load_dataset("gsm8k", args.dataset_config, split=args.split)
        rows = [ds[i] for i in range(min(int(args.max_samples), len(ds)))]
    elif str(args.dataset) == "nemotron":
        cats = [c.strip() for c in str(args.nemotron_categories).split(",") if c.strip()]
        rows: List[Dict[str, Any]] = []
        pool_per_category = max(int(args.samples_per_category), int(args.nemotron_pool_per_category))
        for cat_idx, cat in enumerate(cats):
            print(f"Loading Nemotron split={cat} (streaming)...")
            stream = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=cat, streaming=True)
            buf: List[Dict[str, Any]] = []
            for i, sample in enumerate(stream):
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
        if len(rows) > int(args.max_samples):
            rows = rows[: int(args.max_samples)]
    elif str(args.dataset) == "mmlu":
        subject = args.dataset_config if str(args.dataset_config) != "main" else "all"
        print(f"Loading MMLU subject={subject}...")
        ds = load_dataset("cais/mmlu", subject, split=args.split)
        rows = [ds[i] for i in range(min(int(args.max_samples), len(ds)))]
    elif str(args.dataset) == "humaneval":
        print("Loading HumanEval...")
        ds = load_dataset("openai_humaneval", split="test")
        rows = [ds[i] for i in range(min(int(args.max_samples), len(ds)))]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    layers_all = _find_layers(model)
    n_layers = len(layers_all)
    n_heads_cfg = int(getattr(model.config, "num_attention_heads", 0) or getattr(model.config, "n_heads", 0) or 0)
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

    baseline_value = 0.0 if str(args.baseline) == "zero" else float(args.baseline_scalar)
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

    scores_device = compute_all_heads_joint_ig(
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
        dataset_use_chat_template=bool(args.use_chat_template),
        gsm8k_answer_mode=str(args.gsm8k_answer_mode),
        mask_probs=mask_probs,
        mask_samples_per_prob=int(args.mask_samples_per_prob),
        loss_normalize=str(args.loss_normalize),
        seed=int(mask_seed),
        ig_postprocess=str(args.ig_postprocess),
        mask_batch_size=int(args.mask_batch_size),
        show_progress=bool(args.show_progress),
        progress_update_every=int(args.progress_update_every),
        path_mode=str(args.path_mode),
        path_samples=int(args.path_samples),
        path_seed=int(args.path_seed),
        debug_gate=bool(args.debug_gate),
    )

    importance_scores: Dict[int, torch.Tensor] = {
        int(k): v.detach().to(torch.float32).cpu() for k, v in scores_device.items()
    }
    all_vals = torch.cat([importance_scores[k] for k in sorted(importance_scores.keys())]).to(torch.float32)
    print(
        f"Joint head_scores: mean={all_vals.mean().item():.6f}, std={all_vals.std().item():.6f}, "
        f"min={all_vals.min().item():.6f}, max={all_vals.max().item():.6f}"
    )

    out = {
        "importance_scores": importance_scores,
        "metadata": {
            "method": "dream_all_heads_joint_ig_diffusion_masked_ce_answer_only_multit",
            "model_path": args.model_path,
            "dataset": (
                f"gsm8k/{args.dataset_config}"
                if str(args.dataset) == "gsm8k"
                else (f"mmlu/{args.dataset_config}" if str(args.dataset) == "mmlu" else str(args.dataset))
            ),
            "split": str(args.split),
            "max_samples": int(args.max_samples),
            "seed": int(seed),
            "data_seed": int(data_seed),
            "mask_seed": int(mask_seed),
            "use_chat_template": bool(args.use_chat_template),
            "gsm8k_answer_mode": str(args.gsm8k_answer_mode) if str(args.dataset) == "gsm8k" else None,
            "ig_steps": int(args.ig_steps),
            "path_mode": str(args.path_mode),
            "path_samples": int(args.path_samples),
            "path_seed": int(mask_seed if int(args.path_seed) < 0 else int(args.path_seed)),
            "mask_probs": mask_probs,
            "mask_samples_per_prob": int(args.mask_samples_per_prob),
            "loss_normalize": str(args.loss_normalize),
            "ig_postprocess": str(args.ig_postprocess),
            "mask_batch_size": int(args.mask_batch_size),
            "baseline": str(args.baseline),
            "baseline_value": float(baseline_value),
            "layer_range": [int(layer_start), int(layer_end)],
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "generated_at": datetime.now().isoformat(),
            "note": (
                "Joint IG on head gates α inserted at Dream attention o_proj input for all selected layers at once. "
                "Uses DreamModel.forward(num_logits_to_keep=completion_len) to save memory; "
                "loss is CE on masked completion positions only, averaged across diffusion mask_probs and MC samples."
            ),
        },
    }

    out_path = os.path.join(args.output_dir, "head_importance.pt")
    torch.save(out, out_path)
    print(f"\n✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


