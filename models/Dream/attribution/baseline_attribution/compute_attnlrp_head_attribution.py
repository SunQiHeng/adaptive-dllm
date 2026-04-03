#!/usr/bin/env python3
"""
AttnLRP-inspired head attribution baseline for Dream.

This script reuses the same dataset construction and diffusion-style masked CE objective as
`loss_attribution/compute_loss_attribution_all_heads.py`, but replaces multi-step IG with a
single-backward relevance proxy on explicit per-head gates.

For each selected head gate alpha_h (initialized to 1), we compute:

    relevance_h = -(dL / d alpha_h) * alpha_h

where L is the masked completion denoising loss averaged across the sampled diffusion mask
variants. The leading minus sign makes larger positive scores correspond to heads that support
the correct target (i.e. increasing the gate would reduce loss).

This is not a literal reimplementation of the full AttnLRP rule set. It is an
AttnLRP-inspired internal head relevance baseline that is easy to compare against the existing
IG and future Shapley baselines while keeping the exact same attribution target and output
format.
"""

from __future__ import annotations

import argparse
import importlib.util
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

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import AutoTokenizer
from datasets import load_dataset

from models.Dream.core.modeling_dream import DreamModel
from models.attribution_utils import load_hf_rows, load_local_rows, normalize_dataset_name


def _should_use_tqdm(show_progress: bool) -> bool:
    return bool(show_progress and (tqdm is not None) and sys.stderr.isatty())


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    return mod


_BASE = _load_module(
    os.path.join(os.path.dirname(__file__), "../loss_attribution/compute_loss_attribution.py"),
    "_dream_loss_attr_layerwise_for_baseline",
)
_JOINT = _load_module(
    os.path.join(os.path.dirname(__file__), "../loss_attribution/compute_loss_attribution_all_heads.py"),
    "_dream_loss_attr_joint_for_baseline",
)

_stable_int_seed = _BASE._stable_int_seed
_find_layers = _BASE._find_layers
_find_attn_and_oproj = _BASE._find_attn_and_oproj
_get_num_heads = _BASE._get_num_heads
_get_mask_token_id = _BASE._get_mask_token_id
_dry_run_check_o_proj_shape = _BASE._dry_run_check_o_proj_shape
_build_gsm8k_prompt_and_completion = _BASE._build_gsm8k_prompt_and_completion
_build_nemotron_prompt_and_completion = _BASE._build_nemotron_prompt_and_completion
_build_mmlu_prompt_and_answer = _BASE._build_mmlu_prompt_and_answer
_build_humaneval_prompt_and_completion = _BASE._build_humaneval_prompt_and_completion
_tokenize_pair = _BASE._tokenize_pair
_build_labels_and_masked_inputs_for_completion_span = _BASE._build_labels_and_masked_inputs_for_completion_span
_masked_ce_answer_only_batch = _BASE._masked_ce_answer_only_batch
_apply_dream_logits_shift = _BASE._apply_dream_logits_shift
_take_seeded_rows = _JOINT._take_seeded_rows
_MultiOProjHeadGate = _JOINT._MultiOProjHeadGate


def _postprocess_relevance(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "signed":
        return x
    if mode == "abs":
        return x.abs()
    if mode == "relu":
        return torch.relu(x)
    raise ValueError(f"Unsupported relevance_postprocess: {mode}")


def compute_all_heads_joint_relevance(
    model: torch.nn.Module,
    layers: List[torch.nn.Module],
    layer_indices: List[int],
    tokenizer,
    dataset_rows: List[Dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    min_completion_tokens: int,
    num_heads_from_config: int,
    use_amp_bf16: bool,
    dataset_name: str,
    dataset_use_chat_template: bool,
    gsm8k_answer_mode: str,
    mask_probs: List[float],
    mask_samples_per_prob: int,
    loss_normalize: str,
    seed: int,
    relevance_postprocess: str,
    mask_batch_size: int,
    show_progress: bool,
    progress_update_every: int = 10,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    if len(layers) != len(layer_indices):
        raise ValueError("layers and layer_indices must have same length.")
    if len(layers) == 0:
        raise ValueError("No layers selected for attribution.")

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
        score_sum_flat = torch.zeros(total_heads, device=device, dtype=torch.float32)
        per_sample_scores: List[torch.Tensor] = []
        total_items = 0
        total_rows_seen = 0
        total_rows_skipped_no_completion = 0
        total_rows_skipped_no_variants = 0
        completion_lens: List[int] = []

        mask_token_id = _get_mask_token_id(model, tokenizer)

        iterator = enumerate(dataset_rows)
        use_tqdm = _should_use_tqdm(show_progress)
        if use_tqdm:
            iterator = tqdm(  # type: ignore[assignment]
                iterator,
                total=len(dataset_rows),
                desc="attnlrp_head",
                dynamic_ncols=True,
                leave=False,
            )

        for row_idx, row in iterator:
            total_rows_seen += 1

            if dataset_name in {"gsm8k", "minerva_math"}:
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
            elif dataset_name in {"mmlu", "cmmlu", "ceval-valid", "gpqa_main_n_shot"}:
                prompt, completion = _build_mmlu_prompt_and_answer(
                    row, tokenizer=tokenizer, use_chat_template=dataset_use_chat_template
                )
            elif dataset_name in {"humaneval", "mbpp"}:
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
                mask_token_id=mask_token_id,
                min_completion_tokens=int(min_completion_tokens),
            )
            completion_len = int(full_input_ids.size(1) - int(completion_start))
            if completion_len <= 0:
                total_rows_skipped_no_completion += 1
                continue
            completion_lens.append(completion_len)

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
                    n_masked = int((labels != -100).sum().item())
                    if n_masked <= 0:
                        continue
                    masked_batches.append((input_ids_masked, labels, attention_mask))

            if len(masked_batches) == 0:
                total_rows_skipped_no_variants += 1
                continue

            all_input_ids = torch.cat([x[0] for x in masked_batches], dim=0)
            all_labels = torch.cat([x[1] for x in masked_batches], dim=0)
            all_attn_1d = torch.cat([x[2] for x in masked_batches], dim=0)
            all_labels_tail = all_labels[:, 1:]

            n_variants = int(all_input_ids.size(0))
            chunk = n_variants if int(mask_batch_size) <= 0 else max(1, int(mask_batch_size))

            alpha_flat = torch.ones(total_heads, device=device, dtype=torch.float32, requires_grad=True)
            gate.alpha_flat = alpha_flat

            model.zero_grad(set_to_none=True)
            loss_weighted_sum = None
            total_variants = 0
            shift_nlk = max(1, completion_len)
            shift_trim = shift_nlk - 1

            if use_amp_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for start in range(0, n_variants, chunk):
                        end = min(start + chunk, n_variants)
                        raw = model(all_input_ids[start:end], num_logits_to_keep=shift_nlk).logits
                        logits = _apply_dream_logits_shift(raw, shift_trim)
                        l = _masked_ce_answer_only_batch(logits, all_labels_tail[start:end], normalize=loss_normalize)
                        bs = int(end - start)
                        total_variants += bs
                        lw = l * float(bs)
                        loss_weighted_sum = lw if loss_weighted_sum is None else (loss_weighted_sum + lw)
            else:
                for start in range(0, n_variants, chunk):
                    end = min(start + chunk, n_variants)
                    raw = model(all_input_ids[start:end], num_logits_to_keep=shift_nlk).logits
                    logits = _apply_dream_logits_shift(raw, shift_trim)
                    l = _masked_ce_answer_only_batch(logits, all_labels_tail[start:end], normalize=loss_normalize)
                    bs = int(end - start)
                    total_variants += bs
                    lw = l * float(bs)
                    loss_weighted_sum = lw if loss_weighted_sum is None else (loss_weighted_sum + lw)

            if loss_weighted_sum is None or total_variants <= 0:
                total_rows_skipped_no_variants += 1
                continue

            loss = loss_weighted_sum / float(total_variants)
            loss.backward()
            if alpha_flat.grad is None:
                raise RuntimeError("alpha_flat.grad is None; hook may not be applied correctly.")

            raw_score = -(alpha_flat.grad.detach().to(torch.float32) * alpha_flat.detach().to(torch.float32))
            score = _postprocess_relevance(raw_score, relevance_postprocess)
            score_sum_flat += score
            per_sample_scores.append(raw_score.detach().cpu())
            total_items += 1
            if show_progress and (not use_tqdm) and progress_update_every > 0:
                if (total_items % int(progress_update_every)) == 0 or total_items == len(dataset_rows):
                    print(f"[progress] attnlrp_head processed={total_items}/{len(dataset_rows)}")

        if total_items == 0:
            raise RuntimeError("No valid samples were processed. Check dataset/length/masking settings.")

        mean_flat = score_sum_flat / float(total_items)
        out: Dict[int, torch.Tensor] = {}
        for li in layer_indices:
            offset, n_heads = offsets[int(li)]
            out[int(li)] = mean_flat[offset : offset + n_heads].clone()

        diagnostics = {
            "total_rows_seen": int(total_rows_seen),
            "total_items_processed": int(total_items),
            "total_rows_skipped_no_completion": int(total_rows_skipped_no_completion),
            "total_rows_skipped_no_variants": int(total_rows_skipped_no_variants),
            "completion_len_tokens_min": int(min(completion_lens)) if completion_lens else 0,
            "completion_len_tokens_med": int(sorted(completion_lens)[len(completion_lens) // 2]) if completion_lens else 0,
            "completion_len_tokens_max": int(max(completion_lens)) if completion_lens else 0,
            "per_sample_scores": per_sample_scores,
        }
        return out, diagnostics
    finally:
        gate.remove()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="nemotron",
        choices=["gsm8k", "minerva_math", "nemotron", "mmlu", "cmmlu", "ceval-valid", "gpqa_main_n_shot", "humaneval", "mbpp"],
    )
    parser.add_argument("--dataset_config", type=str, default="main", help="gsm8k config or mmlu subject.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_path", type=str, default=None, help="Optional local .json/.jsonl file or directory to load attribution rows from.")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--samples_per_category", type=int, default=50)
    parser.add_argument("--nemotron_pool_per_category", type=int, default=1000)
    parser.add_argument("--nemotron_categories", type=str, default="code,math,science,chat,safety")
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--gsm8k_answer_mode", type=str, default="final", choices=["final", "final_hash", "full"])
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--mask_seed", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--min_completion_tokens", type=int, default=0)
    parser.add_argument("--mask_probs", type=str, default="1.0")
    parser.add_argument("--mask_samples_per_prob", type=int, default=1)
    parser.add_argument("--loss_normalize", type=str, default="mean_masked", choices=["sum", "mean_masked"])
    parser.add_argument("--relevance_postprocess", type=str, default="relu", choices=["abs", "signed", "relu"])
    parser.add_argument("--mask_batch_size", type=int, default=1)
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--progress_update_every", type=int, default=10)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--layer_start", type=int, default=0)
    parser.add_argument("--layer_end", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_amp_bf16", action="store_true")
    parser.add_argument("--save_per_sample", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seed = int(args.seed)
    data_seed = int(args.data_seed) if args.data_seed is not None else seed
    mask_seed = int(args.mask_seed) if args.mask_seed is not None else seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========================================================")
    print("Dream AttnLRP-style Head Attribution Baseline")
    print("========================================================")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"device={device}")
    print(f"model_path={args.model_path}")
    dataset_name = normalize_dataset_name(str(args.dataset))
    print(f"dataset={dataset_name} split={args.split} max_samples={args.max_samples}")
    print(f"data_path={args.data_path}")
    print(f"seed={seed} data_seed={data_seed} mask_seed={mask_seed}")
    print(f"max_length={args.max_length} min_completion_tokens={args.min_completion_tokens}")
    print(f"mask_probs={args.mask_probs} mask_samples_per_prob={args.mask_samples_per_prob}")
    print(f"loss_normalize={args.loss_normalize} relevance_postprocess={args.relevance_postprocess}")
    print(f"Progress: {'tqdm' if _should_use_tqdm(bool(args.show_progress)) else (f'print_every_{int(args.progress_update_every)}' if bool(args.show_progress) else 'disabled')}")
    print(f"gradient_checkpointing={bool(args.gradient_checkpointing)}")
    print("========================================================")

    model = DreamModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if bool(args.gradient_checkpointing):
        attn_do = float(getattr(getattr(model, "config", None), "attention_dropout", 0.0) or 0.0)
        preserve_rng_state = bool(attn_do == 0.0)
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass
        if hasattr(model, "model") and hasattr(model.model, "_gradient_checkpointing_func"):
            model.model._gradient_checkpointing_func = partial(
                checkpoint,
                use_reentrant=False,
                preserve_rng_state=preserve_rng_state,
            )
        if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing"):
            model.model.gradient_checkpointing = True
        model.train()
    else:
        model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.data_path:
        rows = load_local_rows(
            args.data_path,
            max_samples=int(args.max_samples),
            data_seed=int(data_seed),
            split=str(args.split),
            dataset_name=dataset_name,
        )
    elif dataset_name == "nemotron":
        cats = [c.strip() for c in str(args.nemotron_categories).split(",") if c.strip()]
        rows = []
        pool_per_category = max(int(args.samples_per_category), int(args.nemotron_pool_per_category))
        for cat_idx, cat in enumerate(cats):
            print(f"Loading Nemotron split={cat} (streaming)...")
            stream = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=cat, streaming=True)
            buf: List[Dict[str, Any]] = []
            for sample in stream:
                buf.append(sample)
                if len(buf) >= int(pool_per_category):
                    break
            if len(buf) > 1:
                g = torch.Generator()
                g.manual_seed(_stable_int_seed(int(data_seed), int(cat_idx)))
                idx = torch.randperm(len(buf), generator=g).tolist()
                buf = [buf[j] for j in idx]
            rows.extend(buf[: min(int(args.samples_per_category), len(buf))])
        if len(rows) > int(args.max_samples):
            rows = rows[: int(args.max_samples)]
    else:
        rows = load_hf_rows(
            dataset_name,
            dataset_config=str(args.dataset_config),
            split=str(args.split),
            max_samples=int(args.max_samples),
            data_seed=int(data_seed),
        )

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

    mask_probs = [float(x.strip()) for x in str(args.mask_probs).split(",") if x.strip()]
    if not mask_probs:
        raise ValueError("--mask_probs cannot be empty.")

    selected_layer_indices = list(range(layer_start, layer_end + 1))
    selected_layers = [layers_all[i] for i in selected_layer_indices]

    scores_device, diagnostics = compute_all_heads_joint_relevance(
        model=model,
        layers=selected_layers,
        layer_indices=selected_layer_indices,
        tokenizer=tokenizer,
        dataset_rows=rows,
        device=device,
        max_length=int(args.max_length),
        min_completion_tokens=int(args.min_completion_tokens),
        num_heads_from_config=int(n_heads_cfg),
        use_amp_bf16=bool(args.use_amp_bf16 and device.type == "cuda"),
        dataset_name=dataset_name,
        dataset_use_chat_template=bool(args.use_chat_template),
        gsm8k_answer_mode=str(args.gsm8k_answer_mode),
        mask_probs=mask_probs,
        mask_samples_per_prob=int(args.mask_samples_per_prob),
        loss_normalize=str(args.loss_normalize),
        seed=int(mask_seed),
        relevance_postprocess=str(args.relevance_postprocess),
        mask_batch_size=int(args.mask_batch_size),
        show_progress=bool(args.show_progress),
        progress_update_every=int(args.progress_update_every),
    )

    importance_scores = {int(k): v.detach().to(torch.float32).cpu() for k, v in scores_device.items()}
    all_vals = torch.cat([importance_scores[k] for k in sorted(importance_scores.keys())]).to(torch.float32)
    print(
        f"Head relevance scores: mean={all_vals.mean().item():.6f}, std={all_vals.std().item():.6f}, "
        f"min={all_vals.min().item():.6f}, max={all_vals.max().item():.6f}"
    )

    out = {
        "importance_scores": importance_scores,
        "metadata": {
            "method": "dream_attnlrp_style_head_relevance_diffusion_masked_ce_answer_only_multit",
            "model_path": args.model_path,
            "dataset": (
                f"{dataset_name}/{args.dataset_config}"
                if dataset_name in {"gsm8k", "mmlu", "cmmlu", "ceval-valid", "gpqa_main_n_shot", "mbpp"}
                else dataset_name
            ),
            "data_path": str(args.data_path) if args.data_path else None,
            "split": str(args.split),
            "max_samples": int(args.max_samples),
            "seed": int(seed),
            "data_seed": int(data_seed),
            "mask_seed": int(mask_seed),
            "use_chat_template": bool(args.use_chat_template),
            "gsm8k_answer_mode": str(args.gsm8k_answer_mode) if dataset_name in {"gsm8k", "minerva_math"} else None,
            "max_length": int(args.max_length),
            "min_completion_tokens": int(args.min_completion_tokens),
            "mask_probs": mask_probs,
            "mask_samples_per_prob": int(args.mask_samples_per_prob),
            "loss_normalize": str(args.loss_normalize),
            "relevance_postprocess": str(args.relevance_postprocess),
            "mask_batch_size": int(args.mask_batch_size),
            "layer_range": [int(layer_start), int(layer_end)],
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "generated_at": datetime.now().isoformat(),
            "note": (
                "AttnLRP-inspired single-backward head relevance baseline. A scalar head gate alpha is attached to each "
                "selected Dream attention head at o_proj input. Scores are computed as -(dL/d alpha)*alpha under the same "
                "diffusion-style masked completion CE objective used by the IG baseline, then aggregated across samples."
            ),
            "total_rows_seen": int(diagnostics.get("total_rows_seen", 0)),
            "total_items_processed": int(diagnostics.get("total_items_processed", 0)),
            "total_rows_skipped_no_completion": int(diagnostics.get("total_rows_skipped_no_completion", 0)),
            "total_rows_skipped_no_variants": int(diagnostics.get("total_rows_skipped_no_variants", 0)),
            "completion_len_tokens_min": int(diagnostics.get("completion_len_tokens_min", 0)),
            "completion_len_tokens_med": int(diagnostics.get("completion_len_tokens_med", 0)),
            "completion_len_tokens_max": int(diagnostics.get("completion_len_tokens_max", 0)),
        },
    }

    if bool(args.save_per_sample):
        per = diagnostics.get("per_sample_scores", None)
        if isinstance(per, list) and len(per) > 0:
            torch.save(
                {
                    "per_sample_scores": torch.stack(per, dim=0),
                    "note": "Raw signed per-sample head relevance vectors before postprocess aggregation.",
                },
                os.path.join(args.output_dir, "per_sample_relevance.pt"),
            )

    out_path = os.path.join(args.output_dir, "head_importance.pt")
    torch.save(out, out_path)
    print(f"\n✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()
