#!/usr/bin/env python3
"""
Exact leave-one-out (LOO) head attribution baseline for LLaDA.

For each selected head h, this computes:

    score_h = L(alpha_h=0, alpha_-h=1) - L(alpha=1)

under the same diffusion-style masked completion CE objective used by HeadIG. Positive
scores therefore mean that removing the head increases loss, so larger scores directly
mean "more important" for downstream sparse allocation and prune-most evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from transformers import AutoTokenizer
from datasets import load_dataset

from models.LLaDA.core.modeling import LLaDAModelLM
from models.LLaDA.core.configuration import ActivationCheckpointingStrategy
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
    "_llada_loss_attr_layerwise_for_loo",
)
_JOINT = _load_module(
    os.path.join(os.path.dirname(__file__), "../loss_attribution/compute_loss_attribution_all_heads.py"),
    "_llada_loss_attr_joint_for_loo",
)

_find_layers = _BASE._find_layers
_find_attn_and_oproj = _BASE._find_attn_and_oproj
_get_num_heads = _BASE._get_num_heads
_dry_run_check_o_proj_shape = _BASE._dry_run_check_o_proj_shape
_build_gsm8k_prompt_and_answer = _BASE._build_gsm8k_prompt_and_answer
_build_mmlu_prompt_and_answer = _BASE._build_mmlu_prompt_and_answer
_build_humaneval_prompt_and_completion = _BASE._build_humaneval_prompt_and_completion
_build_nemotron_prompt_and_completion = _BASE._build_nemotron_prompt_and_completion
_tokenize_pair = _BASE._tokenize_pair
_get_mask_token_id = _BASE._get_mask_token_id
_build_labels_and_masked_inputs_for_completion_span = _BASE._build_labels_and_masked_inputs_for_completion_span
_stable_int_seed = _BASE._stable_int_seed
_masked_ce_answer_only_batch = _BASE._masked_ce_answer_only_batch
_row_fingerprint = _JOINT._row_fingerprint
_MultiOProjHeadGate = _JOINT._MultiOProjHeadGate


def _postprocess_scores(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "signed":
        return x
    if mode == "abs":
        return x.abs()
    if mode == "relu":
        return torch.relu(x)
    raise ValueError(f"Unsupported score_postprocess: {mode}")


def _prepare_rows(
    tokenizer,
    dataset_rows: List[Dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    mask_token_id: int,
    dataset_name: str,
    mask_probs: List[float],
    mask_samples_per_prob: int,
    seed: int,
    min_completion_tokens: int,
    gsm8k_answer_mode: str,
    gsm8k_fewshot_prefix: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    total_rows_seen = 0
    total_rows_skipped_no_variants = 0
    completion_lens: List[int] = []

    for row_idx, row in enumerate(dataset_rows):
        total_rows_seen += 1
        if dataset_name in {"gsm8k", "minerva_math"}:
            prompt, completion = _build_gsm8k_prompt_and_answer(
                row["question"],
                row["answer"],
                answer_mode=str(gsm8k_answer_mode),
                fewshot_prefix=str(gsm8k_fewshot_prefix),
            )
        elif dataset_name == "nemotron":
            prompt, completion = _build_nemotron_prompt_and_completion(row)
        elif dataset_name in {"mmlu", "cmmlu", "ceval-valid", "gpqa_main_n_shot"}:
            prompt, completion = _build_mmlu_prompt_and_answer(row)
        elif dataset_name in {"humaneval", "mbpp"}:
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
        completion_lens.append(int(full_input_ids.size(1) - int(completion_start)))

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
                if int((labels != -100).sum().item()) <= 0:
                    continue
                masked_batches.append((input_ids_masked, labels, attention_mask))

        if len(masked_batches) == 0:
            total_rows_skipped_no_variants += 1
            continue

        prepared.append(
            {
                "input_ids": torch.cat([x[0] for x in masked_batches], dim=0),
                "labels": torch.cat([x[1] for x in masked_batches], dim=0),
                "attn": torch.cat([x[2] for x in masked_batches], dim=0),
            }
        )

    diag = {
        "total_rows_seen": int(total_rows_seen),
        "total_rows_skipped_no_variants": int(total_rows_skipped_no_variants),
        "completion_len_tokens_min": int(min(completion_lens)) if completion_lens else 0,
        "completion_len_tokens_med": int(sorted(completion_lens)[len(completion_lens) // 2]) if completion_lens else 0,
        "completion_len_tokens_max": int(max(completion_lens)) if completion_lens else 0,
    }
    return prepared, diag


def _evaluate_utility(
    model,
    gate,
    prepared_rows: List[Dict[str, Any]],
    alpha_flat: torch.Tensor,
    *,
    use_amp_bf16: bool,
    loss_normalize: str,
    mask_batch_size: int,
) -> float:
    gate.alpha_flat = alpha_flat
    row_utils: List[float] = []
    with torch.inference_mode():
        for item in prepared_rows:
            all_input_ids = item["input_ids"]
            all_labels = item["labels"]
            all_attn = item["attn"]
            n_variants = int(all_input_ids.size(0))
            chunk = n_variants if int(mask_batch_size) <= 0 else max(1, int(mask_batch_size))

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
            row_loss = loss_weighted_sum / float(total_variants)
            row_utils.append(float((-row_loss).detach().to(torch.float32).cpu().item()))

    if not row_utils:
        raise RuntimeError("Utility evaluation had no valid prepared rows.")
    return float(sum(row_utils) / len(row_utils))


def compute_leave_one_out(
    model: torch.nn.Module,
    layers: List[torch.nn.Module],
    layer_indices: List[int],
    tokenizer,
    dataset_rows: List[Dict[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    num_heads_from_config: int,
    use_amp_bf16: bool,
    dataset_name: str,
    mask_probs: List[float],
    mask_samples_per_prob: int,
    loss_normalize: str,
    seed: int,
    score_postprocess: str,
    mask_batch_size: int,
    show_progress: bool,
    min_completion_tokens: int,
    gsm8k_answer_mode: str,
    gsm8k_fewshot_prefix: str,
    progress_update_every: int = 10,
) -> Dict[int, torch.Tensor]:
    specs: List[Dict[str, Any]] = []
    offsets: Dict[int, Tuple[int, int]] = {}
    total_heads = 0

    for li, layer in zip(layer_indices, layers):
        attn, o_proj = _find_attn_and_oproj(layer)
        n_heads = _get_num_heads(attn, fallback_from_config=num_heads_from_config)
        hidden_size = int(o_proj.in_features)
        if hidden_size % n_heads != 0:
            raise ValueError(f"Layer {li}: hidden_size={hidden_size} not divisible by n_heads={n_heads}")
        head_dim = hidden_size // n_heads
        _dry_run_check_o_proj_shape(o_proj, hidden_size)
        offsets[int(li)] = (int(total_heads), int(n_heads))
        specs.append({"o_proj": o_proj, "n_heads": int(n_heads), "head_dim": int(head_dim), "offset": int(total_heads)})
        total_heads += int(n_heads)

    mask_token_id = _get_mask_token_id(model, tokenizer)
    prepared_rows, prep_diag = _prepare_rows(
        tokenizer,
        dataset_rows,
        device=device,
        max_length=max_length,
        mask_token_id=mask_token_id,
        dataset_name=dataset_name,
        mask_probs=mask_probs,
        mask_samples_per_prob=mask_samples_per_prob,
        seed=seed,
        min_completion_tokens=min_completion_tokens,
        gsm8k_answer_mode=gsm8k_answer_mode,
        gsm8k_fewshot_prefix=gsm8k_fewshot_prefix,
    )
    if not prepared_rows:
        raise RuntimeError("No valid prepared rows after dataset preprocessing.")

    gate = _MultiOProjHeadGate(specs)
    gate.install()
    try:
        alpha_full = torch.ones(total_heads, device=device, dtype=torch.float32)
        u_full = _evaluate_utility(
            model,
            gate,
            prepared_rows,
            alpha_full,
            use_amp_bf16=use_amp_bf16,
            loss_normalize=loss_normalize,
            mask_batch_size=mask_batch_size,
        )
        head_scores = torch.zeros(total_heads, dtype=torch.float32)

        iterator = range(total_heads)
        use_tqdm = _should_use_tqdm(show_progress)
        if use_tqdm:
            iterator = tqdm(iterator, total=total_heads, desc="leave_one_out", dynamic_ncols=True, leave=False)

        for step_idx, head_idx in enumerate(iterator, 1):
            alpha_without = alpha_full.clone()
            alpha_without[int(head_idx)] = 0.0
            u_without = _evaluate_utility(
                model,
                gate,
                prepared_rows,
                alpha_without,
                use_amp_bf16=use_amp_bf16,
                loss_normalize=loss_normalize,
                mask_batch_size=mask_batch_size,
            )
            head_scores[int(head_idx)] = float(u_full - u_without)
            if show_progress and (not use_tqdm) and progress_update_every > 0:
                if (step_idx % int(progress_update_every)) == 0 or step_idx == total_heads:
                    print(f"[progress] leave_one_out heads={step_idx}/{total_heads}")

        head_scores = _postprocess_scores(head_scores, score_postprocess)

        out: Dict[int, torch.Tensor] = {}
        for li in layer_indices:
            offset, n_heads = offsets[int(li)]
            out[int(li)] = head_scores[offset : offset + n_heads].clone()

        compute_leave_one_out._diag = {  # type: ignore[attr-defined]
            **prep_diag,
            "prepared_rows": int(len(prepared_rows)),
            "full_utility": float(u_full),
            "utility_evaluations": int(total_heads + 1),
            "total_heads_attributed": int(total_heads),
        }
        return out
    finally:
        gate.remove()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "minerva_math", "nemotron", "mmlu", "cmmlu", "ceval-valid", "gpqa_main_n_shot", "humaneval", "mbpp"],
    )
    p.add_argument("--dataset_config", type=str, default="main")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--data_path", type=str, default=None, help="Optional local .json/.jsonl file or directory to load attribution rows from.")
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--dataset_shuffle", action="store_true", default=False)
    p.add_argument("--samples_per_category", type=int, default=50)
    p.add_argument("--nemotron_pool_per_category", type=int, default=1000)
    p.add_argument("--nemotron_categories", type=str, default="code,math,science,chat,safety")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--data_seed", type=int, default=-1)
    p.add_argument("--mask_seed", type=int, default=-1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--mask_probs", type=str, default="0.15,0.3,0.5,0.7,0.9")
    p.add_argument("--mask_samples_per_prob", type=int, default=2)
    p.add_argument("--loss_normalize", type=str, default="mean_masked", choices=["sum", "mean_masked"])
    p.add_argument("--score_postprocess", type=str, default="signed", choices=["abs", "signed", "relu"])
    p.add_argument("--mask_batch_size", type=int, default=2)
    p.add_argument("--min_completion_tokens", type=int, default=0)
    p.add_argument(
        "--activation_checkpointing",
        type=str,
        default="none",
        choices=["none", "whole_layer", "one_in_two", "one_in_three", "one_in_four", "fine_grained"],
    )
    p.add_argument("--no_progress", action="store_true", default=False)
    p.add_argument("--progress_update_every", type=int, default=10)
    p.add_argument("--gsm8k_answer_mode", type=str, default="final_hash", choices=["final", "final_hash", "full"])
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--layer_start", type=int, default=0)
    p.add_argument("--layer_end", type=int, default=-1)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp_bf16", action="store_true", default=True)
    p.add_argument("--debug_dump_samples", type=int, default=0)
    args = p.parse_args()

    base_seed = int(args.seed)
    data_seed = base_seed if int(args.data_seed) < 0 else int(args.data_seed)
    mask_seed = base_seed if int(args.mask_seed) < 0 else int(args.mask_seed)

    torch.manual_seed(base_seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("LLaDA Leave-One-Out Head Attribution Baseline")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    dataset_name = normalize_dataset_name(str(args.dataset))
    print(f"Dataset: {dataset_name}/{args.dataset_config} split={args.split} max_samples={args.max_samples}")
    print(f"Local data_path: {args.data_path}")
    print(f"Seeds: base={base_seed} data_seed={data_seed} mask_seed={mask_seed}")
    print(f"Mask probs: {args.mask_probs} (samples/prob={args.mask_samples_per_prob})")
    print(f"Loss normalize: {args.loss_normalize} | score_postprocess={args.score_postprocess}")
    print(f"Progress: {'disabled' if bool(args.no_progress) else ('tqdm' if _should_use_tqdm(True) else f'print_every_{int(args.progress_update_every)}')}")
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

    for p_ in model.parameters():
        p_.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    gsm8k_fewshot_prefix = ""
    per_cat_counts: Dict[str, int] = {}
    if args.data_path:
        rows = load_local_rows(
            args.data_path,
            max_samples=int(args.max_samples),
            data_seed=int(data_seed),
            split=str(args.split),
            dataset_name=dataset_name,
        )
    elif dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", args.dataset_config, split=args.split)
        if bool(args.dataset_shuffle):
            ds = ds.shuffle(seed=int(data_seed))
        rows = [ds[i] for i in range(min(args.max_samples, len(ds)))]
        if int(args.num_fewshot) > 0:
            ds_train = load_dataset("gsm8k", args.dataset_config, split="train")
            g = torch.Generator().manual_seed(int(data_seed))
            idx = torch.randperm(len(ds_train), generator=g)[: int(args.num_fewshot)].tolist()
            parts = []
            for i in idx:
                r = ds_train[int(i)]
                parts.append(f"Question: {r['question']}\nAnswer: {r['answer']}\n\n")
            gsm8k_fewshot_prefix = "".join(parts)
    elif dataset_name == "nemotron":
        cats = [c.strip() for c in args.nemotron_categories.split(",") if c.strip()]
        rows = []
        per_cat_counts: Dict[str, int] = {}
        pool_per_category = max(int(args.samples_per_category), int(args.nemotron_pool_per_category))
        for cat_idx, cat in enumerate(cats):
            stream = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=cat, streaming=True)
            buf = []
            for sample in stream:
                s2 = dict(sample) if isinstance(sample, dict) else sample
                if isinstance(s2, dict):
                    s2["__nemotron_cat__"] = str(cat)
                buf.append(s2)
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
        if bool(args.dataset_shuffle) and len(rows) > 1:
            g_all = torch.Generator()
            g_all.manual_seed(_stable_int_seed(int(data_seed), 999_001))
            perm = torch.randperm(len(rows), generator=g_all).tolist()
            rows = [rows[i] for i in perm]
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

    if int(args.debug_dump_samples) > 0:
        n = min(int(args.debug_dump_samples), len(rows))
        print(f"[data] debug_dump_samples (first {n}):")
        for i in range(n):
            r = rows[i]
            if isinstance(r, dict):
                fp = _row_fingerprint(str(dataset_name), r)
                print(f"  i={i:03d} fp={fp}")

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

    mask_probs = [float(x.strip()) for x in str(args.mask_probs).split(",") if x.strip()]
    if len(mask_probs) == 0:
        raise ValueError("--mask_probs cannot be empty.")

    selected_layer_indices = list(range(layer_start, layer_end + 1))
    selected_layers = [layers_all[i] for i in selected_layer_indices]

    importance_scores_device = compute_leave_one_out(
        model=model,
        layers=selected_layers,
        layer_indices=selected_layer_indices,
        tokenizer=tokenizer,
        dataset_rows=rows,
        device=device,
        max_length=int(args.max_length),
        num_heads_from_config=int(n_heads_cfg),
        use_amp_bf16=bool(args.use_amp_bf16 and device.type == "cuda"),
        dataset_name=dataset_name,
        mask_probs=mask_probs,
        mask_samples_per_prob=int(args.mask_samples_per_prob),
        loss_normalize=str(args.loss_normalize),
        seed=int(mask_seed),
        score_postprocess=str(args.score_postprocess),
        mask_batch_size=int(args.mask_batch_size),
        show_progress=(not bool(args.no_progress)),
        min_completion_tokens=int(args.min_completion_tokens),
        gsm8k_answer_mode=str(args.gsm8k_answer_mode),
        gsm8k_fewshot_prefix=str(gsm8k_fewshot_prefix),
        progress_update_every=int(args.progress_update_every),
    )

    importance_scores = {int(k): v.detach().to(torch.float32).cpu() for k, v in importance_scores_device.items()}
    all_vals = torch.cat([importance_scores[k] for k in sorted(importance_scores.keys())]).to(torch.float32)
    print(
        f"LOO head scores: mean={all_vals.mean().item():.6f}, std={all_vals.std().item():.6f}, "
        f"min={all_vals.min().item():.6f}, max={all_vals.max().item():.6f}"
    )

    out = {
        "importance_scores": importance_scores,
        "metadata": {
            "method": "llada_leave_one_out_head_loss_delta_diffusion_masked_ce_answer_only_multit",
            "score_direction": "higher_is_more_important",
            "score_definition": "loss_without_head_minus_full_loss",
            "attribution_granularity": "attention_head",
            "model_path": args.model_path,
            "dataset": (
                f"{dataset_name}/{args.dataset_config}"
                if dataset_name in {"gsm8k", "mmlu", "cmmlu", "ceval-valid", "gpqa_main_n_shot", "mbpp"}
                else dataset_name
            ),
            "data_path": str(args.data_path) if args.data_path else None,
            "split": args.split,
            "max_samples": int(args.max_samples),
            "rows_loaded": int(len(rows)),
            "seed": int(base_seed),
            "data_seed": int(data_seed),
            "mask_seed": int(mask_seed),
            "dataset_shuffle": bool(args.dataset_shuffle),
            "samples_per_category": int(args.samples_per_category) if dataset_name == "nemotron" else None,
            "nemotron_pool_per_category": int(args.nemotron_pool_per_category) if dataset_name == "nemotron" else None,
            "nemotron_per_category_counts": per_cat_counts if dataset_name == "nemotron" else None,
            "gsm8k_answer_mode": str(args.gsm8k_answer_mode) if dataset_name in {"gsm8k", "minerva_math"} else None,
            "gsm8k_num_fewshot": int(args.num_fewshot) if dataset_name in {"gsm8k", "minerva_math"} else 0,
            "min_completion_tokens": int(args.min_completion_tokens),
            "max_length": int(args.max_length),
            "mask_probs": mask_probs,
            "mask_samples_per_prob": int(args.mask_samples_per_prob),
            "loss_normalize": str(args.loss_normalize),
            "score_postprocess": str(args.score_postprocess),
            "mask_batch_size": int(args.mask_batch_size),
            "layer_range": [int(layer_start), int(layer_end)],
            "generated_at": datetime.now().isoformat(),
            "note": (
                "Exact leave-one-out head attribution under LLaDA's diffusion-style masked completion CE objective. "
                "Each score is L(without head)-L(full), so positive/larger values directly indicate more helpful heads."
            ),
        },
    }

    diag = getattr(compute_leave_one_out, "_diag", None)
    if isinstance(diag, dict):
        out["metadata"].update(diag)

    out_path = os.path.join(args.output_dir, "head_importance.pt")
    torch.save(out, out_path)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
