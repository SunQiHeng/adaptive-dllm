#!/usr/bin/env python3
"""
Minimal RULER-style evaluation runner for LLaDA.

Why this exists:
- `eval_llada.py` relies on lm-eval-harness task registry. RULER is often not bundled there.
- This script evaluates "prompt -> short answer" datasets (JSONL or HuggingFace datasets)
  with an adjustable max context length (len_k * 1024 tokens, approximate).

Expected example schema (best-effort):
- prompt/input: one of ["prompt", "input", "question"]
- answer/output: one of ["answer", "output", "target", "label"]
OR a pair:
- context + query: ["context"] + ["query"|"question"]

Metrics:
- exact_match (strict, after strip)
- normalized_exact_match (lowercase + remove punctuation + collapse whitespace)
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer

# Reuse the exact same model wrapper + generation functions as lm-eval runner
from evaluation.llada.eval_llada import LLaDAEvalHarness  # type: ignore


_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _first_line(s: str) -> str:
    if not s:
        return ""
    return s.splitlines()[0].strip()


def _first_token(norm_s: str) -> str:
    norm_s = norm_s.strip()
    if not norm_s:
        return ""
    return norm_s.split(" ", 1)[0]


@dataclass
class Example:
    idx: int
    prompt: str
    answers: List[str]
    meta: Dict[str, Any]


def _to_str_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, (int, float)):
        return [str(x)]
    if isinstance(x, list):
        out: List[str] = []
        for v in x:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, (int, float)):
                out.append(str(v))
        return out
    return []


def _extract_prompt_answer(obj: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
    # Prompt candidates
    if "input" in obj and isinstance(obj["input"], str):
        prompt = obj["input"]
    elif "prompt" in obj and isinstance(obj["prompt"], str):
        prompt = obj["prompt"]
    elif "question" in obj and isinstance(obj["question"], str):
        prompt = obj["question"]
    elif "context" in obj and isinstance(obj["context"], str):
        q = obj.get("query") or obj.get("question") or ""
        if not isinstance(q, str):
            q = ""
        prompt = obj["context"] + ("\n\n" + q if q else "")
    else:
        raise ValueError(f"Cannot infer prompt from keys: {sorted(obj.keys())}")

    # Answer candidates (RULER HF typically uses `answers`: List[str])
    answers: List[str] = []
    for k in ("answers", "answer", "output", "target", "label"):
        answers = _to_str_list(obj.get(k))
        if answers:
            break
    if not answers:
        raise ValueError(f"Cannot infer answer(s) from keys: {sorted(obj.keys())}")

    meta = {
        k: v
        for k, v in obj.items()
        if k
        not in {
            "prompt",
            "input",
            "question",
            "context",
            "query",
            "answers",
            "answer",
            "output",
            "target",
            "label",
        }
    }
    return prompt, answers, meta


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resolve_data_paths(data_path: str, len_k: int) -> List[str]:
    if os.path.isfile(data_path):
        return [data_path]
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")

    # Prefer files that mention the target length, e.g. "*4k*.jsonl"
    candidates = []
    for root, _, files in os.walk(data_path):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            candidates.append(os.path.join(root, fn))

    if not candidates:
        raise FileNotFoundError(f"No .jsonl files found under directory: {data_path}")

    key = f"{len_k}k"
    preferred = [p for p in candidates if key in os.path.basename(p).lower()]
    return preferred or candidates


def _load_examples(
    *,
    data_path: Optional[str],
    hf_dataset: Optional[str],
    hf_config: Optional[str],
    split: str,
    len_k: int,
    limit: Optional[int],
) -> List[Example]:
    raw: List[Dict[str, Any]] = []
    if data_path:
        paths = _resolve_data_paths(data_path, len_k)
        for p in paths:
            raw.extend(list(_iter_jsonl(p)))
    else:
        if not hf_dataset:
            raise ValueError("Either --data_path or --hf_dataset must be provided.")
        from datasets import load_dataset  # local import

        ds = load_dataset(hf_dataset, hf_config, split=split)
        raw = [dict(x) for x in ds]

    exs: List[Example] = []
    for i, obj in enumerate(raw):
        try:
            prompt, answers, meta = _extract_prompt_answer(obj)
        except Exception as e:
            raise ValueError(f"Bad example at idx={i}: {e}") from e
        exs.append(Example(idx=i, prompt=prompt, answers=answers, meta=meta))

    if limit is not None:
        exs = exs[: int(limit)]

    return exs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_type", choices=["standard", "sparse", "adaptive"], default="standard")
    ap.add_argument("--device", default="cuda")

    # Generation params (keep aligned with eval_llada defaults)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--gen_length", type=int, default=64)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--remasking", default="low_confidence")
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--skip", type=float, default=0.2)
    ap.add_argument("--select", type=float, default=0.3)
    ap.add_argument("--block_size", type=int, default=32)

    # Adaptive params (optional)
    ap.add_argument("--adaptive_config_path", default=None)
    ap.add_argument("--importance_source", default="precomputed")
    ap.add_argument("--precomputed_importance_path", default=None)
    ap.add_argument("--min_sparsity", type=float, default=0.15)
    ap.add_argument("--max_sparsity", type=float, default=0.85)

    # Data
    ap.add_argument("--len_k", type=int, default=4, help="Max prompt length in K tokens (approx K*1024).")
    ap.add_argument("--limit", type=int, default=None, help="Optional number of examples to evaluate.")
    ap.add_argument("--data_path", default=None, help="JSONL file or directory containing RULER-style eval data.")
    ap.add_argument("--hf_dataset", default=None, help="HuggingFace dataset name (optional).")
    ap.add_argument("--hf_config", default=None, help="HuggingFace dataset config name (optional).")
    ap.add_argument("--split", default="test")

    ap.add_argument("--output_path", required=True)
    ap.add_argument("--samples_path", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    # Load tokenizer separately for prompt-length bookkeeping
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    max_prompt_tokens = int(args.len_k) * 1024

    examples = _load_examples(
        data_path=args.data_path,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        split=args.split,
        len_k=int(args.len_k),
        limit=args.limit,
    )

    # Instantiate the same harness model used by lm-eval
    lm = LLaDAEvalHarness(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        remasking=args.remasking,
        cfg=args.cfg,
        skip=args.skip,
        select=args.select,
        block_size=args.block_size,
        adaptive_config_path=args.adaptive_config_path,
        importance_source=args.importance_source,
        precomputed_importance_path=args.precomputed_importance_path,
        min_sparsity=args.min_sparsity,
        max_sparsity=args.max_sparsity,
        # For RULER we only need generation; keep likelihood MC small
        mc_num=1,
        batch_size=1,
        is_check_greedy=False,
    )

    correct = 0
    correct_norm = 0
    correct_contains = 0
    correct_first_line_contains = 0
    correct_yn_first_token = 0
    n = 0

    samples_f = None
    if args.samples_path:
        os.makedirs(os.path.dirname(args.samples_path), exist_ok=True)
        samples_f = open(args.samples_path, "w", encoding="utf-8")

    try:
        for ex in examples:
            # Truncate prompt to len_k * 1024 tokens (approx). Keep right-truncation (tail) by default to preserve query.
            ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > max_prompt_tokens:
                ids = ids[-max_prompt_tokens:]
            prompt_ids = torch.tensor([ids], dtype=torch.long, device=lm.device)

            # Reuse the harness's generate_until path by calling its model_type-specific generation indirectly:
            # We call the same generation functions as in eval_llada via lm.generate_until-compatible logic:
            # - build a dummy request-like input and call lm.generate_until (expects Instance args),
            # but it's simpler to call the internal generation functions via lm.model_type.
            if lm.model_type == "standard":
                from models.LLaDA.generation.generate import generate as gen_fn
                out_ids = gen_fn(
                    lm.model,
                    prompt_ids,
                    steps=lm.steps,
                    gen_length=lm.gen_length,
                    block_length=lm.block_length,
                    temperature=0,
                    cfg_scale=lm.cfg,
                    remasking=lm.remasking,
                    mask_id=lm.mask_id,
                )
            elif lm.model_type == "sparse":
                from models.LLaDA.generation.sparsed_generate import generate as gen_fn
                out_ids = gen_fn(
                    lm.model,
                    prompt_ids,
                    steps=lm.steps,
                    gen_length=lm.gen_length,
                    block_length=lm.block_length,
                    temperature=0,
                    cfg_scale=lm.cfg,
                    remasking=lm.remasking,
                    mask_id=lm.mask_id,
                    SparseD_param=lm.sparse_param,
                )
            else:
                from models.LLaDA.generation.adaptive_sparsed_generate import generate as gen_fn
                out_ids = gen_fn(
                    lm.model,
                    prompt_ids,
                    steps=lm.steps,
                    gen_length=lm.gen_length,
                    block_length=lm.block_length,
                    temperature=0,
                    cfg_scale=lm.cfg,
                    remasking=lm.remasking,
                    mask_id=lm.mask_id,
                    SparseD_param=lm.sparse_param,
                )

            gen_text = tokenizer.decode(out_ids[0][prompt_ids.shape[1] :], skip_special_tokens=True).strip()
            golds = [str(a).strip() for a in ex.answers if str(a).strip()]
            if not golds:
                golds = [""]

            em = 1 if any(gen_text == g for g in golds) else 0
            pred_norm = _normalize(gen_text)
            gold_norms = [_normalize(g) for g in golds]
            nem = 1 if any(pred_norm == g for g in gold_norms) else 0

            # More forgiving signals for RULER-style short answers:
            # - contains_match: gold substring appears in prediction (normalized)
            # - first_line_contains_match: same, but only within first line (often closer to direct answer)
            # - yn_first_token_acc: for yes/no questions, compare first token of first line
            pred_first_line = _first_line(gen_text)
            pred_first_norm = _normalize(pred_first_line)
            contains = 1 if any(g and (g in pred_norm) for g in gold_norms) else 0
            first_line_contains = 1 if any(g and (g in pred_first_norm) for g in gold_norms) else 0

            yn_first_tok = 0
            if any(g in ("yes", "no") for g in gold_norms):
                pred_tok = _first_token(pred_first_norm)
                yn_first_tok = 1 if any(g == pred_tok for g in gold_norms if g in ("yes", "no")) else 0

            correct += em
            correct_norm += nem
            correct_contains += contains
            correct_first_line_contains += first_line_contains
            correct_yn_first_token += yn_first_tok
            n += 1

            if samples_f:
                samples_f.write(
                    json.dumps(
                        {
                            "idx": ex.idx,
                            "prompt_trunc_len": len(ids),
                            "len_k": args.len_k,
                            "prediction": gen_text,
                            "prediction_first_line": pred_first_line,
                            "golds": golds,
                            "exact_match": em,
                            "normalized_exact_match": nem,
                            "contains_match": contains,
                            "first_line_contains_match": first_line_contains,
                            "yn_first_token_acc": yn_first_tok,
                            "meta": ex.meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    finally:
        if samples_f:
            samples_f.close()

    metrics = {
        "task": "ruler",
        "len_k": int(args.len_k),
        "n": n,
        "exact_match": (correct / n) if n else 0.0,
        "normalized_exact_match": (correct_norm / n) if n else 0.0,
        "contains_match": (correct_contains / n) if n else 0.0,
        "first_line_contains_match": (correct_first_line_contains / n) if n else 0.0,
        "yn_first_token_acc": (correct_yn_first_token / n) if n else 0.0,
        "model_path": args.model_path,
        "model_type": args.model_type,
        "gen_length": args.gen_length,
        "steps": args.steps,
        "block_length": args.block_length,
        "select": args.select,
        "skip": args.skip,
        "block_size": args.block_size,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


