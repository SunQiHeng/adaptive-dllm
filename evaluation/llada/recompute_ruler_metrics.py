#!/usr/bin/env python3
"""
Recompute RULER metrics from an existing samples.jsonl (no model inference).

Use this when you already have:
  - samples.jsonl produced by eval_ruler_llada.py (or compatible schema)
and want less-strict metrics than exact string match.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List


_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _first_line(s: str) -> str:
    return (s or "").splitlines()[0].strip() if s else ""


def _first_token(norm_s: str) -> str:
    norm_s = (norm_s or "").strip()
    if not norm_s:
        return ""
    return norm_s.split(" ", 1)[0]


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_path", required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    n = 0
    em_sum = 0
    nem_sum = 0
    contains_sum = 0
    first_line_contains_sum = 0
    yn_first_token_sum = 0

    with open(args.samples_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = str(obj.get("prediction", "")).strip()
            golds = _to_str_list(obj.get("golds"))
            golds = [str(g).strip() for g in golds if str(g).strip()]
            if not golds:
                golds = [""]

            pred_norm = _normalize(pred)
            gold_norms = [_normalize(g) for g in golds]

            em = 1 if any(pred == g for g in golds) else 0
            nem = 1 if any(pred_norm == g for g in gold_norms) else 0

            pred_first = _first_line(pred)
            pred_first_norm = _normalize(pred_first)
            contains = 1 if any(g and (g in pred_norm) for g in gold_norms) else 0
            first_line_contains = 1 if any(g and (g in pred_first_norm) for g in gold_norms) else 0

            yn_first_tok = 0
            if any(g in ("yes", "no") for g in gold_norms):
                pred_tok = _first_token(pred_first_norm)
                yn_first_tok = 1 if any(g == pred_tok for g in gold_norms if g in ("yes", "no")) else 0

            em_sum += em
            nem_sum += nem
            contains_sum += contains
            first_line_contains_sum += first_line_contains
            yn_first_token_sum += yn_first_tok
            n += 1

    metrics: Dict[str, Any] = {
        "task": "ruler",
        "n": n,
        "exact_match": (em_sum / n) if n else 0.0,
        "normalized_exact_match": (nem_sum / n) if n else 0.0,
        "contains_match": (contains_sum / n) if n else 0.0,
        "first_line_contains_match": (first_line_contains_sum / n) if n else 0.0,
        "yn_first_token_acc": (yn_first_token_sum / n) if n else 0.0,
        "samples_path": args.samples_path,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



