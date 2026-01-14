#!/usr/bin/env python3
"""
Download RULER from HuggingFace and export to JSONL files suitable for eval_ruler_llada.py.

Default dataset: self-long/RULER-llama3-1M

It will export one JSONL per dataset config (e.g., niah_single_1_8k.jsonl) into:
  <out_dir>/jsonl/<split>/<config>.jsonl

Each line schema:
  {
    "input": "...",
    "answers": ["...","..."],
    "meta": {...}
  }
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


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
    ap.add_argument("--dataset", default="self-long/RULER-llama3-1M")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out_dir", default="/data/qh_models/ruler")
    ap.add_argument("--lengths", default="8k", help="Comma-separated length suffixes to export, e.g. 4k,8k,16k")
    ap.add_argument("--include_configs", default="", help="Optional comma-separated explicit config names to export.")
    ap.add_argument("--max_examples_per_config", type=int, default=0, help="0 = all; otherwise cap for quick testing.")
    args = ap.parse_args()

    from datasets import get_dataset_config_names, load_dataset  # local import

    os.makedirs(args.out_dir, exist_ok=True)
    out_jsonl_root = os.path.join(args.out_dir, "jsonl", args.split)
    os.makedirs(out_jsonl_root, exist_ok=True)

    cfgs = get_dataset_config_names(args.dataset)
    lengths = [x.strip() for x in args.lengths.split(",") if x.strip()]

    include = [x.strip() for x in args.include_configs.split(",") if x.strip()]
    if include:
        target_cfgs = [c for c in cfgs if c in include]
    else:
        target_cfgs = [c for c in cfgs if any(c.endswith("_" + suf) for suf in lengths)]

    if not target_cfgs:
        raise SystemExit(f"No configs matched. lengths={lengths}, include={include}")

    print(f"[ruler] dataset={args.dataset} split={args.split}")
    print(f"[ruler] exporting {len(target_cfgs)} configs -> {out_jsonl_root}")

    for cfg in target_cfgs:
        out_path = os.path.join(out_jsonl_root, f"{cfg}.jsonl")
        if os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            continue

        print(f"[dl] {cfg} ...")
        ds = load_dataset(args.dataset, cfg, split=args.split)
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in ds:
                inp = row.get("input") or row.get("prompt") or row.get("question")
                if not isinstance(inp, str):
                    continue
                answers = _to_str_list(row.get("answers") or row.get("answer") or row.get("target"))
                if not answers:
                    continue
                meta: Dict[str, Any] = {k: v for k, v in row.items() if k not in {"input", "prompt", "question", "answers", "answer", "target", "predictions"}}
                meta.update({"hf_dataset": args.dataset, "hf_config": cfg, "split": args.split})
                f.write(json.dumps({"input": inp, "answers": answers, "meta": meta}, ensure_ascii=False) + "\n")
                n += 1
                if args.max_examples_per_config and n >= args.max_examples_per_config:
                    break
        print(f"[ok] wrote {n} -> {out_path}")

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


