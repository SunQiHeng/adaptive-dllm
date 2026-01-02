#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _load_scores(pt_path: Path) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    data = torch.load(pt_path, map_location="cpu")
    scores = data.get("importance_scores", None)
    if not isinstance(scores, dict) or len(scores) == 0:
        raise ValueError(f"Invalid or empty importance_scores in: {pt_path}")
    out: Dict[int, torch.Tensor] = {}
    for k, v in scores.items():
        lk = int(k)
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        out[lk] = v.detach().to(torch.float32).clone()
    meta = data.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    return out, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Average two head_importance.pt files (per-layer, per-head).")
    p.add_argument("--dir_a", type=str, required=True, help="Directory containing head_importance.pt")
    p.add_argument("--dir_b", type=str, required=True, help="Directory containing head_importance.pt")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory (will be created).")
    p.add_argument("--weight_a", type=float, default=1.0)
    p.add_argument("--weight_b", type=float, default=1.0)
    p.add_argument(
        "--require_same_metadata",
        action="store_true",
        default=False,
        help="If set, error when key metadata fields mismatch (dataset/mask_probs/etc).",
    )
    args = p.parse_args()

    a_dir = Path(args.dir_a)
    b_dir = Path(args.dir_b)
    a_pt = a_dir / "head_importance.pt"
    b_pt = b_dir / "head_importance.pt"
    if not a_pt.exists():
        raise FileNotFoundError(f"Missing: {a_pt}")
    if not b_pt.exists():
        raise FileNotFoundError(f"Missing: {b_pt}")

    sA, mA = _load_scores(a_pt)
    sB, mB = _load_scores(b_pt)

    keysA = set(sA.keys())
    keysB = set(sB.keys())
    if keysA != keysB:
        raise ValueError(f"Layer keys mismatch: onlyA={sorted(keysA-keysB)} onlyB={sorted(keysB-keysA)}")

    wA = float(args.weight_a)
    wB = float(args.weight_b)
    if wA < 0 or wB < 0 or (wA + wB) == 0:
        raise ValueError("Weights must be non-negative and not both zero.")

    avg: Dict[int, torch.Tensor] = {}
    for l in sorted(keysA):
        a = sA[l]
        b = sB[l]
        if a.shape != b.shape:
            raise ValueError(f"Layer {l} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        avg[l] = (wA * a + wB * b) / (wA + wB)

    if bool(args.require_same_metadata):
        check_keys = [
            "method",
            "model_path",
            "dataset",
            "split",
            "max_samples",
            "ig_steps",
            "mask_probs",
            "mask_samples_per_prob",
            "loss_normalize",
            "ig_postprocess",
            "mask_batch_size",
            "baseline",
            "baseline_value",
            "layer_range",
        ]
        for k in check_keys:
            if mA.get(k, None) != mB.get(k, None):
                raise ValueError(f"metadata mismatch on key '{k}': A={mA.get(k)} vs B={mB.get(k)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "importance_scores": {int(k): v.detach().cpu() for k, v in avg.items()},
        "metadata": {
            "method": "average_of_two_runs",
            "generated_at": datetime.now().isoformat(),
            "sources": [
                {"dir": str(a_dir), "weight": wA, "metadata": mA},
                {"dir": str(b_dir), "weight": wB, "metadata": mB},
            ],
            "note": "importance_scores are averaged per-layer, per-head on float32 CPU.",
        },
    }

    out_path = out_dir / "head_importance.pt"
    torch.save(out, out_path)
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


