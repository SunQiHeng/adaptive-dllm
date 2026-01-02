#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch


def _load_matrix(pt_path: Path) -> Tuple[torch.Tensor, Dict]:
    data = torch.load(pt_path, map_location="cpu")
    scores = data.get("importance_scores", None)
    if not isinstance(scores, dict) or len(scores) == 0:
        raise ValueError(f"Invalid or empty importance_scores in: {pt_path}")
    layers = sorted(int(k) for k in scores.keys())
    rows = []
    for l in layers:
        v = scores[int(l)]
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        rows.append(v.detach().to(torch.float32).flatten())
    mat = torch.stack(rows, dim=0)  # [n_layers, n_heads]
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    return mat, meta


def _robust_minmax(x: torch.Tensor, lo: float, hi: float) -> Tuple[float, float]:
    v = x.flatten()
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return 0.0, 1.0
    lo_v = float(torch.quantile(v, torch.tensor(lo)))
    hi_v = float(torch.quantile(v, torch.tensor(hi)))
    if lo_v == hi_v:
        mn = float(v.min())
        mx = float(v.max())
        if mn == mx:
            return mn - 1.0, mx + 1.0
        return mn, mx
    return lo_v, hi_v


def main() -> None:
    p = argparse.ArgumentParser(description="Plot a head-importance heatmap (layer x head).")
    p.add_argument("--in_dir", type=str, default="", help="Directory containing head_importance.pt")
    p.add_argument("--in_pt", type=str, default="", help="Path to head_importance.pt (overrides --in_dir)")
    p.add_argument("--out", type=str, default="", help="Output image path (.png/.pdf). Default: alongside input.")
    p.add_argument("--title", type=str, default="", help="Plot title (optional)")
    p.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap")
    p.add_argument("--robust_lo", type=float, default=0.01, help="Lower quantile for color scaling")
    p.add_argument("--robust_hi", type=float, default=0.99, help="Upper quantile for color scaling")
    p.add_argument(
        "--center_zero",
        action="store_true",
        default=False,
        help="If set, use a symmetric color range around 0 (useful for signed IG).",
    )
    args = p.parse_args()

    # Headless-safe backend
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_pt: Path
    if args.in_pt:
        in_pt = Path(args.in_pt)
    else:
        if not args.in_dir:
            raise ValueError("Please pass --in_pt or --in_dir.")
        in_pt = Path(args.in_dir) / "head_importance.pt"

    if not in_pt.exists():
        raise FileNotFoundError(f"Missing: {in_pt}")

    mat, meta = _load_matrix(in_pt)
    vmin, vmax = _robust_minmax(mat, float(args.robust_lo), float(args.robust_hi))
    if bool(args.center_zero):
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    title = args.title.strip()
    if not title:
        # Best-effort auto title from metadata
        dataset = meta.get("dataset", None)
        seed = meta.get("seed", None)
        data_seed = meta.get("data_seed", None)
        mask_seed = meta.get("mask_seed", None)
        title = f"Dream head importance (dataset={dataset}, seed={seed}, dseed={data_seed}, mseed={mask_seed})"

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    im = ax.imshow(mat.numpy(), aspect="auto", interpolation="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    cbar = fig.colorbar(im, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_label("Head importance")
    fig.tight_layout()

    out_path: Path
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = in_pt.parent / f"head_importance_heatmap_{ts}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


