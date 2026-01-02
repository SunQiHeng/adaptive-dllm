#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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


def _robust_minmax(a: torch.Tensor, b: torch.Tensor, lo: float, hi: float) -> Tuple[float, float]:
    x = torch.cat([a.flatten(), b.flatten()], dim=0)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return 0.0, 1.0
    lo_v = float(torch.quantile(x, torch.tensor(lo)))
    hi_v = float(torch.quantile(x, torch.tensor(hi)))
    if lo_v == hi_v:
        # fallback
        mn = float(x.min())
        mx = float(x.max())
        if mn == mx:
            return mn - 1.0, mx + 1.0
        return mn, mx
    return lo_v, hi_v


def main() -> None:
    p = argparse.ArgumentParser(description="Plot two head-importance heatmaps side-by-side.")
    p.add_argument("--dir_a", type=str, required=True, help="Directory containing head_importance.pt")
    p.add_argument("--dir_b", type=str, required=True, help="Directory containing head_importance.pt")
    p.add_argument("--out", type=str, required=True, help="Output image path (.png/.pdf)")
    p.add_argument("--title_a", type=str, default="", help="Title for left subplot (optional)")
    p.add_argument("--title_b", type=str, default="", help="Title for right subplot (optional)")
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

    a_dir = Path(args.dir_a)
    b_dir = Path(args.dir_b)
    a_pt = a_dir / "head_importance.pt"
    b_pt = b_dir / "head_importance.pt"
    if not a_pt.exists():
        raise FileNotFoundError(f"Missing: {a_pt}")
    if not b_pt.exists():
        raise FileNotFoundError(f"Missing: {b_pt}")

    A, meta_a = _load_matrix(a_pt)
    B, meta_b = _load_matrix(b_pt)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A={tuple(A.shape)} vs B={tuple(B.shape)}")

    vmin, vmax = _robust_minmax(A, B, float(args.robust_lo), float(args.robust_hi))
    if bool(args.center_zero):
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    title_a = args.title_a.strip()
    title_b = args.title_b.strip()
    if not title_a:
        seed = meta_a.get("seed", None)
        data_seed = meta_a.get("data_seed", None)
        mask_seed = meta_a.get("mask_seed", None)
        title_a = f"A (seed={seed}, dseed={data_seed}, mseed={mask_seed})"
    if not title_b:
        seed = meta_b.get("seed", None)
        data_seed = meta_b.get("data_seed", None)
        mask_seed = meta_b.get("mask_seed", None)
        title_b = f"B (seed={seed}, dseed={data_seed}, mseed={mask_seed})"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    im0 = axes[0].imshow(A.numpy(), aspect="auto", interpolation="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(title_a)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")

    im1 = axes[1].imshow(B.numpy(), aspect="auto", interpolation="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(title_b)
    axes[1].set_xlabel("Head")

    # One shared colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label("Head importance")

    fig.suptitle("Head importance heatmaps (layer x head)", y=0.98)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


