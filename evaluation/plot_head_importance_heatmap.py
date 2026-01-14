#!/usr/bin/env python
"""
Plot side-by-side heatmaps for head importance scores saved by this repo.

Expected file format (torch.load):
{
  "importance_scores": {layer_idx(int): tensor[num_heads], ...},
  "metadata": {...}
}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import numpy as np


def _load_matrix(pt_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    obj = torch.load(pt_path, map_location="cpu")
    if not isinstance(obj, dict) or "importance_scores" not in obj:
        raise ValueError(f"Unexpected format in {pt_path}: expected dict with 'importance_scores'")

    scores = obj["importance_scores"]
    meta = obj.get("metadata", {})
    if not isinstance(scores, dict) or len(scores) == 0:
        raise ValueError(f"Unexpected 'importance_scores' in {pt_path}: expected non-empty dict")

    # Keys are expected to be layer indices (int). Sort deterministically.
    layer_keys: List[int] = sorted(int(k) for k in scores.keys())
    rows = []
    for lk in layer_keys:
        v = scores[lk]
        if not torch.is_tensor(v):
            raise ValueError(f"Layer {lk} in {pt_path} is not a tensor: {type(v)}")
        rows.append(v.detach().cpu().float().numpy())

    mat = np.stack(rows, axis=0)  # [num_layers, num_heads]
    return mat, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt1", required=True, help="First head_importance.pt path")
    parser.add_argument("--pt2", required=True, help="Second head_importance.pt path")
    parser.add_argument("--out", required=True, help="Output png path")
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap name (default: coolwarm)",
    )
    parser.add_argument(
        "--center0",
        action="store_true",
        help="Use a diverging colormap centered at 0 with symmetric range based on max abs across both matrices",
    )
    parser.add_argument(
        "--abs",
        dest="use_abs",
        action="store_true",
        help="Plot absolute values of scores",
    )
    parser.add_argument("--title1", default=None, help="Optional title for left heatmap")
    parser.add_argument("--title2", default=None, help="Optional title for right heatmap")
    args = parser.parse_args()

    m1, meta1 = _load_matrix(args.pt1)
    m2, meta2 = _load_matrix(args.pt2)

    if args.use_abs:
        m1 = np.abs(m1)
        m2 = np.abs(m2)

    if m1.shape != m2.shape:
        raise ValueError(f"Shape mismatch: {m1.shape} vs {m2.shape}")

    # Lazy import matplotlib so the script can still be imported without it.
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    num_layers, num_heads = m1.shape

    title1 = args.title1 or Path(args.pt1).resolve().parent.name
    title2 = args.title2 or Path(args.pt2).resolve().parent.name

    vmin = min(float(m1.min()), float(m2.min()))
    vmax = max(float(m1.max()), float(m2.max()))

    norm = None
    if args.center0 and not args.use_abs:
        max_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -max_abs, max_abs
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, sharey=True)

    imshow_kwargs = dict(aspect="auto", cmap=args.cmap, origin="lower")
    # Newer matplotlib disallows passing (norm) together with (vmin/vmax).
    if norm is not None:
        imshow_kwargs["norm"] = norm
    else:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax

    axes[0].imshow(m1, **imshow_kwargs)
    axes[0].set_title(title1)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")

    im1 = axes[1].imshow(m2, **imshow_kwargs)
    axes[1].set_title(title2)
    axes[1].set_xlabel("Head")

    # Ticks: keep readable
    xticks = list(range(0, num_heads, 4))
    yticks = list(range(0, num_layers, 4))
    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xlim(-0.5, num_heads - 0.5)
        ax.set_ylim(-0.5, num_layers - 0.5)

    cbar = fig.colorbar(im1, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Head importance score" + (" (abs)" if args.use_abs else ""))

    # Footer with method (if present)
    method1 = meta1.get("method", None)
    method2 = meta2.get("method", None)
    if method1 or method2:
        fig.suptitle(f"method1={method1} | method2={method2}", fontsize=10)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved heatmap to: {out_path}")


if __name__ == "__main__":
    main()


