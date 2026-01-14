#!/usr/bin/env python
"""
Plot a 2x2 grid of heatmaps for head importance scores.

Expected file format (torch.load):
{
  "importance_scores": {layer_idx(int): tensor[num_heads], ...},
  "metadata": {...}
}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _load_matrix(pt_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    obj = torch.load(pt_path, map_location="cpu")
    if not isinstance(obj, dict) or "importance_scores" not in obj:
        raise ValueError(f"Unexpected format in {pt_path}: expected dict with 'importance_scores'")

    scores = obj["importance_scores"]
    meta = obj.get("metadata", {})
    if not isinstance(scores, dict) or len(scores) == 0:
        raise ValueError(f"Unexpected 'importance_scores' in {pt_path}: expected non-empty dict")

    layer_keys: List[int] = sorted(int(k) for k in scores.keys())
    rows = []
    for lk in layer_keys:
        v = scores[lk]
        if not torch.is_tensor(v):
            raise ValueError(f"Layer {lk} in {pt_path} is not a tensor: {type(v)}")
        rows.append(v.detach().cpu().float().numpy())
    mat = np.stack(rows, axis=0)  # [num_layers, num_heads]
    return mat, meta if isinstance(meta, dict) else {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt11", required=True, help="Top-left head_importance.pt")
    p.add_argument("--pt12", required=True, help="Top-right head_importance.pt")
    p.add_argument("--pt21", required=True, help="Bottom-left head_importance.pt")
    p.add_argument("--pt22", required=True, help="Bottom-right head_importance.pt")
    p.add_argument("--out", required=True, help="Output png path")
    p.add_argument("--title11", default=None)
    p.add_argument("--title12", default=None)
    p.add_argument("--title21", default=None)
    p.add_argument("--title22", default=None)
    p.add_argument("--cmap", default="coolwarm")
    p.add_argument("--center0", action="store_true", help="Center colormap at 0 with symmetric range across all 4")
    p.add_argument("--abs", dest="use_abs", action="store_true", help="Plot absolute values")
    args = p.parse_args()

    mats: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    for pt in [args.pt11, args.pt12, args.pt21, args.pt22]:
        m, meta = _load_matrix(pt)
        mats.append(m)
        metas.append(meta)

    if any(m.shape != mats[0].shape for m in mats[1:]):
        raise ValueError(f"Shape mismatch across inputs: {[m.shape for m in mats]}")

    if args.use_abs:
        mats = [np.abs(m) for m in mats]

    vmin = min(float(m.min()) for m in mats)
    vmax = max(float(m.max()) for m in mats)

    norm = None
    if args.center0 and not args.use_abs:
        from matplotlib.colors import TwoSlopeNorm

        max_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -max_abs, max_abs
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    import matplotlib.pyplot as plt

    imshow_kwargs = dict(aspect="auto", cmap=args.cmap, origin="lower")
    if norm is not None:
        imshow_kwargs["norm"] = norm
    else:
        imshow_kwargs["vmin"] = vmin
        imshow_kwargs["vmax"] = vmax

    num_layers, num_heads = mats[0].shape

    def default_title(pt: str) -> str:
        return Path(pt).resolve().parent.name

    titles = [
        args.title11 or default_title(args.pt11),
        args.title12 or default_title(args.pt12),
        args.title21 or default_title(args.pt21),
        args.title22 or default_title(args.pt22),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True, sharex=True, sharey=True)
    ims = []
    ims.append(axes[0, 0].imshow(mats[0], **imshow_kwargs))
    ims.append(axes[0, 1].imshow(mats[1], **imshow_kwargs))
    ims.append(axes[1, 0].imshow(mats[2], **imshow_kwargs))
    ims.append(axes[1, 1].imshow(mats[3], **imshow_kwargs))

    axes[0, 0].set_title(titles[0])
    axes[0, 1].set_title(titles[1])
    axes[1, 0].set_title(titles[2])
    axes[1, 1].set_title(titles[3])

    for ax in axes[:, 0]:
        ax.set_ylabel("Layer")
    for ax in axes[1, :]:
        ax.set_xlabel("Head")

    xticks = list(range(0, num_heads, 4))
    yticks = list(range(0, num_layers, 4))
    for ax in axes.reshape(-1):
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xlim(-0.5, num_heads - 0.5)
        ax.set_ylim(-0.5, num_layers - 0.5)

    cbar = fig.colorbar(ims[-1], ax=axes.reshape(-1).tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("Head importance score" + (" (abs)" if args.use_abs else ""))

    # Put a small line with (method) if present and consistent
    methods = [m.get("method", None) for m in metas]
    if any(methods):
        fig.suptitle(" | ".join(str(x) for x in methods), fontsize=9)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved 2x2 heatmap to: {out_path}")


if __name__ == "__main__":
    main()


