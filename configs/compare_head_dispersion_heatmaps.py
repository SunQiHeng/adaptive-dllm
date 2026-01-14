#!/usr/bin/env python3
from __future__ import annotations

"""
Compare two `head_dispersion.pt` outputs side-by-side as heatmaps (per metric).

Writes one figure per metric with 3 panels:
  A (seed/run1), B (seed/run2), and A-B (difference).

Usage:
  python adaptive-dllm/configs/compare_head_dispersion_heatmaps.py \
    --dir_a .../head_dispersion_dream_base_disp1 \
    --dir_b .../head_dispersion_dream_base_disp2 \
    --out_dir .../configs/_tmp_disp_compare
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch


def _load_metric_matrix(pt_path: Path, metric: str) -> Tuple[torch.Tensor, Dict]:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    disp = data.get("dispersion", None)
    if not isinstance(disp, dict):
        raise ValueError(f"Invalid dispersion in: {pt_path}")
    metric_dict = disp.get(metric, None)
    if not isinstance(metric_dict, dict) or len(metric_dict) == 0:
        raise ValueError(f"Missing metric={metric!r} in: {pt_path}")

    layers = sorted(int(k) for k in metric_dict.keys())
    rows = []
    for l in layers:
        v = metric_dict[int(l)]
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        rows.append(v.detach().to(torch.float32).flatten())
    mat = torch.stack(rows, dim=0)  # [n_layers, n_heads]
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    return mat, meta


def _robust_minmax(x: torch.Tensor, lo: float, hi: float) -> Tuple[float, float]:
    xf = x.flatten()
    xf = xf[torch.isfinite(xf)]
    if xf.numel() == 0:
        return 0.0, 1.0
    lo_v = float(torch.quantile(xf, torch.tensor(float(lo))))
    hi_v = float(torch.quantile(xf, torch.tensor(float(hi))))
    if lo_v == hi_v:
        mn = float(xf.min())
        mx = float(xf.max())
        if mn == mx:
            return mn - 1.0, mx + 1.0
        return mn, mx
    return lo_v, hi_v


def _fmt_title(prefix: str, d: Path, meta: Dict) -> str:
    seed = meta.get("seed", None)
    ds = meta.get("dataset", None)
    bs = meta.get("block_size", None)
    qs = meta.get("query_span", None)
    agg = meta.get("aggregation", None)
    return f"{prefix}: {d.name}\\n(seed={seed}, dataset={ds}, bs={bs}, qs={qs}, agg={agg})"


def main() -> None:
    p = argparse.ArgumentParser(description="Compare head dispersion heatmaps for two runs (all metrics).")
    p.add_argument("--dir_a", type=str, required=True)
    p.add_argument("--dir_b", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--cmap", type=str, default="viridis")
    p.add_argument("--robust_lo", type=float, default=0.01)
    p.add_argument("--robust_hi", type=float, default=0.99)
    p.add_argument(
        "--diff_center_zero",
        action="store_true",
        default=True,
        help="Center diff heatmap around 0 (recommended).",
    )
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_dir = Path(args.dir_a)
    b_dir = Path(args.dir_b)
    a_pt = a_dir / "head_dispersion.pt"
    b_pt = b_dir / "head_dispersion.pt"
    if not a_pt.exists():
        raise FileNotFoundError(f"Missing: {a_pt}")
    if not b_pt.exists():
        raise FileNotFoundError(f"Missing: {b_pt}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine metrics from file A (they should match B)
    data_a = torch.load(a_pt, map_location="cpu", weights_only=False)
    disp = data_a.get("dispersion", {})
    if not isinstance(disp, dict) or not disp:
        raise ValueError(f"Invalid dispersion in: {a_pt}")
    metrics = sorted(disp.keys())

    print("metrics:", metrics)

    for metric in metrics:
        A, meta_a = _load_metric_matrix(a_pt, metric)
        B, meta_b = _load_metric_matrix(b_pt, metric)
        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch for {metric}: A={tuple(A.shape)} vs B={tuple(B.shape)}")

        D = A - B
        # Shared scaling for A/B
        vmin, vmax = _robust_minmax(torch.cat([A, B], dim=0), float(args.robust_lo), float(args.robust_hi))
        # Separate scaling for diff
        dvmin, dvmax = _robust_minmax(D, float(args.robust_lo), float(args.robust_hi))
        if bool(args.diff_center_zero):
            m = max(abs(dvmin), abs(dvmax))
            dvmin, dvmax = -m, m

        fig, axes = plt.subplots(1, 3, figsize=(21, 6), sharey=True)
        axes[0].imshow(A.numpy(), aspect="auto", interpolation="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(_fmt_title("A", a_dir, meta_a))
        axes[0].set_xlabel("Head")
        axes[0].set_ylabel("Layer")

        im1 = axes[1].imshow(B.numpy(), aspect="auto", interpolation="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(_fmt_title("B", b_dir, meta_b))
        axes[1].set_xlabel("Head")

        im2 = axes[2].imshow(D.numpy(), aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=dvmin, vmax=dvmax)
        axes[2].set_title("A - B")
        axes[2].set_xlabel("Head")

        fig.suptitle(f"Head dispersion heatmaps: {metric} (layer x head)", y=0.98)
        fig.colorbar(im1, ax=axes[:2].ravel().tolist(), shrink=0.9, pad=0.02, label=metric)
        fig.colorbar(im2, ax=[axes[2]], shrink=0.9, pad=0.02, label="diff")
        fig.tight_layout()

        out_path = out_dir / f"head_dispersion_compare_{metric}.png"
        fig.savefig(str(out_path), dpi=220)
        plt.close(fig)
        print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


