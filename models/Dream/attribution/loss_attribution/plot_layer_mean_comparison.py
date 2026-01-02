#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List

import torch


def _load_layer_means(pt_path: Path, *, mode: str) -> Tuple[List[int], torch.Tensor]:
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    imp = obj.get("importance_scores", None)
    if not isinstance(imp, dict) or len(imp) == 0:
        raise ValueError(f"Invalid or empty importance_scores in: {pt_path}")

    layers = sorted(int(k) for k in imp.keys())
    vals = []
    for l in layers:
        t = imp[int(l)]
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        t = t.detach().to(torch.float32).flatten()
        if mode == "abs_mean":
            vals.append(t.abs().mean())
        elif mode == "mean":
            vals.append(t.mean())
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return layers, torch.stack(vals, dim=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare per-layer mean head importance (Dream vs LLaDA).")
    p.add_argument("--dream_pt", type=str, required=True, help="Path to Dream head_importance.pt")
    p.add_argument("--llada_pt", type=str, required=True, help="Path to LLaDA head_importance.pt")
    p.add_argument("--mode", type=str, default="abs_mean", choices=["abs_mean", "mean"])
    p.add_argument("--out", type=str, default="", help="Output image path (.png/.pdf). Default: alongside dream_pt.")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dream_pt = Path(args.dream_pt)
    llada_pt = Path(args.llada_pt)
    if not dream_pt.exists():
        raise FileNotFoundError(f"Missing: {dream_pt}")
    if not llada_pt.exists():
        raise FileNotFoundError(f"Missing: {llada_pt}")

    d_layers, d_vals = _load_layer_means(dream_pt, mode=str(args.mode))
    l_layers, l_vals = _load_layer_means(llada_pt, mode=str(args.mode))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(d_layers, d_vals.numpy(), marker="o", markersize=3, linewidth=1.5, label=f"Dream ({len(d_layers)} layers)")
    ax.plot(l_layers, l_vals.numpy(), marker="o", markersize=3, linewidth=1.5, label=f"LLaDA ({len(l_layers)} layers)")

    ax.set_xlabel("Layer")
    ylab = "Per-layer mean |head_importance|" if args.mode == "abs_mean" else "Per-layer mean head_importance (signed)"
    ax.set_ylabel(ylab)
    ax.set_title("Per-layer mean head importance comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = dream_pt.parent / f"layer_mean_comparison_{args.mode}_{ts}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()


