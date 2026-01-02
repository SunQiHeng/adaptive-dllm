import argparse
import os
from datetime import datetime

import torch


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a negated head-importance file for LLaDA adaptive sparse tests.\n"
            "Multiplies each per-layer head score vector by -1."
        )
    )
    parser.add_argument(
        "--in_pt",
        required=True,
        help="Path to an existing head_importance.pt (expects key 'importance_scores').",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory to write head_importance.pt (and metadata).",
    )
    args = parser.parse_args()

    src = args.in_pt
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    obj = torch.load(src, weights_only=False, map_location="cpu")
    if "importance_scores" not in obj:
        raise KeyError(f"Invalid file: {src}. Expected key 'importance_scores'. Got keys: {list(obj.keys())}")

    imp = obj["importance_scores"]
    if not isinstance(imp, dict) or len(imp) == 0:
        raise ValueError(f"Invalid 'importance_scores' in {src}: expected non-empty dict[layer_idx -> tensor].")

    layer_keys = sorted(imp.keys())
    first_k = layer_keys[0]
    dtype = imp[first_k].dtype

    neg = {}
    for layer_idx in layer_keys:
        t = imp[layer_idx]
        if not torch.is_tensor(t):
            raise ValueError(f"Layer {layer_idx} importance is not a tensor: {type(t)}")
        t = t.detach().to(device="cpu", dtype=dtype)
        if t.dim() != 1:
            raise ValueError(f"Expected 1D tensor per-layer importance, got shape={tuple(t.shape)}")
        neg[layer_idx] = -t

    out = {
        "importance_scores": neg,
        "metadata": {
            "source_path": os.path.abspath(src),
            "generated_at": datetime.now().isoformat(),
            "note": "Per-layer head scores are multiplied by -1.",
        },
    }

    out_path = os.path.join(out_dir, "head_importance.pt")
    torch.save(out, out_path)
    print(f"✅ Wrote negated importance to: {out_path}")
    print(f"   layers: {len(neg)}, heads_per_layer: {int(neg[first_k].numel())}")


if __name__ == "__main__":
    main()


