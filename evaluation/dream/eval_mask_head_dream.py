#!/usr/bin/env python3
"""
Dream head pruning / masking evaluation with lm-eval-harness.

注册新 model：
  --model dream_mask_head_eval

--model_args 支持：
- model_path=...
- prune_which=most|least|random
- prune_k=... 或 prune_k_frac=... （二选一）
- layer_start=0, layer_end=...
- random_prune_seed=...
- importance_path=/path/to/head_importance.pt  （random 模式可不传）

其余 generation / MC 参数沿用 `evaluation/dream/eval_dream.py:DreamEvalHarness`。
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from evaluation.dream.eval_dream import DreamEvalHarness, set_seed
from models.Dream.core.mask_head_modeling_dream import (
    apply_head_keep_masks_,
    build_head_keep_masks_global,
    build_random_head_keep_masks_global,
    load_importance_scores_pt,
    patch_dream_attention_for_head_masking,
)


def _str_to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).lower().strip()
    return s in ("1", "true", "t", "yes", "y")


def _strip_quotes(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()
    return t


@register_model("dream_mask_head_eval")
class DreamMaskHeadEvalHarness(DreamEvalHarness):
    """
    在 Dream standard attention 上做 head 剪枝验证。
    """

    def __init__(
        self,
        model_path="/data/qh_models/Dream-v0-Instruct-7B",
        importance_path=None,
        prune_which: str = "most",  # most|least|random
        prune_k=None,
        prune_k_frac=None,
        layer_start: int = 0,
        layer_end: int = -1,
        random_prune_seed: int = 1234,
        keep_at_least_one_head=True,
        **kwargs,
    ):
        # 强制 standard（head pruning 的对照最清晰）
        super().__init__(model_path=model_path, model_type="standard", **kwargs)

        prune_which = str(prune_which).strip()

        meta = {}
        if prune_which == "random":
            n_layers = int(getattr(self.model.config, "num_hidden_layers"))
            n_heads = int(getattr(self.model.config, "num_attention_heads"))
            keep_masks = build_random_head_keep_masks_global(
                n_layers=n_layers,
                n_heads=n_heads,
                k=int(prune_k) if prune_k is not None else None,
                k_frac=float(prune_k_frac) if prune_k_frac is not None else None,
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                seed=int(random_prune_seed),
                keep_at_least_one_head=_str_to_bool(keep_at_least_one_head),
            )
        else:
            importance_path = _strip_quotes(importance_path)
            if not importance_path:
                raise ValueError("importance_path is required unless prune_which='random'")
            scores, meta = load_importance_scores_pt(importance_path)
            keep_masks = build_head_keep_masks_global(
                scores,
                prune_which=prune_which,
                k=int(prune_k) if prune_k is not None else None,
                k_frac=float(prune_k_frac) if prune_k_frac is not None else None,
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                keep_at_least_one_head=_str_to_bool(keep_at_least_one_head),
            )

        # Patch + apply masks
        patch_dream_attention_for_head_masking(self.model)
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = None
        apply_head_keep_masks_(self.model, keep_masks, device=dev)

        # 打印剪枝概况（全局）
        total_pruned = 0
        total_heads = 0
        for li, m in sorted(keep_masks.items()):
            n = int(m.numel())
            pruned = int((~m).sum().item())
            total_pruned += pruned
            total_heads += n
        print(f"[head_prune] mode={prune_which} total_pruned={total_pruned}/{total_heads} ({(total_pruned/max(1,total_heads))*100:.1f}%)")
        if meta:
            print(f"[head_prune] importance metadata keys: {sorted(list(meta.keys()))[:20]}")


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()

