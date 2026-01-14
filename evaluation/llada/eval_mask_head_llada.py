#!/usr/bin/env python3
"""
LLaDA head pruning / masking evaluation with lm-eval-harness.

用法与 `eval_llada.py` 保持一致（同样调用 `cli_evaluate()`），但注册了一个新的 model：
  --model llada_mask_head_eval

你可以通过 --model_args 传入：
- model_path=...
- importance_path=/path/to/head_importance.pt
- prune_which=most|least
- prune_k=64   或 prune_k_frac=0.25（二选一）
- layer_start=0, layer_end=31

其余 generation / MC 参数沿用 `evaluation/llada/eval_llada.py:LLaDAEvalHarness`。
"""

from __future__ import annotations

import os
import sys
from typing import Any

# 让本仓库可被 import（与 eval_llada.py 一致）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from evaluation.llada.eval_llada import LLaDAEvalHarness, set_seed  # 复用全部评测逻辑
from models.LLaDA.core.mask_head_modeling import (
    apply_head_keep_masks_,
    build_head_keep_masks,
    build_random_head_keep_masks,
    load_importance_scores_pt,
    patch_llada_blocks_for_head_masking,
)


def _str_to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    return False


def _strip_quotes(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    # Strip one layer of matching quotes
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()
    return t


@register_model("llada_mask_head_eval")
class LLaDAMaskHeadEvalHarness(LLaDAEvalHarness):
    """
    在标准 attention 的 LLaDA 上做 head 剪枝验证。

    继承 `LLaDAEvalHarness` 的全部 loglikelihood / generate_until 逻辑，
    只在初始化时加载 importance 并对模型做 head mask patch。
    """

    def __init__(
        self,
        model_path="GSAI-ML/LLaDA-8B-Base",
        # head pruning args
        importance_path=None,
        prune_which: str = "most",  # most|least|random
        prune_k=None,
        prune_k_frac=None,
        layer_start: int = 0,
        layer_end: int = -1,
        keep_at_least_one_head=True,
        random_prune_seed: int = 1234,
        # keep super's signature compatible
        **kwargs,
    ):
        # 强制使用 standard（因为我们只验证 head pruning，不引入 sparse/adaptive scheduling）
        super().__init__(model_path=model_path, model_type="standard", **kwargs)

        prune_which = str(prune_which).strip()
        meta = {}
        if prune_which == "random":
            # Random baseline does not need importance file; uses model config for shapes.
            n_layers = int(getattr(self.model.config, "n_layers"))
            n_heads = int(getattr(self.model.config, "n_heads"))
            keep_masks = build_random_head_keep_masks(
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
                raise ValueError(
                    "importance_path is required unless prune_which='random' (path to head_importance.pt)"
                )
            scores, meta = load_importance_scores_pt(importance_path)
            keep_masks = build_head_keep_masks(
                scores,
                prune_which=prune_which,
                k=int(prune_k) if prune_k is not None else None,
                k_frac=float(prune_k_frac) if prune_k_frac is not None else None,
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                keep_at_least_one_head=_str_to_bool(keep_at_least_one_head),
            )

        # Patch attention() once and then write masks
        patch_llada_blocks_for_head_masking(self.model)
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = None
        apply_head_keep_masks_(self.model, keep_masks, device=dev)

        # 简单打印统计信息，方便在日志里确认配置生效
        total_pruned = 0
        total_heads = 0
        for li, m in sorted(keep_masks.items()):
            n = int(m.numel())
            pruned = int((~m).sum().item())
            total_pruned += pruned
            total_heads += n
            if pruned > 0:
                print(f"[head_prune] layer={li:02d} pruned={pruned}/{n} ({pruned/n*100:.1f}%)")
        print(
            f"[head_prune] TOTAL pruned={total_pruned}/{total_heads} "
            f"({(total_pruned/max(1,total_heads))*100:.1f}%)"
        )
        if meta:
            print(f"[head_prune] importance metadata keys: {sorted(list(meta.keys()))[:20]}")


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()

