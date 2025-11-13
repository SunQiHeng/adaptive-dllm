"""
LLaDA Sparse Attention Module

Utilities for adaptive KV cache allocation and sparse attention mechanisms.
"""

from .adaptive_utils import (
    allocate_adaptive_cache_from_importance,
    get_topk_heads_per_layer,
    create_pruning_mask_from_importance,
    save_head_importance_config,
    load_head_importance_config,
    visualize_head_importance,
    print_importance_summary,
)

# Shorter aliases for convenience
allocate_adaptive_cache = allocate_adaptive_cache_from_importance
get_topk_heads = get_topk_heads_per_layer
create_pruning_mask = create_pruning_mask_from_importance

__all__ = [
    # Original names
    "allocate_adaptive_cache_from_importance",
    "get_topk_heads_per_layer",
    "create_pruning_mask_from_importance",
    "save_head_importance_config",
    "load_head_importance_config",
    "visualize_head_importance",
    "print_importance_summary",
    # Shorter aliases
    "allocate_adaptive_cache",
    "get_topk_heads",
    "create_pruning_mask",
]

