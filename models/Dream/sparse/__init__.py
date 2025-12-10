"""Dream Sparse Attention Module"""
from .adaptive_utils_dream import (
    allocate_adaptive_cache_from_importance,
    get_topk_heads_per_layer,
    create_pruning_mask_from_importance,
    save_head_importance_config,
    load_head_importance_config,
    visualize_head_importance,
    print_importance_summary,
    generate_random_head_importance,
    allocate_adaptive_sparsity_from_importance,
    create_adaptive_sparsity_config,
    print_adaptive_sparsity_summary,
)

__all__ = [
    'allocate_adaptive_cache_from_importance',
    'get_topk_heads_per_layer',
    'create_pruning_mask_from_importance',
    'save_head_importance_config',
    'load_head_importance_config',
    'visualize_head_importance',
    'print_importance_summary',
    'generate_random_head_importance',
    'allocate_adaptive_sparsity_from_importance',
    'create_adaptive_sparsity_config',
    'print_adaptive_sparsity_summary',
]

