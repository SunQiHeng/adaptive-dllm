"""
Utility functions to apply head importance scores to adaptive sparse attention configuration.

Adapted from LLaDA implementation for Dream model architecture.
"""
import torch
from typing import Dict, Optional


def allocate_adaptive_cache_from_importance(
    importance_scores: Dict[int, torch.Tensor],
    total_budget: int,
    min_cache_size: int = 64,
    max_cache_size: Optional[int] = None,
    normalize_per_layer: bool = True
) -> torch.Tensor:
    """
    Allocate adaptive KV cache sizes based on head importance scores.
    
    根据 head 重要性分数分配每个 head 的 cache 大小。
    
    Args:
        importance_scores: Dictionary mapping layer_idx -> importance tensor of shape (n_heads,)
        total_budget: Total KV cache budget (total number of tokens to cache across all heads in a layer)
        min_cache_size: Minimum cache size for each head
        max_cache_size: Maximum cache size for each head (optional)
        normalize_per_layer: Whether to normalize importance scores per layer
    
    Returns:
        Tensor of shape (n_layers, n_heads) containing cache sizes for each head
    
    Example:
        >>> importance = {0: torch.tensor([0.5, 0.3, 0.2]), 1: torch.tensor([0.4, 0.4, 0.2])}
        >>> cache_sizes = allocate_adaptive_cache_from_importance(
        ...     importance, total_budget=300, min_cache_size=50
        ... )
        >>> print(cache_sizes)  # shape: (2, 3)
    """
    n_layers = len(importance_scores)
    n_heads = importance_scores[0].shape[0]
    
    # Initialize cache size tensor
    cache_sizes = torch.zeros(n_layers, n_heads)
    
    for layer_idx in range(n_layers):
        scores = importance_scores[layer_idx]
        
        # Normalize scores within the layer
        if normalize_per_layer:
            scores = scores / (scores.sum() + 1e-8)
        
        # Calculate remaining budget after allocating minimum to each head
        remaining_budget = total_budget - min_cache_size * n_heads
        
        if remaining_budget < 0:
            raise ValueError(
                f"Total budget {total_budget} is too small for "
                f"{n_heads} heads with min_cache_size={min_cache_size}"
            )
        
        # Allocate remaining budget proportionally to importance scores
        allocated_sizes = min_cache_size + (scores * remaining_budget)
        
        # Apply max_cache_size constraint if specified
        if max_cache_size is not None:
            allocated_sizes = torch.clamp(allocated_sizes, min=min_cache_size, max=max_cache_size)
            
            # If we clipped some heads, redistribute the excess to others
            excess = allocated_sizes.sum() - total_budget
            if excess < 0:
                # We have leftover budget, redistribute to non-maxed heads
                non_maxed_mask = allocated_sizes < max_cache_size
                if non_maxed_mask.any():
                    num_non_maxed = non_maxed_mask.sum()
                    allocated_sizes[non_maxed_mask] += (-excess) / num_non_maxed
        
        cache_sizes[layer_idx] = allocated_sizes.round()
        
        # Final adjustment to ensure exact budget
        current_total = cache_sizes[layer_idx].sum()
        diff = total_budget - current_total
        if diff != 0:
            # Add/subtract difference to the most/least important head
            if diff > 0:
                idx = scores.argmax()
            else:
                idx = scores.argmin()
            cache_sizes[layer_idx, idx] += diff
    
    return cache_sizes.int()


def get_topk_heads_per_layer(
    importance_scores: Dict[int, torch.Tensor],
    k: int
) -> Dict[int, torch.Tensor]:
    """
    Get top-k most important heads for each layer.
    
    获取每一层中最重要的 k 个 heads。
    
    Args:
        importance_scores: Dictionary mapping layer_idx -> importance tensor of shape (n_heads,)
        k: Number of top heads to select
    
    Returns:
        Dictionary mapping layer_idx -> tensor of head indices (shape: (k,))
    
    Example:
        >>> importance = {0: torch.tensor([0.5, 0.3, 0.2, 0.4])}
        >>> top_heads = get_topk_heads_per_layer(importance, k=2)
        >>> print(top_heads[0])  # tensor([0, 3]) - indices of top-2 heads
    """
    topk_heads = {}
    
    for layer_idx, scores in importance_scores.items():
        n_heads = scores.shape[0]
        if k > n_heads:
            raise ValueError(f"k={k} is larger than number of heads={n_heads}")
        
        # Get top-k indices
        _, indices = torch.topk(scores, k=k, largest=True)
        topk_heads[layer_idx] = indices
    
    return topk_heads


def create_pruning_mask_from_importance(
    importance_scores: Dict[int, torch.Tensor],
    keep_ratio: float = 0.75
) -> Dict[int, torch.Tensor]:
    """
    Create a pruning mask based on importance scores.
    
    根据重要性分数创建剪枝 mask。
    
    Args:
        importance_scores: Dictionary mapping layer_idx -> importance tensor of shape (n_heads,)
        keep_ratio: Ratio of heads to keep (0.0 to 1.0)
    
    Returns:
        Dictionary mapping layer_idx -> boolean mask (shape: (n_heads,))
        True means keep the head, False means prune
    
    Example:
        >>> importance = {0: torch.tensor([0.5, 0.3, 0.2, 0.4])}
        >>> mask = create_pruning_mask_from_importance(importance, keep_ratio=0.5)
        >>> print(mask[0])  # tensor([True, False, False, True]) - keep top-50% heads
    """
    if not 0.0 <= keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be between 0.0 and 1.0, got {keep_ratio}")
    
    pruning_masks = {}
    
    for layer_idx, scores in importance_scores.items():
        n_heads = scores.shape[0]
        k = max(1, int(n_heads * keep_ratio))  # Keep at least 1 head
        
        # Create mask: True for top-k heads, False for others
        threshold = torch.topk(scores, k=k, largest=True)[0].min()
        mask = scores >= threshold
        
        pruning_masks[layer_idx] = mask
    
    return pruning_masks


def save_head_importance_config(
    importance_scores: Dict[int, torch.Tensor],
    cache_sizes: Optional[torch.Tensor],
    output_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save head importance scores and cache configuration to file.
    
    保存 head 重要性分数和 cache 配置到文件。
    
    Args:
        importance_scores: Dictionary mapping layer_idx -> importance tensor
        cache_sizes: Tensor of cache sizes (n_layers, n_heads) or None
        output_path: Path to save the configuration
        metadata: Optional metadata dictionary
    """
    save_dict = {
        'importance_scores': {k: v.cpu() for k, v in importance_scores.items()},
        'metadata': metadata or {}
    }
    
    if cache_sizes is not None:
        save_dict['cache_sizes'] = cache_sizes.cpu()
    
    torch.save(save_dict, output_path)
    print(f"Saved head importance configuration to {output_path}")


def load_head_importance_config(load_path: str) -> Dict:
    """
    Load head importance scores and cache configuration from file.
    
    从文件加载 head 重要性分数和 cache 配置。
    
    Args:
        load_path: Path to load the configuration from
    
    Returns:
        Dictionary containing importance_scores, cache_sizes (if present), and metadata
    """
    config = torch.load(load_path)
    print(f"Loaded head importance configuration from {load_path}")
    return config


def visualize_head_importance(
    importance_scores: Dict[int, torch.Tensor],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Visualize head importance scores as a heatmap.
    
    将 head 重要性可视化为热力图。
    
    Args:
        importance_scores: Dictionary mapping layer_idx -> importance tensor
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib is required for visualization. Install it with: pip install matplotlib")
        return
    
    n_layers = len(importance_scores)
    n_heads = importance_scores[0].shape[0]
    
    # Create matrix for heatmap
    importance_matrix = np.zeros((n_layers, n_heads))
    for layer_idx in range(n_layers):
        importance_matrix[layer_idx] = importance_scores[layer_idx].cpu().numpy()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    ax.set_xlabel('Head Index', fontsize=12)
    ax.set_ylabel('Layer Index', fontsize=12)
    ax.set_title('Head Importance Scores Across Layers', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Importance Score', fontsize=12)
    
    # Set ticks
    ax.set_xticks(np.arange(n_heads))
    ax.set_yticks(np.arange(n_layers))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_importance_summary(importance_scores: Dict[int, torch.Tensor]):
    """
    Print a summary of head importance scores.
    
    打印 head 重要性分数的摘要。
    """
    print("\n" + "="*70)
    print("HEAD IMPORTANCE SUMMARY")
    print("="*70)
    
    for layer_idx in sorted(importance_scores.keys()):
        scores = importance_scores[layer_idx]
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Mean importance: {scores.mean():.4f}")
        print(f"  Std importance:  {scores.std():.4f}")
        print(f"  Max importance:  {scores.max():.4f} (head {scores.argmax().item()})")
        print(f"  Min importance:  {scores.min():.4f} (head {scores.argmin().item()})")
        
        # Show top-3 and bottom-3
        sorted_indices = torch.argsort(scores, descending=True)
        top_3 = [(idx.item(), scores[idx].item()) for idx in sorted_indices[:3]]
        bottom_3 = [(idx.item(), scores[idx].item()) for idx in sorted_indices[-3:]]
        
        print(f"  Top-3 heads:    {', '.join([f'{idx} ({score:.4f})' for idx, score in top_3])}")
        print(f"  Bottom-3 heads: {', '.join([f'{idx} ({score:.4f})' for idx, score in bottom_3])}")
    
    print("\n" + "="*70)


def generate_random_head_importance(
    n_layers: int,
    n_heads: int,
    distribution: str = 'uniform',
    min_val: float = 0.1,
    max_val: float = 1.0,
    seed: Optional[int] = None
) -> Dict[int, torch.Tensor]:
    """
    Generate random head importance scores for testing/initialization.
    
    生成随机的 head 重要性分数用于测试/初始化。
    
    Args:
        n_layers: Number of layers
        n_heads: Number of heads per layer
        distribution: Distribution type ('uniform', 'normal', 'exponential')
        min_val: Minimum importance value
        max_val: Maximum importance value
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping layer_idx -> importance tensor of shape (n_heads,)
    
    Example:
        >>> importance = generate_random_head_importance(
        ...     n_layers=4, n_heads=8, distribution='uniform', seed=42
        ... )
        >>> print(importance[0].shape)  # torch.Size([8])
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    importance_scores = {}
    
    for layer_idx in range(n_layers):
        if distribution == 'uniform':
            scores = torch.rand(n_heads) * (max_val - min_val) + min_val
        elif distribution == 'normal':
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 4
            scores = torch.randn(n_heads) * std + mean
            scores = torch.clamp(scores, min_val, max_val)
        elif distribution == 'exponential':
            # Exponential distribution favors smaller values
            scores = torch.rand(n_heads) ** 2 * (max_val - min_val) + min_val
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        # Normalize to sum to n_heads for consistency
        scores = scores / scores.sum() * n_heads
        importance_scores[layer_idx] = scores
    
    return importance_scores


def allocate_adaptive_sparsity_from_importance(
    importance_scores: Dict[int, torch.Tensor],
    base_sparsity: float = 0.5,
    min_sparsity: float = 0.1,
    max_sparsity: float = 0.9,
    inverse_importance: bool = True
) -> Dict[int, torch.Tensor]:
    """
    Allocate per-head sparsity levels based on importance scores.
    
    根据重要性分数为每个 head 分配稀疏度。
    
    For adaptive sparse attention, less important heads can be more sparse,
    while more important heads should be kept denser.
    
    Args:
        importance_scores: Dictionary mapping layer_idx -> importance tensor of shape (n_heads,)
        base_sparsity: Base sparsity level (0.0 to 1.0, where 1.0 means fully sparse)
        min_sparsity: Minimum sparsity level for any head
        max_sparsity: Maximum sparsity level for any head
        inverse_importance: If True, less important heads get higher sparsity
    
    Returns:
        Dictionary mapping layer_idx -> sparsity tensor of shape (n_heads,)
        Each value represents the fraction of tokens to KEEP (1.0 - sparsity)
    
    Example:
        >>> importance = {0: torch.tensor([0.5, 0.3, 0.2])}
        >>> sparsity = allocate_adaptive_sparsity_from_importance(
        ...     importance, base_sparsity=0.5, min_sparsity=0.2, max_sparsity=0.8
        ... )
        >>> print(sparsity[0])  # Higher importance -> lower sparsity (more tokens kept)
    """
    sparsity_levels = {}
    
    for layer_idx, scores in importance_scores.items():
        # Normalize importance scores to [0, 1]
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        if inverse_importance:
            # Less important heads get higher sparsity (fewer tokens kept)
            # More important heads get lower sparsity (more tokens kept)
            sparsity = max_sparsity - normalized_scores * (max_sparsity - min_sparsity)
        else:
            # More important heads get higher sparsity
            sparsity = min_sparsity + normalized_scores * (max_sparsity - min_sparsity)
        
        # Convert sparsity to keep_ratio (fraction of tokens to KEEP)
        keep_ratio = 1.0 - sparsity
        sparsity_levels[layer_idx] = keep_ratio
    
    return sparsity_levels


def create_adaptive_sparsity_config(
    n_layers: int,
    n_heads: int,
    importance_scores: Optional[Dict[int, torch.Tensor]] = None,
    strategy: str = 'random',
    base_sparsity: float = 0.5,
    min_sparsity: float = 0.1,
    max_sparsity: float = 0.9,
    seed: Optional[int] = None
) -> Dict:
    """
    Create a complete adaptive sparsity configuration.
    
    创建完整的自适应稀疏配置。
    
    Args:
        n_layers: Number of layers
        n_heads: Number of heads per layer
        importance_scores: Pre-computed importance scores (optional)
        strategy: Strategy for generating importance ('random', 'uniform', 'normal')
        base_sparsity: Base sparsity level
        min_sparsity: Minimum sparsity level for any head
        max_sparsity: Maximum sparsity level for any head
        seed: Random seed
    
    Returns:
        Dictionary containing:
            - importance_scores: Head importance scores
            - sparsity_levels: Per-head sparsity levels (keep ratios)
            - metadata: Configuration metadata
    """
    # Generate or use provided importance scores
    if importance_scores is None:
        importance_scores = generate_random_head_importance(
            n_layers=n_layers,
            n_heads=n_heads,
            distribution=strategy,
            seed=seed
        )
    
    # Allocate sparsity based on importance
    sparsity_levels = allocate_adaptive_sparsity_from_importance(
        importance_scores=importance_scores,
        base_sparsity=base_sparsity,
        min_sparsity=min_sparsity,
        max_sparsity=max_sparsity
    )
    
    config = {
        'importance_scores': importance_scores,
        'sparsity_levels': sparsity_levels,
        'metadata': {
            'n_layers': n_layers,
            'n_heads': n_heads,
            'base_sparsity': base_sparsity,
            'strategy': strategy,
            'seed': seed
        }
    }
    
    return config


def print_adaptive_sparsity_summary(config: Dict):
    """
    Print a summary of adaptive sparsity configuration.
    
    打印自适应稀疏配置摘要。
    """
    print("\n" + "="*70)
    print("ADAPTIVE SPARSITY CONFIGURATION")
    print("="*70)
    
    metadata = config.get('metadata', {})
    print(f"\nMetadata:")
    print(f"  Layers: {metadata.get('n_layers', 'N/A')}")
    print(f"  Heads per layer: {metadata.get('n_heads', 'N/A')}")
    print(f"  Base sparsity: {metadata.get('base_sparsity', 'N/A')}")
    print(f"  Strategy: {metadata.get('strategy', 'N/A')}")
    
    importance_scores = config['importance_scores']
    sparsity_levels = config['sparsity_levels']
    
    print("\nPer-layer Statistics:")
    for layer_idx in sorted(importance_scores.keys()):
        imp_scores = importance_scores[layer_idx]
        keep_ratios = sparsity_levels[layer_idx]
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Importance: mean={imp_scores.mean():.4f}, std={imp_scores.std():.4f}")
        print(f"  Keep ratio: mean={keep_ratios.mean():.4f}, min={keep_ratios.min():.4f}, max={keep_ratios.max():.4f}")
        print(f"  Effective sparsity: {(1 - keep_ratios.mean()):.2%}")
    
    print("\n" + "="*70)

