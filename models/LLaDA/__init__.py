"""
LLaDA: Latent Diffusion for Language Generation

A clean and modular implementation with support for:
- Core model architecture
- Diffusion-based text generation  
- Head importance attribution
- Sparse attention mechanisms

Modules:
- core: Model architecture and configuration
- generation: Text generation functions
- attribution: Head importance attribution
- sparse: Sparse attention utilities
- docs: Documentation

Quick Start:
```python
from models.LLaDA import LLaDAConfig, LLaDAModelLM, generate

# Load model
config = LLaDAConfig.from_pretrained("path/to/model")
model = LLaDAModelLM.from_pretrained("path/to/model", config=config)

# Generate
output = generate(model, prompt, steps=128, gen_length=128)
```
"""

# Core
from .core import LLaDAConfig, LLaDAModel, LLaDAModelLM, ModelConfig

# Generation
from .generation import generate

# Attribution
from .attribution import BlockwiseIntegratedGradientsAttribution

# Sparse
from .sparse import (
    allocate_adaptive_cache_from_importance,
    allocate_adaptive_cache,  # alias
    get_topk_heads_per_layer,
    get_topk_heads,  # alias
    create_pruning_mask_from_importance,
    create_pruning_mask,  # alias
    save_head_importance_config,
    load_head_importance_config,
    visualize_head_importance,
    print_importance_summary,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "LLaDAConfig",
    "ModelConfig",
    "LLaDAModel",
    "LLaDAModelLM",
    # Generation
    "generate",
    # Attribution
    "BlockwiseIntegratedGradientsAttribution",
    # Sparse (full names)
    "allocate_adaptive_cache_from_importance",
    "get_topk_heads_per_layer",
    "create_pruning_mask_from_importance",
    "save_head_importance_config",
    "load_head_importance_config",
    "visualize_head_importance",
    "print_importance_summary",
    # Sparse (aliases)
    "allocate_adaptive_cache",
    "get_topk_heads",
    "create_pruning_mask",
]
