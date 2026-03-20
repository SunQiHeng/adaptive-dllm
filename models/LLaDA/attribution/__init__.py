"""
LLaDA Attribution Module

Head importance attribution using Integrated Gradients for diffusion language models.
"""

__all__ = []

try:
    from .head_attribution_block import BlockwiseIntegratedGradientsAttribution
    __all__.append("BlockwiseIntegratedGradientsAttribution")
except ImportError:
    # Keep attribution package importable even if optional helper files are absent.
    pass

