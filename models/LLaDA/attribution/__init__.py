"""
LLaDA Attribution Module

Head importance attribution using Integrated Gradients for diffusion language models.
"""

from .head_attribution_block import BlockwiseIntegratedGradientsAttribution

__all__ = [
    "BlockwiseIntegratedGradientsAttribution",
]

