"""
LLaDA Attribution Module

Head importance attribution using Integrated Gradients for diffusion language models.
"""

from .head_attribution import IntegratedGradientsHeadAttribution
from .head_attribution_v2 import BlockwiseIntegratedGradientsAttribution

__all__ = [
    "IntegratedGradientsHeadAttribution",
    "BlockwiseIntegratedGradientsAttribution",
]

