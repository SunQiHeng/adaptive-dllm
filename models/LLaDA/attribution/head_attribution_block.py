"""
Compatibility shim.

Historically `BlockwiseIntegratedGradientsAttribution` lived at:
  models/LLaDA/attribution/head_attribution_block.py

Some experiments moved the implementation into:
  models/LLaDA/attribution/ouput_attribution/head_attribution_block.py

Keep both import paths working to avoid breaking older scripts and package imports.
"""

from .ouput_attribution.head_attribution_block import BlockwiseIntegratedGradientsAttribution

__all__ = ["BlockwiseIntegratedGradientsAttribution"]


