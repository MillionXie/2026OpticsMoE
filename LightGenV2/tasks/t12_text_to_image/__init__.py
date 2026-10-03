"""T12: single-pass, text-conditioned object image generation."""

from .modeling import TextConditionedVAE, build_model

__all__ = ["TextConditionedVAE", "build_model"]
