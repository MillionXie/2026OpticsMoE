"""Frozen Qwen3-VL patch stem followed by an eight-stage optical backbone."""

from .model import QwenStemOpticalImageNetBackbone
from .stem import StaticQwenPatchStem

__all__ = ["QwenStemOpticalImageNetBackbone", "StaticQwenPatchStem"]
