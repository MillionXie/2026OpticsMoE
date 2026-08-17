"""Shared two-plane Vision hybrid backbone for dense prediction tasks."""

from .modeling import (
    LesionSegmentationDecoder,
    PoseHeatmapDecoder,
    SaliencyDensityDecoder,
    Vision2HybridDenseStudent,
    restore_qwen_block_major_spatial,
)
from .settings import apply_vision2_hybrid_settings

__all__ = [
    "LesionSegmentationDecoder",
    "PoseHeatmapDecoder",
    "SaliencyDensityDecoder",
    "Vision2HybridDenseStudent",
    "apply_vision2_hybrid_settings",
    "restore_qwen_block_major_spatial",
]
