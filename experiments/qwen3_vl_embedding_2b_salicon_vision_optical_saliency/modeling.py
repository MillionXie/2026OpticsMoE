from __future__ import annotations

from typing import Any

import torch

from experiments.qwen3_vl_embedding_2b_fss1000_vision_optical_saliency.modeling import (
    FrozenQwenVisionTeacher,
    LightweightSegmentationHead,
    LoadedVisionBackbone,
    VisionOpticalSaliencyStudent,
    load_vision_backbone,
    preprocess_vision,
    restore_detector_spatial,
    restore_packed_spatial,
    trainable_parameter_report,
)


class ContinuousSaliencyHead(LightweightSegmentationHead):
    """Spatial decoder whose raw logits parameterize a normalized density map."""


def build_teacher(
    loaded: LoadedVisionBackbone, settings: Any
) -> FrozenQwenVisionTeacher:
    head = ContinuousSaliencyHead(
        input_dim=settings.vision_hidden_size,
        projection_dim=settings.segmentation_projection_dim,
        decoder_channels=settings.segmentation_channels,
        groupnorm_groups=settings.segmentation_groupnorm_groups,
        output_size=settings.image_size,
        refinement_enabled=False,
        progressive_refinement_enabled=False,
        detector_residual_enabled=False,
    ).to(loaded.device)
    return FrozenQwenVisionTeacher(loaded, head)


def build_student(
    loaded: LoadedVisionBackbone, settings: Any
) -> VisionOpticalSaliencyStudent:
    if getattr(settings, "vision2_hybrid_enabled", False):
        from experiments.vision2_hybrid_dense.modeling import (
            SaliencyDensityDecoder,
            Vision2HybridDenseStudent,
        )

        return Vision2HybridDenseStudent(
            loaded,
            settings,
            SaliencyDensityDecoder(
                input_dim=settings.electronic_width,
                output_size=settings.image_size,
            ),
        )
    head = ContinuousSaliencyHead(
        input_dim=settings.detector_output_size,
        projection_dim=settings.segmentation_projection_dim,
        decoder_channels=settings.segmentation_channels,
        groupnorm_groups=settings.segmentation_groupnorm_groups,
        output_size=settings.image_size,
        refinement_enabled=False,
        progressive_refinement_enabled=False,
        detector_residual_enabled=False,
    ).to(loaded.device)
    student = VisionOpticalSaliencyStudent(loaded, settings, head)
    loaded.model.requires_grad_(False)
    student.core.requires_grad_(True)
    student.core.output_adapter.requires_grad_(False)
    student.head.requires_grad_(True)
    return student


def assert_student_trainability(student: VisionOpticalSaliencyStudent) -> None:
    names = {
        name
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    if hasattr(student.core, "hybrid"):
        required_fragments = (
            "core.hybrid.input_adapter",
            "core.hybrid.blocks.0.token_depthwise",
            "core.hybrid.optical_branch.core.router",
            "core.hybrid.optical_branch.core.expert_layers",
            "core.hybrid.optical_branch.core.global_phase",
            "head.token_projection",
            "head.classifier",
        )
    else:
        required_fragments = (
            "core.input_adapter",
            "core.router",
            "core.expert_layers",
            "core.global_phase",
            "head.token_projection",
            "head.classifier",
        )
    missing = [
        fragment
        for fragment in required_fragments
        if not any(fragment in name for name in names)
    ]
    if missing:
        raise RuntimeError(
            f"Optical saliency Student has missing trainable groups: {missing}"
        )
    forbidden = [
        name
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
        and (
            "visual.patch_embed" in name
            or name in {"core.output_adapter.weight", "core.output_adapter.bias"}
            or name in {
                "core.hybrid.output_adapter.weight",
                "core.hybrid.output_adapter.bias",
            }
        )
    ]
    if forbidden:
        raise RuntimeError(f"Unexpected trainable frozen parameters: {forbidden}")


__all__ = [
    "ContinuousSaliencyHead",
    "FrozenQwenVisionTeacher",
    "LoadedVisionBackbone",
    "VisionOpticalSaliencyStudent",
    "assert_student_trainability",
    "build_student",
    "build_teacher",
    "load_vision_backbone",
    "preprocess_vision",
    "restore_detector_spatial",
    "restore_packed_spatial",
    "trainable_parameter_report",
]

