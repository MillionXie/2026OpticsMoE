from __future__ import annotations

from typing import Any, Mapping

import torch

from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.modeling import (
    ElectronicRetrievalReadout,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.modeling import (
    FourLayerOpticalReplacement as RobustFourLayerOpticalReplacement,
    load_backbone,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust.optical_blocks import (
    LanguageTwoBlockOpticalReplacement,
    VisionTwoBlockOpticalReplacement,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5.modeling import (
    STAGE_ARCHITECTURES,
    _load_payload,
    _validate_metadata,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
)

from .router import FairElectronicAmplitudeRouter, OpticalDetectorTopKRouter
from .artifacts import (
    save_phase_preview as save_router_phase_preview,
    save_phase_snapshot as save_router_phase_snapshot,
)


ROUTER_PREFIX = "core.optical_branch.core.router."


def architecture_label(settings: Any) -> str:
    ste = "ste" if settings.router_straight_through else "noste"
    return (
        "vision2_language2_moe4_10cm_router_ablation_"
        f"{settings.router_backend}_k{settings.top_k}_"
        f"{settings.router_weight_normalization}_{ste}_"
        f"c{settings.router_contract_sha256[:12]}_v2"
    )


def _router_for(settings: Any, geometry: Any) -> torch.nn.Module:
    if settings.router_backend == "electronic":
        return FairElectronicAmplitudeRouter(geometry, settings)
    if settings.router_backend == "optical":
        return OpticalDetectorTopKRouter(geometry, settings)
    raise RuntimeError(f"Unsupported router backend {settings.router_backend!r}")


def _install_router(surrogate: torch.nn.Module, settings: Any) -> None:
    core = surrogate.core.optical_branch.core
    core.router = _router_for(settings, core.geometry)


class RouterAblationReplacement(RobustFourLayerOpticalReplacement):
    """Warmstart5 body with an explicitly contracted router implementation."""

    training_architecture_label = "vision2_language2_moe4_10cm_router_ablation"

    def __init__(self, *args: Any, settings: Any, **kwargs: Any) -> None:
        self.router_backend = str(settings.router_backend)
        self.router_top_k = int(settings.top_k)
        self.router_weight_normalization = str(
            settings.router_weight_normalization
        )
        self.router_straight_through = bool(settings.router_straight_through)
        super().__init__(*args, settings=settings, **kwargs)
        self.checkpoint_architecture = architecture_label(settings)

    def router_parameters(self) -> list[torch.nn.Parameter]:
        return [
            *self.vision_surrogate.core.optical_branch.core.router.parameters(),
            *self.language_surrogate.core.optical_branch.core.router.parameters(),
        ]

    def phase_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        # Keep the shared phase CSV schema on its audited four feature groups.
        # Router phase belongs to the independent router optimizer group and
        # is saved by our custom phase snapshots/previews.  Adding new group
        # names here would make the legacy fixed-column CSV writer reject the
        # extra diagnostic fields after a completed epoch.
        return super().phase_parameter_groups()

    def student_architecture_report(self) -> dict[str, Any]:
        report = super().student_architecture_report()
        router_parameters = sum(
            parameter.numel() for parameter in self.router_parameters()
        )
        report.update(
            {
                "type": self.training_architecture_label,
                "checkpoint_architecture": self.checkpoint_architecture,
                "initialization": "strict_warmstart5_stage_b_ema_except_router_phase",
                "router": {
                    "backend": self.router_backend,
                    "vision_language_share_parameters": False,
                    "vision_global_reuses_vision_expert_route": True,
                    "language_global_reuses_language_expert_route": True,
                    "top_k": self.router_top_k,
                    "weight_normalization": self.router_weight_normalization,
                    "straight_through_gradient": self.router_straight_through,
                    "trainable_parameters": router_parameters,
                    "electronic_trainable_head_after_router_ccd": False,
                    "extra_ccd_exposures": (
                        2 if self.router_backend == "optical" else 0
                    ),
                },
                "physical_feature_stage_count": 4,
                "physical_capture_count_with_router": (
                    6 if self.router_backend == "optical" else 4
                ),
            }
        )
        return report

    def save_multiplane_phase_snapshot(
        self,
        output_dir: Any,
        *,
        epoch: int,
        train_loss: float,
        weight_variant: str,
    ) -> dict[str, Any]:
        return save_router_phase_snapshot(
            self,
            output_dir,
            epoch=epoch,
            train_loss=train_loss,
            weight_variant=weight_variant,
        )

    def save_multiplane_phase_preview(self, path: Any, *, title: str) -> None:
        save_router_phase_preview(self, path, title=title)


def build_hybrid_student(
    loaded: LoadedBackbone, settings: Any
) -> tuple[RouterAblationReplacement, ElectronicRetrievalReadout]:
    settings.resolve_architecture(loaded.model)
    vision = VisionTwoBlockOpticalReplacement(
        settings.vision_hidden_size, settings
    )
    language = LanguageTwoBlockOpticalReplacement(
        settings.text_hidden_size, settings
    )
    _install_router(vision, settings)
    _install_router(language, settings)
    vision = vision.to(loaded.device)
    language = language.to(loaded.device)
    replacement = RouterAblationReplacement(
        loaded.model, vision, language, settings=settings
    )
    readout = ElectronicRetrievalReadout(
        settings.detector_output_size, settings.embedding_dim
    ).to(loaded.device)
    replacement.configure_student_trainability()
    readout.requires_grad_(True)
    return replacement, readout


def _copy_surrogate_state(
    target: Mapping[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    *,
    reset_router: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not reset_router:
        if set(target) != set(source):
            raise RuntimeError(
                "Electronic router state is not exactly compatible with warmstart5: "
                f"missing={sorted(set(target).difference(source))} "
                f"unexpected={sorted(set(source).difference(target))}"
            )
        output: dict[str, torch.Tensor] = {}
        for key, target_value in target.items():
            source_value = source[key]
            if tuple(source_value.shape) != tuple(target_value.shape):
                raise RuntimeError(f"Warmstart tensor shape mismatch for {key}")
            output[key] = source_value.detach().clone()
        return output, {
            "loaded_tensor_count": len(output),
            "new_router_tensor_count": 0,
            "discarded_electronic_router_tensor_count": 0,
        }

    target_body = {key for key in target if not key.startswith(ROUTER_PREFIX)}
    source_body = {key for key in source if not key.startswith(ROUTER_PREFIX)}
    if target_body != source_body:
        raise RuntimeError(
            "Warmstart5 body mismatch while replacing only its router: "
            f"missing={sorted(target_body.difference(source_body))} "
            f"unexpected={sorted(source_body.difference(target_body))}"
        )
    output = {key: value.detach().clone() for key, value in target.items()}
    for key in sorted(target_body):
        if tuple(source[key].shape) != tuple(target[key].shape):
            raise RuntimeError(f"Warmstart tensor shape mismatch for {key}")
        output[key] = source[key].detach().clone()
    target_router = {key for key in target if key.startswith(ROUTER_PREFIX)}
    source_router = {key for key in source if key.startswith(ROUTER_PREFIX)}
    return output, {
        "loaded_tensor_count": len(target_body),
        "new_router_tensor_count": len(target_router),
        "new_router_tensor_keys": sorted(target_router),
        "discarded_source_router_tensor_count": len(source_router),
        "discarded_source_router_tensor_keys": sorted(source_router),
    }


def load_warmstart5_initialization(
    settings: Any,
    replacement: RouterAblationReplacement,
    readout: ElectronicRetrievalReadout,
) -> dict[str, Any]:
    payload, digest = _load_payload(
        settings.router_source_checkpoint, settings.router_source_sha256
    )
    _validate_metadata(
        payload,
        settings,
        expected_architecture=STAGE_ARCHITECTURES["joint"],
    )
    reports: dict[str, Any] = {}
    for name, surrogate in (
        ("vision", replacement.vision_surrogate),
        ("language", replacement.language_surrogate),
    ):
        state, report = _copy_surrogate_state(
            surrogate.state_dict(),
            payload[f"{name}_optical"],
            reset_router=settings.router_reset_parameters,
        )
        surrogate.load_state_dict(state, strict=True)
        reports[name] = report
    readout.load_state_dict(payload["retrieval_readout"], strict=True)
    return {
        "mode": "warmstart5_stage_b_ema_router_ablation",
        "path": str(settings.router_source_checkpoint),
        "sha256": digest,
        "source_epoch": int(payload["epoch"]),
        "source_train_loss": float(payload["train_loss"]),
        "source_checkpoint_architecture": STAGE_ARCHITECTURES["joint"],
        "target_checkpoint_architecture": replacement.checkpoint_architecture,
        "router_backend": settings.router_backend,
        "top_k": settings.top_k,
        "weight_normalization": settings.router_weight_normalization,
        "straight_through": settings.router_straight_through,
        "router_optimization_seed": settings.router_optimization_seed,
        "dataset_and_batch_seed": settings.random_seed,
        "router_parameters_reset_from_fixed_seed": settings.router_reset_parameters,
        "surrogates": reports,
        "test_metrics_used_for_initialization_selection": False,
    }


__all__ = [
    "ROUTER_PREFIX",
    "RouterAblationReplacement",
    "architecture_label",
    "build_hybrid_student",
    "load_backbone",
    "load_warmstart5_initialization",
]
