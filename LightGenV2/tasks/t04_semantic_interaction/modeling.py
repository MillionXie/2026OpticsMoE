"""OpenMoji four-stage optical-Router model and matched dense D2NN."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from LightGenV2.tasks.t01_object_retrieval.modeling import (
    DenseTwoStageOpticalPath,
    hard_topk_load_balance_loss,
)
from LightGenV2.tasks.t01_object_retrieval.settings import (
    load_settings as load_t01_settings,
)
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.modeling import (
    OpenMojiOpticalEditor,
)
from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import (
    BalancedLanguageCore,
    BalancedVisionCore,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_router_retrieval.router import (
    OpticalDetectorTopKRouter,
)

from .settings import Settings


T01_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "t01_object_retrieval/configs/moe_optical_router_scale_matched_dc20.yaml"
)


def _compact(settings: Settings) -> Any:
    compact = load_t01_settings(T01_CONFIG)
    compact.max_visual_tokens = 196
    compact.max_language_tokens = settings.max_language_tokens
    compact.output_dir = settings.output_dir
    compact.optical_fusion_initial = settings.optical_fusion_initial
    compact.fusion_alpha_initial = settings.optical_fusion_initial
    return compact


class LightGenOpenMojiEditor(OpenMojiOpticalEditor):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        compact = _compact(settings)
        self.compact_optical_settings = compact
        is_d2nn = settings.lightgen_model_variant == "d2nn_active_expert_matched"
        self.language_core = BalancedLanguageCore(
            2048, settings.max_language_tokens, compact
        )
        self.vision_core = BalancedVisionCore(1024, 196, compact)
        if is_d2nn:
            self.language_core.optical_branch = DenseTwoStageOpticalPath(
                self.language_core.width,
                compact,
                max_tokens=settings.max_language_tokens,
            )
            self.vision_core.optical_branch = DenseTwoStageOpticalPath(
                self.vision_core.width,
                compact,
                max_tokens=196,
            )
        else:
            for core in (self.language_core, self.vision_core):
                optical = core.optical_branch.core
                optical.router = OpticalDetectorTopKRouter(
                    optical.geometry, compact
                )
        self.router_backend = "none" if is_d2nn else "optical"
        self.router_top_k = None if is_d2nn else 2
        self.checkpoint_architecture = (
            f"lightgen_t04_{settings.lightgen_model_variant}_language2_vision2_"
            f"17um_10cm_dc20_scale_matched_{'contentenergy_v4' if not is_d2nn else 'v1'}"
        )
        self.to(next(self.vision_stem.parameters()).device)

    def router_importance_loss(self) -> torch.Tensor:
        if self.router_backend != "optical":
            return next(self.parameters()).new_zeros(())
        # Balance the actual CCD energy before score standardization.  This is
        # the physical quantity available at deployment and gives the Router
        # phase a substantially stronger gradient than post-softmax
        # importance alone.
        values = []
        for path in self._optical_paths():
            energy_fraction = path.core.last_routing[
                "detector_energy_fraction"
            ].float()
            mean_energy = energy_fraction.mean(dim=0)
            values.append(4.0 * mean_energy.square().sum() - 1.0)
        return torch.stack(values).mean()

    def router_hard_load_balance_loss(self) -> torch.Tensor:
        if self.router_backend != "optical":
            return next(self.parameters()).new_zeros(())
        values = [
            hard_topk_load_balance_loss(
                path.core.last_routing,
                num_experts=4,
                top_k=2,
            )
            for path in self._optical_paths()
        ]
        return torch.stack(values).mean()

    def architecture_report(self) -> dict[str, Any]:
        compact = self.compact_optical_settings
        phase_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if "raw_phase" in name or "raw_router_phase" in name
        )
        return {
            "type": self.settings.lightgen_model_variant,
            "checkpoint_architecture": self.checkpoint_architecture,
            "task": "OpenMoji text-conditioned semantic grid editing",
            "text": "cached frozen Qwen3-VL-2B-Instruct contextual hidden states",
            "vision": "frozen Qwen patch embedding and position embedding; zero native Transformer blocks",
            "hybrid_blocks": {"language": 2, "vision": 2},
            "router": {
                "backend": self.router_backend,
                "top_k": self.router_top_k,
                "experts": 4 if self.router_backend == "optical" else None,
                "training_semantic_top2_codes": (
                    None
                    if self.router_backend == "none"
                    else {
                        "add": [0, 1],
                        "replace": [0, 2],
                        "move": [1, 3],
                        "remove": [2, 3],
                    }
                ),
                "training_vision_top2_code": (
                    None
                    if self.router_backend == "none"
                    else "two highest-energy input-image quadrants"
                ),
                "semantic_code_loss_weight": self.settings.router_semantic_code_weight,
                "balance_quantity": "pre-normalization CCD detector-energy fraction",
                "inference_uses_task_label": False,
            },
            "fusion": {
                "equation": "scale-matched (1-alpha)E + alpha O",
                "alpha_range": [compact.fusion_alpha_min, compact.fusion_alpha_max],
                "alpha_initial": compact.fusion_alpha_initial,
            },
            "optics": {
                "phase_parameters": phase_parameters,
                # Two modalities each activate Top-2 expert masks.  The second
                # block in each branch is the shared/global mask, so it is not
                # part of the activated expert budget used for D2NN matching.
                "active_expert_phase_budget": 2 * 2 * 224**2,
                "feature_captures": 4,
                "router_captures": 0 if self.router_backend == "none" else 2,
                "pixel_pitch_um": 17.0,
                "distance_m": 0.10,
                "zero_order_intensity_range": [0.20, 0.30],
            },
            "decoder": "electronic 6x6 category and edit heads; no attention/Transformer",
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }


def build_model(settings: Settings, device: torch.device) -> LightGenOpenMojiEditor:
    model = LightGenOpenMojiEditor(settings).to(device)
    model.vision_stem.requires_grad_(False).eval()
    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def initialize_from_legacy(
    model: LightGenOpenMojiEditor, settings: Settings
) -> dict[str, Any]:
    path = settings.legacy_warmstart_checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload.get("ema_model", payload["model"])
    target = model.state_dict()
    loaded = []
    skipped = []
    for name, value in target.items():
        source_value = source.get(name)
        optical_specific = (
            ".optical_branch." in name
            or name.endswith("optical_fusion_logit")
        )
        if (
            source_value is not None
            and tuple(source_value.shape) == tuple(value.shape)
            and not optical_specific
        ):
            target[name] = source_value.to(dtype=value.dtype)
            loaded.append(name)
        else:
            skipped.append(name)
    model.load_state_dict(target, strict=True)
    for core in (model.language_core, model.vision_core):
        core.reset_fusion_logits(settings.optical_fusion_initial)
    return {
        "mode": "legacy_electronics_and_decoder_warmstart_fresh_robust_optics",
        "path": str(path),
        "sha256": _sha256(path),
        "source_epoch": int(payload.get("epoch", -1)),
        "loaded_tensor_count": len(loaded),
        "fresh_or_incompatible_tensor_count": len(skipped),
        "fresh_optical_router": model.router_backend == "optical",
        "fusion_logits_reset": True,
    }


__all__ = ["LightGenOpenMojiEditor", "build_model", "initialize_from_legacy"]
