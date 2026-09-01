from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.train import (
    Context,
    load_config,
    resolve_path,
)
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    QwenStemSeparableOpticalImageNetBackbone,
)

from .train import (
    _verify_p11_epoch88_sources,
    canonical_sha256,
    run_training,
    sha256_tensor,
)


P11_MATCHED_CHECKPOINT_FORMAT = "p13-study-p11-matched-continuation-v1"
P11_MATCHED_EXPORT_FORMAT = "p13-study-p11-matched-backbone-v1"


class P11MatchedContinuationModel(QwenStemSeparableOpticalImageNetBackbone):
    """Expose the P13 growth-training contract for the locked 8-stage control."""

    def new_slots(self) -> tuple[()]:
        return ()

    def carried_slots(self) -> tuple[nn.Module, ...]:
        return tuple(self.stages)

    def new_phase_parameters(self):
        return iter(())

    def carried_phase_parameters(self):
        yield from self.phase_parameters()

    def new_electronic_parameters(self):
        return iter(())

    def carried_electronic_parameters(self):
        yield from self.adapter_parameters()
        yield from self.residual_parameters()

    def apply_depth_ramp(self, epoch: int) -> float:
        del epoch
        return 1.0

    def depth_alpha_report(self) -> dict[str, Any]:
        return {
            "new_stage_count": 0,
            "minimum": 1.0,
            "maximum": 1.0,
            "mean": 1.0,
            "all_full_depth": True,
            "all_exact_bypass": False,
            "control_has_no_growth_gate": True,
        }

    def configure_feedback(self, method: str, *, random_seed: int = 0) -> None:
        del random_seed
        if method != "bp_current":
            raise ValueError("The matched P11 control is intentionally exact BP only")
        for stage in self.stages:
            stage.set_feedback("bp")
        self.feedback_method = "bp_current"

    def feedback_manifest(self) -> dict[str, Any]:
        phases = self.phase_snapshot()
        connections = []
        phase_hashes = []
        for index, stage in enumerate(self.stages):
            phase_hash = sha256_tensor(phases[index])
            phase_hashes.append(phase_hash)
            connections.append(
                {
                    "connector_index_zero_based": index,
                    "axis": stage.optical_axis,
                    "frozen": False,
                    "actual_stage_feedback_mode": stage.feedback_mode,
                    "source_phase_sha256": None,
                    "feedback_phase_sha256": phase_hash,
                    "propagation_transfer_sha256": sha256_tensor(
                        stage.propagator.transfer_function
                    ),
                    "random_substream_seed": None,
                }
            )
        return {
            "format": "p11-matched-current-bp-v1",
            "method": "bp_current",
            "depth": 8,
            "connector_count": 8,
            "random_base_seed": None,
            "source_phase_sequence_sha256": None,
            "feedback_phase_sequence_sha256": canonical_sha256(phase_hashes),
            "connections": connections,
            "control_scope": "locked P11 epoch-88 best plus 20 matched ImageNet epochs",
        }


def initialize_p11_epoch88_control(
    model: nn.Module,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    initialization = config["initialization"]
    if str(initialization.get("mode")) != "p11_epoch88_matched_control":
        raise ValueError("Matched control requires p11_epoch88_matched_control")
    expected_epoch = int(initialization.get("expected_p11_best_epoch", 88))
    backbone_path = resolve_path(initialization["p11_backbone_checkpoint"])
    training_path = resolve_path(initialization["p11_training_checkpoint"])
    identity = _verify_p11_epoch88_sources(initialization, model.config)
    backbone_payload = identity["_backbone_payload"]
    training_payload = identity["_training_payload"]
    if backbone_payload.get("stem_checkpoint_sha256") != model.stem.checkpoint_sha256:
        raise RuntimeError("P11 source stem hash differs from the configured stem")
    training_state = training_payload.get("model")
    backbone_state = backbone_payload.get("backbone")
    if not isinstance(training_state, Mapping) or not isinstance(backbone_state, Mapping):
        raise RuntimeError("P11 source checkpoint state is malformed")
    expected_backbone_keys = {
        name for name in training_state if not str(name).startswith("readout.")
    }
    if set(backbone_state) != expected_backbone_keys:
        raise RuntimeError("P11 backbone export does not match the epoch-88 training state")
    for name in sorted(expected_backbone_keys):
        left = backbone_state[name]
        right = training_state[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise RuntimeError(f"P11 state entry {name} is not a tensor")
        if not torch.equal(left, right):
            raise RuntimeError(f"P11 backbone/training tensor mismatch at {name}")
    model.load_state_dict(training_state, strict=True)
    return {
        "mode": "p11_epoch88_matched_control",
        "source_depth": 8,
        "target_depth": 8,
        "source_best_epoch": expected_epoch,
        "p11_backbone_checkpoint": str(backbone_path),
        "p11_backbone_sha256": identity["backbone_sha256"],
        "p11_training_checkpoint": str(training_path),
        "p11_training_sha256": identity["training_sha256"],
        "p11_config_digest": identity["config_digest"],
        "optimizer_state_reused": False,
        "scheduler_state_reused": False,
        "continuation_schedule_restarted_for_matched_20_epoch_budget": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the epoch-88 P11 matched 20-epoch ImageNet continuation"
    )
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fresh", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    context = Context()
    try:
        run_training(
            load_config(args.config),
            context,
            resume=bool(args.resume),
            model_class=P11MatchedContinuationModel,
            fresh_initializer=initialize_p11_epoch88_control,
            experiment_name="P11 epoch-88 best matched 20-epoch continuation control",
            checkpoint_format=P11_MATCHED_CHECKPOINT_FORMAT,
            export_format=P11_MATCHED_EXPORT_FORMAT,
        )
    finally:
        context.close()


if __name__ == "__main__":
    main()
