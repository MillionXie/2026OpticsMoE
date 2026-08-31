from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    QwenStemSeparableOpticalImageNetBackbone,
)

from .model import (
    P11_SOURCE_STAGE_COUNT,
    P13_SUPPORTED_DEPTHS,
    QwenStemProgressiveOpticalImageNetBackbone,
)


P11_SIGNATURE_KEY = "p11_separable_architecture_signature"
P11_SIGNATURE = (11, 1, 2, 4)
MIGRATION_FORMAT = "p13-progressive-p11-migration-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_sequence_sha256(phases: Sequence[torch.Tensor]) -> str:
    """Hash an ordered phase sequence, including dtype and tensor shape."""

    digest = hashlib.sha256()
    for index, phase in enumerate(phases):
        value = phase.detach().cpu().contiguous()
        digest.update(f"{index}:{value.dtype}:{tuple(value.shape)}:".encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _official_backbone_payload(
    checkpoint: str | Path,
) -> tuple[Path, Mapping[str, Any], dict[str, torch.Tensor]]:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing P11 backbone checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "backbone" not in payload:
        raise RuntimeError(
            "P13 migration requires the official P11 backbone.pt payload with a "
            "top-level 'backbone' state dict"
        )
    raw_state = payload["backbone"]
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise RuntimeError("P11 payload has no non-empty backbone state dict")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError("P11 backbone state must map string names to tensors")
        if name.startswith("readout."):
            raise RuntimeError("P11 reusable backbone export must not contain its task head")
        state[name] = value
    return path, payload, state


def _validate_p11_identity(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    payload: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
) -> None:
    signature = state.get(P11_SIGNATURE_KEY)
    signature_values = (
        tuple(int(value) for value in signature.tolist())
        if isinstance(signature, torch.Tensor) and tuple(signature.shape) == (4,)
        else None
    )
    if signature_values != P11_SIGNATURE:
        raise RuntimeError(
            f"Expected {P11_SIGNATURE_KEY}={P11_SIGNATURE}; this is not the locked P11 source"
        )
    source_stem_sha = payload.get("stem_checkpoint_sha256")
    if not isinstance(source_stem_sha, str) or not source_stem_sha:
        raise RuntimeError("Official P11 export is missing stem_checkpoint_sha256")
    if source_stem_sha != target.stem.checkpoint_sha256:
        raise RuntimeError(
            "P11 and P13 Qwen stem SHA-256 values differ; refusing a non-reproducible migration"
        )
    report = payload.get("model_report")
    if not isinstance(report, Mapping):
        raise RuntimeError("Official P11 export is missing model_report")
    if report.get("optical_mixer_variant") != "separable_token_channel_axis":
        raise RuntimeError("P11 model_report does not identify the separable optical mixer")
    if report.get("num_stages") != P11_SOURCE_STAGE_COUNT:
        raise RuntimeError("P11 model_report does not contain exactly eight source stages")


def migrate_strict_p11_checkpoint(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Strictly migrate the reusable P11 body into the eight P13 anchors.

    The source head is intentionally absent from the official reusable
    ``backbone.pt``. The frozen Qwen stem, 1024->224 adapter and eight complete
    P11 optical/mixer stages are loaded exactly; every added P13 phase remains
    at its deterministic target initialization and every added depth alpha
    remains under the configured progressive-growth schedule.
    """

    source_path, payload, state = _official_backbone_payload(checkpoint)
    _validate_p11_identity(target, payload, state)

    # Load into an actual P11 instance first. This makes missing, unexpected or
    # shape-incompatible tensors fail before the target is modified.
    source = QwenStemSeparableOpticalImageNetBackbone(
        target.stem_checkpoint,
        target.p11_reference_config(),
    )
    incompatible = source.load_state_dict(state, strict=False)
    expected_missing = {f"readout.{name}" for name in source.readout.state_dict()}
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "P11 backbone export failed strict reusable-body validation: "
            f"missing={sorted(actual_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )

    target.stem.load_state_dict(source.stem.state_dict(), strict=True)
    target.adapter.load_state_dict(source.adapter.state_dict(), strict=True)
    mapping: list[dict[str, Any]] = []
    for source_index, target_index in enumerate(target.anchor_indices):
        source_stage = source.stages[source_index]
        target_slot = target.slots[target_index]
        if target_slot.source_stage_index != source_index:
            raise RuntimeError("Internal P13 anchor schedule is inconsistent")
        if target_slot.stage.optical_axis != source_stage.optical_axis:
            raise RuntimeError("P11/P13 anchor optical axes do not match")
        target_slot.stage.load_state_dict(source_stage.state_dict(), strict=True)
        target_slot.set_alpha(1.0)
        mapping.append(
            {
                "p11_source_stage_zero_based": source_index,
                "p13_target_stage_zero_based": target_index,
                "axis": source_stage.optical_axis,
            }
        )

    source_phase_hash = phase_sequence_sha256(
        [stage.raw_phase for stage in source.stages]
    )
    target_anchor_phase_hash = phase_sequence_sha256(
        [slot.stage.raw_phase for slot in target.anchor_slots()]
    )
    if source_phase_hash != target_anchor_phase_hash:
        raise RuntimeError("Migrated P13 anchor phases do not hash-match P11")

    manifest: dict[str, Any] = {
        "format": MIGRATION_FORMAT,
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "source_stem_checkpoint_sha256": payload["stem_checkpoint_sha256"],
        "source_architecture_signature": list(P11_SIGNATURE),
        "target_architecture_signature": [13, 1, 2, target.num_stages],
        "target_num_stages": target.num_stages,
        "target_optical_phase_parameters": sum(
            parameter.numel() for parameter in target.phase_parameters()
        ),
        "anchor_mapping": mapping,
        "source_phase_sequence_sha256": source_phase_hash,
        "target_anchor_phase_sequence_sha256": target_anchor_phase_hash,
        "phase_hash_domain": "ordered raw_phase tensors including dtype and shape",
        "adapter_migrated": True,
        "stem_buffers_migrated": True,
        "source_imagenet_head_migrated": False,
        "new_stage_phase_initialization": "deterministic target seed schedule",
        "new_stage_identity_skip_parameters": 0,
        "new_stage_depth_alpha": target.depth_alpha_report(),
        "source_best_epoch": payload.get("best_epoch"),
        "source_config_digest": payload.get("config_digest"),
    }
    target.migration_manifest = manifest
    return manifest


def save_migrated_prototype(
    target: QwenStemProgressiveOpticalImageNetBackbone,
    checkpoint: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Materialize a migrated prototype checkpoint and JSON audit manifest."""

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = migrate_strict_p11_checkpoint(target, checkpoint)
    report = target.parameter_report()
    checkpoint_path = output / "p13_migrated_initialization.pt"
    torch.save(
        {
            "format": "p13-progressive-backbone-initialization-v1",
            "backbone": target.backbone_state_dict(),
            "config": target.config,
            "migration_manifest": manifest,
            "model_report": report,
        },
        checkpoint_path,
    )
    complete_manifest = {
        "migration": manifest,
        "model": report,
        "prototype_checkpoint": str(checkpoint_path),
        "prototype_checkpoint_sha256": sha256_file(checkpoint_path),
        "formal_training_started": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return complete_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly migrate eight P11 anchors into a P13 depth prototype"
    )
    parser.add_argument("--stem-checkpoint", type=Path, required=True)
    parser.add_argument("--p11-checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--num-stages", type=int, choices=P13_SUPPORTED_DEPTHS, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--new-stage-alpha-init", type=float, default=0.0)
    parser.add_argument("--new-stage-alpha-epsilon", type=float, default=0.01)
    parser.add_argument("--new-stage-ramp-epochs", type=int, default=10)
    parser.add_argument("--activation-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = QwenStemProgressiveOpticalImageNetBackbone(
        args.stem_checkpoint,
        {
            "num_stages": args.num_stages,
            "seed": args.seed,
            "new_stage_alpha_init": args.new_stage_alpha_init,
            "new_stage_alpha_epsilon": args.new_stage_alpha_epsilon,
            "new_stage_ramp_epochs": args.new_stage_ramp_epochs,
            "activation_checkpointing": args.activation_checkpointing,
        },
    )
    manifest = save_migrated_prototype(
        model,
        args.p11_checkpoint,
        args.output_directory,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
