"""Strict P11-backbone-only initialization for a new large-vocabulary head."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_imagenet_backbone.model import (
    QwenStemSeparableOpticalImageNetBackbone,
)


BACKBONE_ASSET_FORMATS = {
    "qwen-static-stem-separable-optical-imagenet-backbone-v1",
    "p11-large-scale-supervised-backbone-v1",
}


class InitializationContractError(RuntimeError):
    """The immutable P11 source does not match the declared identity."""


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def construct_large_vocabulary_model(
    *,
    stem_checkpoint: str | Path,
    model_config: Mapping[str, Any],
) -> QwenStemSeparableOpticalImageNetBackbone:
    config = dict(model_config)
    target_classes = int(config.get("num_classes", 0))
    if target_classes not in {10_450, 11_221, 21_841}:
        raise InitializationContractError(
            "Large-vocabulary head must explicitly declare 10450, 11221 or 21841 classes"
        )
    if int(config.get("num_stages", 8)) != 8:
        raise InitializationContractError("This project is locked to the eight-stage P11 backbone")
    model = QwenStemSeparableOpticalImageNetBackbone(stem_checkpoint, config)
    if model.readout.classifier[-1].out_features != target_classes:
        raise InitializationContractError("Constructed readout does not match target class count")
    return model


def initialize_from_frozen_p11_backbone(
    model: nn.Module,
    *,
    backbone_checkpoint: str | Path,
    expected_backbone_sha256: str,
    expected_stem_sha256: str,
) -> dict[str, Any]:
    """Load only reusable tensors and prove that the new readout was untouched.

    The source checkpoint must contain ``backbone`` and must not contain any
    ``readout.*`` key.  ``load_state_dict(strict=False)`` is accepted only when
    its missing keys are exactly the target model's newly constructed readout.
    """

    path = Path(backbone_checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen P11 backbone is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != str(expected_backbone_sha256):
        raise InitializationContractError(
            f"P11 backbone SHA mismatch: expected {expected_backbone_sha256}, got {actual_sha}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise InitializationContractError("Backbone checkpoint root must be a mapping")
    payload_format = payload.get("format")
    if payload_format is not None and payload_format not in BACKBONE_ASSET_FORMATS:
        raise InitializationContractError(f"Unsupported backbone format: {payload_format!r}")
    source_state = payload.get("backbone")
    if not isinstance(source_state, Mapping):
        raise InitializationContractError("Backbone checkpoint has no 'backbone' state mapping")
    source_state = dict(source_state)
    forbidden = sorted(name for name in source_state if name.startswith("readout."))
    if forbidden:
        raise InitializationContractError(
            f"Frozen backbone illegally contains the 1000-class readout: {forbidden[:4]}"
        )

    model_state = model.state_dict()
    expected_missing = sorted(name for name in model_state if name.startswith("readout."))
    if not expected_missing:
        raise InitializationContractError("Target model has no newly constructed readout")
    head_before = state_dict_sha256(
        {name: value for name, value in model_state.items() if name.startswith("readout.")}
    )
    incompatible = model.load_state_dict(source_state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise InitializationContractError(
            "Backbone-only load was not exact: "
            f"missing={missing}, expected_missing={expected_missing}, unexpected={unexpected}"
        )
    head_after = state_dict_sha256(
        {
            name: value
            for name, value in model.state_dict().items()
            if name.startswith("readout.")
        }
    )
    if head_before != head_after:
        raise InitializationContractError("Backbone load modified the target readout")

    stem = getattr(model, "stem", None)
    live_stem_sha = getattr(stem, "checkpoint_sha256", None)
    embedded_stem_sha = payload.get("stem_checkpoint_sha256")
    if live_stem_sha != expected_stem_sha256:
        raise InitializationContractError(
            f"Live stem SHA mismatch: expected {expected_stem_sha256}, got {live_stem_sha}"
        )
    if embedded_stem_sha is not None and embedded_stem_sha != expected_stem_sha256:
        raise InitializationContractError("Backbone checkpoint embeds a different stem identity")

    return {
        "mode": "strict_frozen_p11_backbone_only",
        "source_checkpoint": str(path),
        "source_checkpoint_sha256": actual_sha,
        "source_checkpoint_format": payload_format,
        "source_backbone_state_sha256": state_dict_sha256(source_state),
        "stem_checkpoint_sha256": live_stem_sha,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "new_readout_state_sha256": head_after,
        "copied_imagenet1k_readout": False,
    }


__all__ = [
    "InitializationContractError",
    "construct_large_vocabulary_model",
    "initialize_from_frozen_p11_backbone",
    "sha256_file",
    "state_dict_sha256",
]
