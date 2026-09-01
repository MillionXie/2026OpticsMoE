"""Read-only metadata and tensor-integrity audit for a PyTorch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def tensor_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = payload.get("model", payload.get("backbone", {}))
    if not isinstance(candidate, Mapping):
        return {"tensor_count": 0, "all_finite": None, "all_nonempty": None}
    tensors = [value for value in candidate.values() if isinstance(value, torch.Tensor)]
    return {
        "tensor_count": len(tensors),
        "all_finite": all(bool(torch.isfinite(value).all()) for value in tensors),
        "all_nonempty": all(value.numel() > 0 for value in tensors),
        "parameter_elements": sum(value.numel() for value in tensors),
    }


def strict_p13_model_audit(
    payload: Mapping[str, Any], stem_checkpoint: Path
) -> dict[str, Any]:
    """Reconstruct the live P13 class and validate its serialized state."""

    from experiments.qwen3_vl_patch_stem_progressive_64stage_optical_imagenet_backbone.model import (
        QwenStemProgressiveOpticalImageNetBackbone,
    )

    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        raise TypeError("P13 strict audit requires a mapping model_config")
    model = QwenStemProgressiveOpticalImageNetBackbone(
        stem_checkpoint, dict(model_config)
    )
    expected_stem_sha256 = payload.get("stem_checkpoint_sha256")
    if expected_stem_sha256 != model.stem.checkpoint_sha256:
        raise RuntimeError(
            "Live stem checkpoint SHA-256 differs from checkpoint metadata"
        )

    if isinstance(payload.get("model"), Mapping):
        state_scope = "full_training_model"
        incompatible = model.load_state_dict(payload["model"], strict=True)
        expected_missing: list[str] = []
    elif isinstance(payload.get("backbone"), Mapping):
        state_scope = "backbone_without_temporary_readout"
        incompatible = model.load_state_dict(payload["backbone"], strict=False)
        expected_missing = sorted(
            key for key in model.state_dict() if key.startswith("readout.")
        )
    else:
        raise TypeError("P13 checkpoint has neither model nor backbone state")

    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if missing != expected_missing or unexpected:
        raise RuntimeError(
            "P13 state is incompatible with the live model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    saved_depth = payload.get("depth_alpha")
    live_depth = dict(model.depth_alpha_report())
    if not isinstance(saved_depth, Mapping) or dict(saved_depth) != live_depth:
        raise RuntimeError("Loaded P13 depth-alpha state differs from metadata")
    return {
        "state_scope": state_scope,
        "strict_compatible": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "num_stages": model.num_stages,
        "stem_checkpoint_sha256": model.stem.checkpoint_sha256,
        "depth_alpha_matches": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--p13-stem",
        type=Path,
        help="reconstruct the P13 model with this static Qwen stem and load its state",
    )
    args = parser.parse_args()
    path = args.checkpoint.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint root is not a mapping: {type(payload)!r}")
    report = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "format": payload.get("format"),
        "checkpoint_role": payload.get("checkpoint_role"),
        "epoch": payload.get("epoch", payload.get("best_epoch")),
        "config_digest": payload.get("config_digest"),
        "num_stages": nested(payload, "model_config", "num_stages")
        or nested(payload, "model_report", "num_stages"),
        "implementation_aggregate_sha256": nested(
            payload, "implementation_manifest", "aggregate_sha256"
        ),
        "feedback_method": nested(payload, "feedback", "method"),
        "depth_alpha": payload.get("depth_alpha"),
        "tensor_audit": tensor_audit(payload),
    }
    if args.p13_stem is not None:
        report["strict_model_load"] = strict_p13_model_audit(
            payload, args.p13_stem.expanduser().resolve()
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
