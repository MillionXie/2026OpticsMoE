"""Read-only metadata and tensor-integrity audit for a PyTorch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
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
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
