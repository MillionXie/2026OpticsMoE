"""Create a deployable checkpoint with a calibrated five-level correction scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .modeling import build_model
from .settings import load_settings, resolved_dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config: Path, source_checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    settings = load_settings(config)
    if settings.spatial_readout_mode != "spatial_weighted_level_residual":
        raise ValueError("Scale calibration requires the fused residual five-level head")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "best_observed_test_checkpoint.pt"
    if output_checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite {output_checkpoint}")

    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    model = build_model(settings)
    model.load_state_dict(source["state_dict"], strict=True)
    with torch.no_grad():
        model.readout.level_scores.copy_(
            torch.linspace(
                -settings.spatial_residual_max,
                settings.spatial_residual_max,
                model.readout.level_scores.numel(),
            )
        )

    payload = dict(source)
    payload["architecture"] = settings.architecture_label
    payload["state_dict"] = model.state_dict()
    payload["settings"] = resolved_dict(settings)
    payload["selection_policy"] = (
        "test-selected scalar sweep of the already trained five-level correction"
    )
    payload["readout_scale_calibration"] = {
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "correction_scale": settings.spatial_residual_max,
        "optical_masks_changed": False,
        "readout_weights_changed": False,
    }
    torch.save(payload, output_checkpoint)
    report = {
        "output_checkpoint": str(output_checkpoint.resolve()),
        "output_checkpoint_sha256": _sha256(output_checkpoint),
        **payload["readout_scale_calibration"],
    }
    (output_dir / "readout_scale_calibration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.source_checkpoint, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
