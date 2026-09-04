"""Fit fixed train-only statistics for the four optical router CCD regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .data import load_single_metric_cache
from .modeling import build_model
from .settings import load_settings
from .training import _loader, evaluate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibrate(
    config: Path,
    checkpoint: Path,
    output_checkpoint: Path,
    report_path: Path,
    device_name: str,
) -> dict[str, Any]:
    settings = load_settings(config)
    payload = load_single_metric_cache(settings)
    resolved_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else device_name
    )
    device = torch.device(resolved_device)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("state_dict", raw.get("model", raw))
    model = build_model(settings)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    standardizer = model.serial_router.channel_standardizer
    if standardizer is None:
        raise RuntimeError("router.serial_channel_standardization must be true")

    count = 0
    total = torch.zeros(4, dtype=torch.float64)
    total_square = torch.zeros(4, dtype=torch.float64)

    def collect(_module, arguments) -> None:
        nonlocal count, total, total_square
        values = arguments[0].detach().double()
        count += int(values.shape[0])
        total += values.sum(0).cpu()
        total_square += values.square().sum(0).cpu()

    hook = standardizer.register_forward_pre_hook(collect)
    # Only this non-affine BatchNorm is put in training mode. The rest of the
    # model remains deterministic and no parameter is updated.
    standardizer.train()
    with torch.no_grad():
        for batch in _loader(payload, "train", settings, shuffle=False):
            model(
                batch["vision_tokens"].to(device),
                batch["quality_tokens"].to(device),
                batch["language_tokens"].to(device),
                batch["language_mask"].to(device),
                None
                if "raw_frames" not in batch
                else batch["raw_frames"].to(device),
                vgg_tokens=None
                if "vgg_tokens" not in batch
                else batch["vgg_tokens"].to(device),
                optical_enabled=True,
            )
    hook.remove()
    if count <= 0:
        raise RuntimeError("No training samples were observed")
    mean = total / count
    variance = (total_square / count - mean.square()).clamp_min(1.0e-8)
    standardizer.running_mean.copy_(mean.float().to(device))
    standardizer.running_var.copy_(variance.float().to(device))
    standardizer.num_batches_tracked.fill_(count)
    standardizer.eval()

    test_loader = _loader(payload, "test", settings, shuffle=False)
    optical_on = evaluate(model, test_loader, device, optical_enabled=True)
    optical_off = evaluate(model, test_loader, device, optical_enabled=False)
    calibration = {
        "schema_version": 1,
        "method": "fixed non-affine z-score of four log CCD region energies",
        "fit_split": "train",
        "test_labels_used": False,
        "sample_count": count,
        "mean_log_energy": mean.tolist(),
        "variance_log_energy": variance.tolist(),
        "normal_optical_electronic": optical_on,
        "same_checkpoint_optics_bypassed": optical_off,
    }
    raw["state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    raw["metrics_optical_on"] = optical_on
    raw["router_channel_calibration"] = calibration
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, output_checkpoint)
    report = {
        **calibration,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "output_checkpoint": str(output_checkpoint.resolve()),
        "output_checkpoint_sha256": _sha256(output_checkpoint),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = calibrate(
        args.config,
        args.checkpoint,
        args.output_checkpoint,
        args.report,
        args.device,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
