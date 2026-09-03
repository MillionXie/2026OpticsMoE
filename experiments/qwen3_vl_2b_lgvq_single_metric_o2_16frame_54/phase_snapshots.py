"""Small, auditable phase-only checkpoints for mask-evolution analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .settings import ExperimentSettings, resolved_dict


CONTRACT = "optical_phase_evolution_snapshot_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_phase_snapshot(
    model: torch.nn.Module,
    settings: ExperimentSettings,
    *,
    epoch: int,
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    """Save raw and physical phase tensors without electronic/optimizer state."""

    planes: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        if "raw_" not in name or "phase" not in name:
            continue
        raw = parameter.detach().cpu().float().contiguous()
        planes[name] = {
            "shape": list(raw.shape),
            "raw_parameter": raw,
            "phase_rad": (2.0 * math.pi * torch.sigmoid(raw)).contiguous(),
        }
    if not planes:
        raise RuntimeError("No trainable raw phase parameters were found")
    fusion_alpha = {
        name: float(layer.alpha.detach().cpu())
        for name, layer in zip(
            ("vision_expert", "vision_global", "language_expert", "language_global"),
            model.fusions,
        )
    }
    payload = {
        "schema_version": 1,
        "contract": CONTRACT,
        "epoch": int(epoch),
        "architecture": settings.architecture_label,
        "target_name": settings.target_name,
        "prompt": settings.prompt,
        "phase_parameterization": "phase_rad = 2*pi*sigmoid(raw_parameter)",
        "unmodulated_leakage": {
            "interpretation": "nominal coherent unmodulated optical-power fraction",
            "train_min": settings.unmodulated_power_fraction_min,
            "train_max": settings.unmodulated_power_fraction_max,
            "evaluation": settings.unmodulated_power_fraction_eval,
        },
        "fusion_alpha": fusion_alpha,
        "metrics_optical_on": dict(metrics or {}),
        "settings": resolved_dict(settings),
        "planes": planes,
    }
    directory = settings.output_dir / "phase_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch_{int(epoch):04d}.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

    manifest_path = directory / "manifest.json"
    records: list[dict[str, Any]] = []
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = list(previous.get("snapshots", []))
    record = {
        "epoch": int(epoch),
        "file": path.name,
        "sha256": _sha256(path),
        "plane_count": len(planes),
        "fusion_alpha": fusion_alpha,
        "test_srcc": None if metrics is None else metrics.get("srcc"),
    }
    records = [item for item in records if int(item["epoch"]) != int(epoch)]
    records.append(record)
    records.sort(key=lambda item: int(item["epoch"]))
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "contract": CONTRACT,
            "target_name": settings.target_name,
            "snapshot_interval_epochs": settings.phase_snapshot_interval_epochs,
            "snapshots": records,
        },
    )
    return path


def load_phase_snapshot(path: str | Path) -> dict[str, Any]:
    """Strict reader used by downstream analysis; raises on malformed files."""

    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract") != CONTRACT:
        raise ValueError(f"Not a {CONTRACT} file: {path}")
    if int(payload.get("epoch", 0)) <= 0 or not isinstance(payload.get("planes"), dict):
        raise ValueError(f"Malformed phase snapshot: {path}")
    for name, plane in payload["planes"].items():
        raw, phase = plane.get("raw_parameter"), plane.get("phase_rad")
        if not torch.is_tensor(raw) or not torch.is_tensor(phase):
            raise ValueError(f"Plane {name!r} is missing tensor data")
        if tuple(raw.shape) != tuple(phase.shape) or list(raw.shape) != plane.get("shape"):
            raise ValueError(f"Plane {name!r} shape metadata disagrees with tensors")
        expected = 2.0 * math.pi * torch.sigmoid(raw.float())
        if not torch.allclose(expected, phase.float(), atol=1.0e-6, rtol=1.0e-6):
            raise ValueError(f"Plane {name!r} phase does not match 2*pi*sigmoid(raw)")
    return payload


def summarize_directory(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory).expanduser().resolve()
    paths = sorted(directory.glob("epoch_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No epoch_*.pt snapshots under {directory}")
    loaded = [load_phase_snapshot(path) for path in paths]
    first = loaded[0]
    rows: list[dict[str, Any]] = []
    for path, payload in zip(paths, loaded):
        if payload["architecture"] != first["architecture"] or payload["target_name"] != first["target_name"]:
            raise RuntimeError("Snapshot directory mixes architectures or targets")
        for name, plane in payload["planes"].items():
            phase = plane["phase_rad"].float()
            baseline = first["planes"][name]["phase_rad"].float()
            delta = torch.atan2(torch.sin(phase - baseline), torch.cos(phase - baseline))
            rows.append(
                {
                    "epoch": int(payload["epoch"]),
                    "plane": name,
                    "parameters": int(phase.numel()),
                    "phase_rad_mean": float(phase.mean()),
                    "phase_rad_std": float(phase.std(unbiased=False)),
                    "wrapped_delta_from_first_rms_rad": float(delta.square().mean().sqrt()),
                    "fraction_changed_over_0p05_rad": float((delta.abs() > 0.05).float().mean()),
                }
            )
    csv_path = directory / "phase_evolution_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "contract": CONTRACT,
        "target_name": first["target_name"],
        "architecture": first["architecture"],
        "snapshot_count": len(loaded),
        "epochs": [int(value["epoch"]) for value in loaded],
        "summary_csv": str(csv_path),
    }
    _write_json(directory / "phase_evolution_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir")
    args = parser.parse_args()
    print(json.dumps(summarize_directory(args.snapshot_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONTRACT", "load_phase_snapshot", "save_phase_snapshot", "summarize_directory"]
