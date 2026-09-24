"""Evaluate one saved T04 checkpoint and its same-weight no-optical ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing import training as legacy
from .modeling import build_model
from .run import PROFILES, TASK_DIR
from .settings import load_settings
from .training import _set_phase_dropout, build_loaders, evaluate_with_routes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    settings = load_settings(TASK_DIR / "configs" / PROFILES[args.profile])
    settings.num_workers = 0
    checkpoint = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh audit directory: {output}")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    _, test_loader = build_loaders(settings)
    model = build_model(settings, device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != model.checkpoint_architecture:
        raise ValueError("Checkpoint and profile architecture differ")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    _set_phase_dropout(model, False)
    with torch.inference_mode():
        normal, normal_records, _ = evaluate_with_routes(model, test_loader, settings, device)
        alphas = {
            name: [float(core.block1_optical_fusion), float(core.block2_optical_fusion)]
            for name, core in (("language", model.language_core), ("vision", model.vision_core))
        }
        for core in (model.language_core, model.vision_core):
            core.set_fusion_ablation("remove_optical")
        no_optical, no_optical_records, _ = legacy._evaluate(model, test_loader, settings, device)
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "epoch": int(payload["epoch"]),
        "profile": args.profile,
        "training_zero_order_intensity_fraction_per_slm": settings.zero_order_intensity_fraction,
        "training_ccd_noise_mean_fraction": settings.ccd_noise_mean_fraction,
        "training_ccd_noise_std_fraction": settings.ccd_noise_std_fraction,
        "evaluation_perturbations": "disabled by model.eval()",
        "fusion_alpha": alphas,
        "normal": normal,
        "remove_optical": no_optical,
        "remove_optical_definition": "same weights; both optical paths disabled; electronic coefficient restored to one; no retraining",
        "changed_accuracy_drop_percentage_points": 100 * (
            normal["overall"]["changed_cell_accuracy"]
            - no_optical["overall"]["changed_cell_accuracy"]
        ),
    }
    _write_json(output / "audit.json", report)
    for name, records in (("normal_predictions.jsonl", normal_records),
                          ("remove_optical_predictions.jsonl", no_optical_records)):
        with (output / name).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "epoch": report["epoch"],
                      "normal_changed": normal["overall"]["changed_cell_accuracy"],
                      "no_optical_changed": no_optical["overall"]["changed_cell_accuracy"],
                      "drop_pp": report["changed_accuracy_drop_percentage_points"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
