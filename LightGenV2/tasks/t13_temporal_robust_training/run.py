"""Independent four-group Temporal robustness entry (no automatic queue)."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from study import GROUPS, asset_preflight, load_protocol, make_config, sha256, split_for_selection, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=GROUPS)
    parser.add_argument("--phase", required=True, choices=("plan", "preflight", "smoke", "train", "evaluate", "theory"))
    parser.add_argument("--seed", type=int, default=163)
    parser.add_argument("--paths", type=Path, help="Private JSON data paths; never committed")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-pitch-um", type=float)
    parser.add_argument("--eval-eta", type=float)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=20260927)
    parser.add_argument("--allow-uncalibrated-noise", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / ("smoke" if args.phase == "smoke" else "simulation") / f"{args.group}_s{args.seed}_{args.phase}"
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new/empty; do not overwrite a run")
    purpose = "nominal" if args.phase == "theory" else ("deployment" if args.phase == "evaluate" else "train")
    if args.phase == "theory" and (args.group != "r0_post" or args.noise_scale != 0 or args.eval_eta is not None or args.device_pitch_um is not None):
        parser.error("G1 theory reuses r0_post with ideal 17um grid, zero leakage and zero noise")
    if args.phase == "train" and (args.device_pitch_um is not None or args.eval_eta is not None):
        parser.error("Training uses the frozen study protocol; pitch/eta flags are evaluation-only")
    paths = json.loads(args.paths.read_text(encoding="utf-8")) if args.paths else {}
    raw = make_config(args.group, purpose=purpose, seed=args.seed, output=output, paths=paths,
                      device_pitch=args.device_pitch_um, eval_eta=args.eval_eta)
    raw["device"] = args.device
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    protocol = load_protocol()
    write_json(output / "run_manifest.json", {
        "group": args.group, "condition": GROUPS[args.group], "phase": args.phase,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "command": sys.argv, "protocol": protocol, "resolved_config_sha256": sha256(config_path),
        "noise_scale": args.noise_scale, "noise_seed": args.noise_seed,
        "selection": "TRAIN-derived validation on common deployment grid/eta; original test excluded",
    })
    if args.phase == "plan":
        print(json.dumps({"group": args.group, "condition": GROUPS[args.group], "config": str(config_path)}, indent=2))
        return 0
    if args.phase == "preflight":
        report = asset_preflight(raw)
        report["ccd_parameters_calibrated"] = protocol["ccd_parameters_calibrated"]
        write_json(output / "preflight.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2 if report["missing"] else 0
    if args.phase == "train":
        assets = asset_preflight(raw)
        write_json(output / "preflight.json", assets)
        if assets["missing"]:
            parser.error("Training assets missing; restore or rebuild caches first")
        if GROUPS[args.group]["ccd"] and not protocol["ccd_parameters_calibrated"] and not args.allow_uncalibrated_noise:
            parser.error("CCD profile is empirical, not calibrated; explicitly authorize pilot with --allow-uncalibrated-noise")
    sys.path.insert(0, str(ROOT / "runtime"))
    import torch
    from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import load_settings, resolved_dict
    from LightGenV2.tasks.t06_video_quality_assessment.models.multivideo9x4 import build_model
    from LightGenV2.tasks.t06_video_quality_assessment import multivideo_training as training
    from experiment import detector_noise_evaluation, install_common_selection_evaluator
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    settings = load_settings(config_path, synthetic=args.phase == "smoke")
    if args.phase == "smoke":
        from LightGenV2.tasks.t06_video_quality_assessment.multivideo import synthetic_smoke
        result = synthetic_smoke(settings)
        write_json(output / "smoke.json", result)
        print(json.dumps(result, indent=2))
        return 0
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import load_single_metric_cache
    payload = load_single_metric_cache(settings)
    model = build_model(settings)
    device = torch.device(args.device)
    if args.phase == "train":
        payload, split = split_for_selection(payload, fraction=protocol["validation_fraction"], seed=protocol["validation_seed"])
        write_json(output / "selection_split.json", split)
        deploy_raw = make_config(args.group, purpose="deployment", seed=args.seed, output=output, paths=paths)
        deploy_path = output / "deployment.yaml"
        deploy_path.write_text(yaml.safe_dump(deploy_raw, sort_keys=False), encoding="utf-8")
        deploy_settings = load_settings(deploy_path)
        previous = install_common_selection_evaluator(training, deploy_settings, seed=protocol["validation_seed"])
        try:
            result = training.train(model, payload, settings, device)
        finally:
            training.evaluate = previous
        # Legacy names are retained; correct their meaning explicitly outside the frozen backend.
        result["legacy_test_fields_refer_to_validation"] = True
        result["original_test_used_for_selection"] = False
        result["test_used_for_selection"] = False
        result["validation_used"] = True
        result["best_validation_srcc"] = result.pop("best_observed_test_srcc")
        write_json(output / "training_summary.json", result)
        write_json(output / "study_training_summary.json", result)
        for name in ("best_checkpoint.pt", "last_checkpoint.pt"):
            path = output / name
            saved = torch.load(path, map_location="cpu", weights_only=False)
            saved.update(selection_policy="TRAIN-derived validation on common deployment physics",
                         test_used_for_selection=False, validation_used=True,
                         study_group=args.group, study_protocol=protocol)
            torch.save(saved, path)
        result["checkpoint_sha256"] = sha256(output / "best_checkpoint.pt")
        write_json(output / "training_summary.json", result)
        write_json(output / "study_training_summary.json", result)
    else:
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if saved.get("study_group") != args.group:
            parser.error("Checkpoint group identity does not match --group")
        model.load_state_dict(saved["state_dict"], strict=True)
        model.to(device).eval()
        torch.manual_seed(args.noise_seed)
        with detector_noise_evaluation(settings, scale=args.noise_scale):
            result = training.evaluate(model, payload, settings, device, optical_enabled=True,
                                       prediction_path=output / "predictions.csv")
        write_json(output / "evaluation.json", {"metrics": result, "checkpoint_sha256": sha256(args.checkpoint),
                   "settings": resolved_dict(settings), "noise_scale": args.noise_scale,
                   "group": args.group, "device_pitch_um": settings.modulator_pixel_pitch_um})
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
