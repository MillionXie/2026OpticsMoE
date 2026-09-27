"""Infer a NEW group checkpoint using frozen teacher fields, no training caches."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime"))
from study import make_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/best_checkpoint.pt")
    parser.add_argument("--package", type=Path, default=ROOT / "teacher_reference")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fields", type=int, default=1)
    parser.add_argument("--noise-scale", type=float, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.fields < 0:
        parser.error("Use new output; fields must be nonnegative (0=all)")
    if not args.checkpoint.is_file():
        parser.error("This group's trained PT is missing; use project.py reference for old teacher inference")
    import torch
    import yaml
    from ccd import camera_operator
    from LightGenV2.tasks.t06_video_quality_assessment.models.multivideo9x4 import build_model
    from LightGenV2.tasks.t06_video_quality_assessment.multivideo_settings import load_settings
    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import forward
    from lgvq_temporal.fixed_weight import regression_metrics
    identity = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    group = identity["group"]
    if saved.get("study_group") != group:
        parser.error("Wrong group checkpoint; old teacher weights are reference-only")
    protocol = saved["study_protocol"]
    raw = make_config(group, purpose="deployment", device_pitch=protocol["device_pitch_um"],
                      eval_eta=protocol["deployment_eta"])
    with tempfile.TemporaryDirectory(prefix="t13_infer_") as temp:
        config = Path(temp) / "deployment.yaml"
        config.write_text(yaml.safe_dump(raw), encoding="utf-8")
        settings = load_settings(config)
    model = build_model(settings).to(args.device).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    release = json.loads((args.package / "release.json").read_text(encoding="utf-8"))
    fields = release["fields"][:args.fields] if args.fields else release["fields"]
    torch.manual_seed(20260927)
    rows, predictions, targets = [], [], []
    with torch.inference_mode(), camera_operator(protocol["ccd_model"], evaluation_scale=args.noise_scale):
        for item in fields:
            path = args.package / item["file"]
            if sha256(path) != item["sha256"]:
                raise ValueError("Frozen field input hash mismatch")
            scores = forward(model, torch.load(path, map_location="cpu", weights_only=False))["prediction"].cpu().flatten()
            for slot, valid in enumerate(item["valid"]):
                if valid:
                    prediction, target = float(scores[slot]), float(item["targets"][slot])
                    predictions.append(prediction)
                    targets.append(target)
                    rows.append({"field": item["key"], "slot": slot, "video": item["sample_ids"][slot],
                                 "prediction": prediction, "target": target})
    metrics = regression_metrics(torch.tensor(predictions), torch.tensor(targets), "temporal")
    write_json(args.output, {"group": group, "scope": "deployment_simulation_not_hardware",
               "checkpoint_sha256": sha256(args.checkpoint), "metrics": metrics,
               "field_count": len(fields), "video_count": len(rows), "rows": rows})
    print(args.output)


if __name__ == "__main__":
    main()
