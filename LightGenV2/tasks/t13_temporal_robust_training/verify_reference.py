"""Check copied runtime against one frozen teacher field, not a full-test claim."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from study import sha256, write_json
from verify_source import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Original unpacked teacher final_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output file must be new")
    verify()
    pin = json.loads((ROOT / "reference" / "source_manifest.json").read_text(encoding="utf-8"))
    release_path = args.package / "release.json"
    if sha256(release_path) != pin["source_release_json_sha256"]:
        parser.error("Not the pinned teacher final_v2 release")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    field = release["fields"][0]
    input_path = args.package / field["file"]
    if sha256(input_path) != field["sha256"]:
        parser.error("Frozen field input identity mismatch")
    sys.path.insert(0, str(ROOT / "runtime"))
    import torch
    from LightGenV2.tasks.t06_video_quality_assessment.lab_runtime import load_model, forward
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, settings = load_model("temporal", args.package / "weights" / "best_checkpoint.pt", args.device)
    batch = torch.load(input_path, map_location="cpu", weights_only=False)
    with torch.inference_mode():
        prediction = forward(model, batch)["prediction"].reshape(-1).cpu()
    golden = json.loads((ROOT / "reference" / "golden_first_field.json").read_text(encoding="utf-8"))
    # release.json still carries historical per-field predictions; use the
    # independently verified CURRENT v2 rows, consistent with v2 full metrics.
    rows = sorted(golden["rows"], key=lambda row: row["slot"])
    if golden["field"] != field["key"] or [row["video"] for row in rows] != field["sample_ids"]:
        parser.error("Golden field identities differ from packaged inputs")
    reference = torch.tensor([row["prediction"] for row in rows])
    delta = float((prediction-reference).abs().max())
    # Match the teacher's already published per-prediction MOS gate (0.01),
    # not its much smaller metric gates; CPU/new-torch is not bit-identical CUDA.
    tolerance = 0.01
    report = {"status": "passed" if delta <= tolerance else "failed", "scope": "one frozen field / smoke only",
              "max_abs_prediction_delta": delta, "tolerance": tolerance, "torch": torch.__version__,
              "device": args.device, "modulator_pixel_pitch_um": settings.modulator_pixel_pitch_um,
              "checkpoint_sha256": pin["checkpoint_sha256"], "reference_full_srcc": pin["reference_metrics"]["srcc"],
              "golden_report_sha256": golden["sha256"]}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
