"""Compare the actual teacher ZIP against its unpacked package byte-for-byte."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from study import sha256, write_json
from build_projects import verify_teacher


def audit(archive, package):
    import yaml
    pin = verify_teacher(package)
    raw = yaml.safe_load((package / "configs/temporal_16x4_s163.yaml").read_text(encoding="utf-8"))
    hashes = json.loads((package / "SHA256.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive) as zipped:
        members = {name for name in zipped.namelist() if not name.endswith("/")}
        weight_matches = [name for name in members if name.endswith("weights/best_checkpoint.pt")]
        if len(weight_matches) != 1:
            raise ValueError("ZIP must contain exactly one pinned inference checkpoint")
        prefix = weight_matches[0][:-len("weights/best_checkpoint.pt")]
        for relative, expected in hashes.items():
            with zipped.open(prefix + relative) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected:
                raise ValueError(f"ZIP differs from unpacked source: {relative}")
        directories = sorted({name[len(prefix):].split("/")[0] for name in members if name.startswith(prefix)})
    return {"status": "passed", "zip": str(archive.resolve()), "zip_sha256": sha256(archive),
            "verified_manifest_files": len(hashes), "root_entries": directories,
            "checkpoint_bytes": (package / "weights/best_checkpoint.pt").stat().st_size,
            "checkpoint_sha256": pin["checkpoint_sha256"],
            "training_config": {"ccd_enabled": raw["robustness"]["ccd_noise"]["enabled"],
                                "coherent_dc_range": [raw["optics"]["unmodulated_power_fraction_min"], raw["optics"]["unmodulated_power_fraction_max"]],
                                "eval_dc": raw["optics"]["unmodulated_power_fraction_eval"],
                                "logical_pitch_um": raw["optics"]["pixel_pitch_um"],
                                "forward_device_pitch_um": raw["optics"]["modulator_pixel_pitch_um"]},
            "not_a_clean_ablation_baseline": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new audit output")
    result = audit(args.zip, args.package)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
