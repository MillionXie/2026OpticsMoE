"""Audit pinned offline T06 adaptation additions against main; copy no assets."""
import hashlib
import json
from pathlib import Path
import subprocess

from plan_source_import import plan

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "81fe403e896a4310c077a26e5c33ee8317752609"
TASK = "LightGenV2/tasks/t06_video_quality_assessment/"
NAMES = ["adapt_measured_readout.py", "test_adapt_measured_readout.py",
         "tune_measured_readout.py", "configs/spatial_hardware_readout_tuning.json",
         "configs/spatial_measured_readout_rank.json",
         "configs/spatial_measured_partial_test20.json",
         "configs/spatial_measured_partial_test20_weighted.json"]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    base = git("rev-parse", "main").decode().strip()
    additions = []
    for name in NAMES:
        path = TASK + name
        data = git("show", SOURCE + ":" + path)
        if git("ls-tree", base, "--", path):
            raise RuntimeError("Reviewed addition already exists: " + path)
        if path.endswith(".py"):
            compile(data, path, "exec")
        else:
            json.loads(data)
        additions.append({"path": path, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
    closure = plan(ROOT, SOURCE, [TASK + name for name in NAMES[:3]], False)
    if closure["missing_entries"] or closure["existing_conflicts"]:
        raise RuntimeError("Unreviewed local source difference")
    dependencies = []
    for path in closure["visited_paths"]:
        if path in {TASK + name for name in NAMES}:
            continue
        old = git("show", base + ":" + path)
        source = git("show", SOURCE + ":" + path)
        differs = old != source
        if differs and path != TASK + "lab_bench.py":
            raise RuntimeError("Unreviewed main dependency difference: " + path)
        dependencies.append({"path": path, "main_sha256": hashlib.sha256(old).hexdigest(),
                             "source_sha256": hashlib.sha256(source).hexdigest(),
                             "different": differs,
                             "decision": "retain main; offline extractor uses unchanged identity/verified_ccd APIs"
                             if differs else "exact source dependency already in main"})
    report = {"source_commit": SOURCE, "expected_main": base, "paths": additions,
              "dependencies": dependencies, "static_imports_only": True,
              "runtime_verified": False, "assets_changed": False,
              "historical_train_package_adapter_commit": "d89223b7d3b9dd3e1b76122fd26fb14cffb6a0a8",
              "historical_train_package_adapter_sha256": "5fa9b372000719eb62c0d381b00cb3de5c20e9a89e84e6a49ac9967626f05b99",
              "later_weighted_adapter_sha256": additions[0]["sha256"]}
    destination = ROOT / "maintenance/storage/T06_ADAPTATION_ADDITIONS_20261004.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"additions": len(additions), "dependencies": len(dependencies),
                      "main_dependencies_retained": sum(row["different"] for row in dependencies),
                      "source_files_copied": False}))


if __name__ == "__main__":
    main()
