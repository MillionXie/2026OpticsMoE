"""Bind the published offline adapter and retained dependencies to CPU evidence."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "5247f84b2b29a2b10616411fa7fb9af4c6a70148"


def main():
    audit = json.loads((ROOT / "maintenance/storage/T06_ADAPTATION_ADDITIONS_20261004.json").read_text(encoding="utf-8"))
    raw = json.loads((ROOT / ".codex_tmp/t11_source_20261004/t06_adaptation_candidate_cpu_contract.json").read_text(encoding="utf-8"))
    if raw["exit_code"]:
        raise RuntimeError("CPU tests failed")
    receipt = json.loads(next(line for line in raw["stdout"].splitlines() if line.startswith('{"commit":')))
    if (receipt["commit"] != COMMIT or receipt["backend_test_exit_code"] or
            receipt["adaptation_protocol_tests"] != {
                "tests_run": 12, "skipped": 0, "successful": True,
                "source_sha256": "3313f37da7de95a8a76058eb89ba549c556337443ff9208000a4fd4c79a67edd"}):
        raise RuntimeError("Wrong candidate or missing protocol verification")
    files = []
    for row in [*audit["paths"], *audit["dependencies"]]:
        data = subprocess.check_output(["git", "-C", str(ROOT), "show", COMMIT + ":" + row["path"]])
        digest = hashlib.sha256(data).hexdigest()
        expected = row.get("sha256", row.get("main_sha256"))
        if digest != expected:
            raise RuntimeError("Source identity changed: " + row["path"])
        files.append({"path": row["path"], "bytes": len(data), "source_blob_sha256": digest,
                      "origin": "pinned later adapter" if "sha256" in row else row["decision"]})
    report = {"source_head": audit["source_commit"], "publication_commit": COMMIT,
              "files": files, "cpu_evidence": receipt,
              "historical_train_package_adapter_commit": audit["historical_train_package_adapter_commit"],
              "historical_train_package_adapter_sha256": audit["historical_train_package_adapter_sha256"],
              "later_weighted_adapter_sha256": audit["later_weighted_adapter_sha256"],
              "scientific_source_changed": False, "assets_or_timings_changed": False,
              "hardware_acquisition_helpers_complete": False,
              "new_training_or_dataset_evaluation_performed": False}
    dest = ROOT / "LightGenV2/tasks/t06_video_quality_assessment/adaptation_source_import_20261004.json"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_identities": len(files), "cpu_protocol_tests": 12,
                      "assets_copied": False, "new_training": False}))


if __name__ == "__main__":
    main()
