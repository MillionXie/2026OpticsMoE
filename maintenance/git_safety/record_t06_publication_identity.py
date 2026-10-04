"""Record verified T06 publication evidence, without copying scientific assets."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "8d6445e28c88346c692acfaef788cd9d4a4e86fe"


def main():
    raw = json.loads((ROOT / ".codex_tmp/t11_source_20261004/t06_candidate_cpu_contract.json").read_text(encoding="utf-8"))
    if raw["exit_code"]:
        raise RuntimeError("Candidate CPU verification failed")
    receipt = json.loads(next(line for line in raw["stdout"].splitlines() if line.startswith('{"commit":')))
    if receipt["commit"] != COMMIT or receipt["backend_test_exit_code"] or len(receipt["strict_reloads"]) != 2:
        raise RuntimeError("Unexpected runtime verification identity")
    closure = json.loads((ROOT / "maintenance/storage/T06_FORMAL_RUNTIME_CLOSURE_20261004.json").read_text(encoding="utf-8"))
    files, retained = [], []
    for row in closure["files"]:
        data = subprocess.check_output(["git", "-C", str(ROOT), "show", COMMIT + ":" + row["path"]])
        digest = hashlib.sha256(data).hexdigest()
        value = {"path": row["path"], "bytes": len(data), "source_blob_sha256": digest}
        if row["path"].endswith("/lab_runtime.py"):
            if digest != row["main_sha256"]:
                raise RuntimeError("Later lab-runtime fixes were not preserved")
            value["origin"] = "retained preceding main atomic JSON read/write fixes"
            retained.append(value)
        else:
            if digest != row["source_sha256"]:
                raise RuntimeError("Pinned closure differs: " + row["path"])
            files.append(value)
    source = {"source_head": closure["source_commit"], "publication_commit": COMMIT,
              "files": files, "additional_dependency_hashes": retained,
              "scientific_source_changed": False,
              "cpu_evidence": receipt, "legacy_default_profile_migration_complete": False,
              "later_adaptation_source_migration_complete": False,
              "all_timings_and_assets_preserved": True, "original_working_directories_changed": False}
    destinations = {
        "LightGenV2/tasks/t06_video_quality_assessment/source_import_20261004.json": source,
        "maintenance/storage/T06_PUBLISHED_CPU_CONTRACT_20261004.json": receipt,
    }
    for path, value in destinations.items():
        destination = ROOT / path
        if destination.exists():
            raise FileExistsError("Protect existing receipt: " + path)
        destination.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_files": len(files), "retained_dependency_files": len(retained),
                      "strict_loads": len(receipt["strict_reloads"]), "assets_copied": False}))


if __name__ == "__main__":
    main()
