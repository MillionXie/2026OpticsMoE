"""Record the existing server T09 runtime and protected asset identities."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEAD = "1c7222a1119385475e1b464ca7efdb67cdc21a4c"


def main():
    source = ROOT / ".codex_tmp/t11_source_20261004/t09_runtime_identity.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw["exit_code"]:
        raise RuntimeError("Server identity audit failed")
    report = json.loads(raw["stdout"])
    if report["head"] != HEAD or not report["read_only"] or report["assets_moved"] or report["source_changed"]:
        raise RuntimeError("Unexpected audit scope")
    for key in ("source_files", "protected_asset_records"):
        paths = set()
        for row in report[key]:
            if (row["path"] in paths or not row["path"].startswith("LightGenV2/tasks/t09_multimodal_matching/")
                    or ".." in Path(row["path"]).parts or len(row["sha256"]) != 64):
                raise RuntimeError("Invalid source/asset identity")
            paths.add(row["path"])
    report["source_overlay_complete"] = False
    report["closure_runtime_verified"] = False
    report["note"] = "Shared demo_check backend is dirty and not included in task-only source inventory; retain original runtime until audited. Process cwd list is an observation, not ownership or safe-deletion proof."
    dest = ROOT / "maintenance/storage/T09_RUNTIME_IDENTITY_20261004.json"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source_records": len(report["source_files"]),
                      "protected_asset_records": len(report["protected_asset_records"]),
                      "task_overlay_files": [r["path"] for r in report["source_files"] if r["different_from_commit"]],
                      "runtime_migration_complete": False, "assets_changed": False}))


if __name__ == "__main__":
    main()
