"""Plan a pinned T11 source-only import; never replace working files."""
import hashlib
import json
from pathlib import Path
import subprocess

from audit_task_sources import audit

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "72180033c6edd605c7151b2faa8a599dc15910ee"
PREFIX = "LightGenV2/tasks/t11_lifelong_optics/"


def main():
    report = audit(ROOT, SOURCE, PREFIX)
    additions, replacements = [], []
    base = subprocess.check_output(["git", "rev-parse", "main"]).decode().strip()
    for row in report["files"]:
        name = row["path"]
        if Path(name).suffix not in {".py", ".json", ".md", ".txt"}:
            continue
        result = subprocess.run(["git", "show", f"{base}:{name}"], capture_output=True)
        if result.returncode:
            additions.append({"path": name, "sha256": row["source_sha256"], "bytes": row["bytes"]})
        elif hashlib.sha256(result.stdout).hexdigest() != row["source_sha256"]:
            if name != PREFIX + "README.md":
                raise RuntimeError("Unexpected existing main source conflict: " + name)
            replacements.append({"path": name, "source_sha256": row["source_sha256"],
                                 "expected_target_sha256": hashlib.sha256(result.stdout).hexdigest(),
                                 "review_reason": "Preserve original protocols and add actual server joint/sequential D2NN controls; model code remains separately identified"})
    receipts = {
        "T11_RUNTIME_LOCAL_AUDIT_20261004.json": report,
        "T11_PINNED_ADDITIONS_20261004.json": {"source_commit": SOURCE, "paths": additions},
        "T11_REVIEWED_ENTRY_20261004.json": {"source_commit": SOURCE, "expected_main": base,
                                           "task_prefix": PREFIX, "paths": replacements},
    }
    for name, value in receipts.items():
        dest = ROOT / "maintenance" / "storage" / name
        if dest.exists():
            raise FileExistsError("Protect earlier receipt: " + name)
        dest.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"additions": len(additions), "replacements": len(replacements),
                      "local_counts": report["counts"], "base_main": base}))


if __name__ == "__main__":
    main()
