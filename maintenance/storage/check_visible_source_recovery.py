"""Read-only revalidation of visible sources against a supplied historical index.

This is not a deletion gate: recovery does not prove absence of runtime dependencies.
The index may be stale. Unknown or changed files are explicitly retained as exceptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def inspect(root: Path, index: dict) -> dict:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args])

    visible = [p for p in git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0") if p]
    by_path = {r["path"]: r for r in index["files"]}
    rows = []
    for name in visible:
        old = by_path.get(name)
        if old is None:
            continue  # Index covers sources, not all generated artifacts.
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            rows.append({"path": name, "status": "outside_root_retained"})
            continue
        if path.is_symlink() or not path.is_file():
            rows.append({"path": name, "status": "nonregular_retained"})
            continue
        payload = path.read_bytes()
        sha = digest(payload)
        row = {"path": name, "sha256": sha, "status": "unverified_retained"}
        if sha != old["sha256"]:
            row["status"] = "changed_since_index_retained"
        else:
            for recovery in old.get("archive_matches", []):
                commit, target = recovery["commit"], recovery["path"]
                proc = subprocess.run(["git", "-C", str(root), "show", commit + ":" + target], capture_output=True)
                if proc.returncode:
                    continue
                if proc.stdout == payload:
                    kind = "exact_blob"
                elif proc.stdout.replace(b"\r\n", b"\n") == payload.replace(b"\r\n", b"\n"):
                    kind = "eol_only_blob"
                else:
                    continue
                row.update(status=kind, recovery_commit=commit, recovery_path=target)
                break
        rows.append(row)
    return {
        "head": git("rev-parse", "HEAD").decode().strip(),
        "index_head": index.get("head"),
        "indexed_visible_sources": len(rows),
        "unindexed_visible_files": sum(name not in by_path for name in visible),
        "counts": dict(Counter(r["status"] for r in rows)),
        "files": rows,
        "mutations": [],
        "limits": "Local indexed sources only; no claim about Linux bytes, new unindexed files, live occupation, dependencies or deletion safety.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    result = inspect(args.root.resolve(), json.loads(args.index.read_text(encoding="utf-8")))
    if args.summary:
        result.pop("files")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
