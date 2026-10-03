"""Compare a pinned task Git tree with current files without overwriting anything."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], input=stdin)


def audit(root: Path, ref: str, prefix: str) -> dict:
    root = root.resolve()
    target = (root / prefix).resolve()
    parts = PurePosixPath(prefix.replace("\\", "/")).parts
    if (len(parts) < 3 or parts[:2] != ("LightGenV2", "tasks")
            or ".." in parts or not target.is_relative_to(root / "LightGenV2" / "tasks")):
        raise ValueError("Audit must stay within a LightGenV2 task")
    head = git(root, "rev-parse", "--verify", ref).decode().strip()
    records = git(root, "ls-tree", "-r", "-z", head, "--", prefix).split(b"\0")
    entries = []
    for record in filter(None, records):
        metadata, name = record.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        if kind != b"blob" or mode != b"100644":
            raise ValueError("Special tree entry requires manual review")
        entries.append((name.decode("utf-8"), oid))
    payload = git(root, "cat-file", "--batch", stdin=b"".join(oid + b"\n" for _, oid in entries))
    offset, rows = 0, []
    for relative, _ in entries:
        end = payload.index(b"\n", offset)
        header = payload[offset:end].split()
        size = int(header[-1])
        data = payload[end + 1:end + 1 + size]
        offset = end + size + 2
        path = root / relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            state, local_sha = "unsafe_path", None
        elif not path.is_file():
            state, local_sha = "missing", None
        else:
            local = path.read_bytes()
            local_sha = hashlib.sha256(local).hexdigest()
            if local == data:
                state = "byte_identical"
            elif local.replace(b"\r\n", b"\n") == data.replace(b"\r\n", b"\n"):
                state = "line_endings_only"
            else:
                state = "different_preserve_local"
        rows.append({"path": relative, "source_sha256": hashlib.sha256(data).hexdigest(),
                     "local_sha256": local_sha, "state": state, "bytes": size})
    counts = {state: sum(row["state"] == state for row in rows) for state in
              ("byte_identical", "line_endings_only", "missing", "different_preserve_local", "unsafe_path")}
    return {"source_head": head, "task_prefix": prefix, "counts": counts, "files": rows,
            "read_only": True, "deletion_authorized": False,
            "limitations": "Pinned commit only; server working overlay and local-only files require separate audit. No runtime/model equivalence claim."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    report = audit(Path(__file__).resolve().parents[2], args.ref, args.task)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
