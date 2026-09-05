"""Build the authoritative index of Git-tracked project source trees.

Physical runtime directories are deliberately excluded. Server-only runs are
indexed separately by ``build_run_index.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from collections import defaultdict
from pathlib import Path


SOURCE_ROOTS = {
    "experiments": "general_experiment",
    "FixedFeedbackSFT/projects": "fixed_feedback",
}


def git_lines(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in result.stdout.splitlines() if line]


def build_rows(root: Path) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for line in git_lines(root, "ls-files", "-s"):
        metadata, path = line.split("\t", 1)
        blob = metadata.split()[1]
        normalized = path.replace("\\", "/")
        for source_root in SOURCE_ROOTS:
            prefix = source_root + "/"
            if not normalized.startswith(prefix):
                continue
            remainder = normalized[len(prefix) :]
            if "/" not in remainder:
                break
            project = remainder.split("/", 1)[0]
            if project != "__pycache__":
                grouped[(source_root, project)].append((normalized, blob))
            break

    rows: list[dict[str, object]] = []
    for (source_root, project), entries in sorted(grouped.items()):
        digest = hashlib.sha256()
        for path, blob in sorted(entries):
            digest.update(path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(blob.encode("ascii"))
            digest.update(b"\n")
        is_fa = source_root == "FixedFeedbackSFT/projects"
        artifact_root = (
            f"FixedFeedbackSFT/runs/{project}"
            if is_fa
            else f"experiments/{project}/runs"
        )
        rows.append(
            {
                "source_family": SOURCE_ROOTS[source_root],
                "project": project,
                "physical_source_root": f"{source_root}/{project}",
                "python_module": f"experiments.{project}",
                "tracked_file_count": len(entries),
                "source_tree_sha256": digest.hexdigest(),
                "default_artifact_root": artifact_root,
                "source_sync_contract": "GitHub main -> local and server",
                "artifact_sync_contract": "server authoritative; manifest/checksum only",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "source_project_index.csv",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rows = build_rows(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    head = git_lines(root, "rev-parse", "HEAD")[0]
    print({"git_commit": head, "projects": len(rows), "output": str(output)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
