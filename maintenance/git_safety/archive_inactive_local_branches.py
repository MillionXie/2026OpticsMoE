"""Retire unused LOCAL branch names without merging or losing their history.

Requires an explicit --apply. Creates archive refs and a verified private Git
bundle first; conditionally removes only the planned head refs. Working trees,
remote branches, the user index and experiment files are never changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PREFIX = "refs/archive/local-inactive-20261004/"


def git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def metadata_path(root: Path, name: str) -> Path:
    # --path-format is unavailable on Git 2.25 and can be echoed as plain text.
    value = git(root, 'rev-parse', '--git-path', name).decode().strip()
    if not value or '\n' in value or '\r' in value:
        raise RuntimeError('Ambiguous Git metadata path')
    path = Path(value)
    return path if path.is_absolute() else root / path


def state(root: Path) -> dict:
    index = metadata_path(root, 'index')
    return {
        "head": git(root, "rev-parse", "HEAD").decode().strip(),
        "main": git(root, "rev-parse", "refs/heads/main").decode().strip(),
        "worktrees": git(root, "worktree", "list", "--porcelain").decode(),
        "index_sha256": digest(index) if index.exists() else None,
        "tracked_status_sha256": hashlib.sha256(git(root, "status", "--porcelain", "-uno")).hexdigest(),
    }


def plan(root: Path) -> dict:
    before = state(root)
    checked = {line[7:] for line in before["worktrees"].splitlines() if line.startswith("branch ")}
    refs = git(root, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads").decode().splitlines()
    rows = []
    for line in refs:
        ref, head = line.split()
        if ref == "refs/heads/main" or ref in checked:
            continue
        name = ref.removeprefix("refs/heads/")
        rows.append({"branch": name, "original_ref": ref, "head": head,
                     "archive_ref": PREFIX + name,
                     "purpose_from_commit_subject": git(root, "log", "-1", "--format=%s", head).decode().strip(),
                     "merged_into_main": subprocess.run(
                         ["git", "-C", str(root), "merge-base", "--is-ancestor", head, before["main"]],
                         capture_output=True).returncode == 0})
    return {"schema_version": 1, "audited_on": "2026-10-04", "scope": "local branch names only",
            "before": before, "branch_count_before": len(refs), "branches": rows,
            "note": "Archival is NOT scientific-source integration into main. No remote refs are removed."}


def apply(root: Path, receipt: dict, destination: Path) -> dict:
    if state(root) != receipt["before"]:
        raise RuntimeError("HEAD/main/index/worktree/status changed; aborting")
    if not receipt["branches"]:
        raise RuntimeError("No inactive local branches")
    destination.mkdir(parents=True, exist_ok=False)
    config = metadata_path(root, 'config')
    if config.is_file():
        (destination / "original_git_config.private").write_bytes(config.read_bytes())
    # A single --stdin batch is atomic, including on older server Git versions.
    # Explicit start/prepare/commit commands require newer Git and are not needed.
    creations = []
    for row in receipt["branches"]:
        creations.extend((f"verify {row['original_ref']} {row['head']}",
                          f"create {row['archive_ref']} {row['head']}"))
    git(root, "update-ref", "--stdin", data=("\n".join(creations) + "\n").encode())
    bundle = destination / "inactive_local_branches.bundle"
    # The published main commit is a recorded prerequisite, retained separately.
    git(root, "bundle", "create", str(bundle),
        *(row["archive_ref"] for row in receipt["branches"]), "^" + receipt["before"]["main"])
    git(root, "bundle", "verify", str(bundle))
    bundled = {line.split()[1]: line.split()[0]
               for line in git(root, "bundle", "list-heads", str(bundle)).decode().splitlines()}
    for row in receipt["branches"]:
        if bundled.get(row["archive_ref"]) != row["head"]:
            raise RuntimeError("Bundle ref mismatch; branch names have NOT been removed")
    if state(root) != receipt["before"]:
        raise RuntimeError("Concurrent state change; branch names have NOT been removed")
    deletions = [f"verify refs/heads/main {receipt['before']['main']}"]
    for row in receipt["branches"]:
        deletions.extend((f"verify {row['archive_ref']} {row['head']}",
                          f"delete {row['original_ref']} {row['head']}"))
    git(root, "update-ref", "--stdin", data=("\n".join(deletions) + "\n").encode())
    after = state(root)
    if after != receipt["before"]:
        raise RuntimeError("Unexpected state change after archival; inspect before further work")
    result = {**receipt, "applied": True, "after": after,
              "branch_count_after": len(git(root, "for-each-ref", "--format=%(refname)", "refs/heads").splitlines()),
              "bundle": str(bundle), "bundle_sha256": digest(bundle),
              "bundle_prerequisite_main": receipt["before"]["main"],
              "config_backup_private": True, "working_files_changed": False,
              "remote_refs_changed": False, "experiment_data_deleted": False}
    (destination / "receipt.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-directory", type=Path)
    args = parser.parse_args()
    root = args.repo.resolve()
    receipt = plan(root)
    if args.apply:
        if not args.backup_directory:
            parser.error("--apply requires --backup-directory")
        receipt = apply(root, receipt, args.backup_directory.resolve())
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"branch_count_before": receipt["branch_count_before"],
                      "inactive_branches": len(receipt["branches"]), "applied": args.apply,
                      "branch_count_after": receipt.get("branch_count_after"),
                      "bundle_sha256": receipt.get("bundle_sha256")}))


if __name__ == "__main__":
    main()
