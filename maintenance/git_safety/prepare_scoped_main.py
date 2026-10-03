"""Prepare audited commit deltas on main without touching HEAD, refs or user index.

Writes only temporary Git index and unreferenced Git objects. Never checks out,
merges a whole branch, updates a ref, pushes, or alters working files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from review_git import forbidden_artifact


def git(root, *args, env=None, data=None):
    result = subprocess.run(["git", "-C", str(root), *args], input=data,
                            capture_output=True, env=env)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def prepare(root: Path, commits: list[str], message: str, paths: list[str] | None = None) -> dict:
    root = root.resolve()
    main = git(root, "rev-parse", "refs/heads/main").decode().strip()
    head = git(root, "rev-parse", "HEAD").decode().strip()
    trees = git(root, "worktree", "list", "--porcelain").decode()
    if "branch refs/heads/main\n" in trees:
        raise RuntimeError("main is checked out; do not prepare behind its working tree")
    index_path = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").decode().strip())
    index_before = hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None
    source_commits, allowed = [], set()
    selected = {}
    for ref in commits:
        commit = git(root, "rev-parse", "--verify", ref + "^{commit}").decode().strip()
        source_commits.append(commit)
        if len(git(root, "rev-list", "--parents", "-n", "1", commit).split()) != 2:
            raise RuntimeError("Each input must be a reviewed non-merge commit")
        changed = set(filter(None, git(root, "diff-tree", "--no-commit-id", "--name-only",
                                       "-r", "-z", commit).decode().split("\0")))
        if paths is not None:
            changed = {name for name in changed if any(
                name.startswith(prefix) if prefix.endswith("/") else name == prefix for prefix in paths)}
        selected[commit] = sorted(changed)
        allowed.update(changed)
    with tempfile.TemporaryDirectory(prefix="lightgen_git_index_") as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))
        git(root, "read-tree", main, env=env)
        for commit in source_commits:
            if not selected[commit]:
                continue
            patch = git(root, "diff", "--binary", commit + "^", commit, "--", *selected[commit])
            git(root, "apply", "--cached", "--3way", "--whitespace=nowarn", "-", env=env, data=patch)
        if git(root, "ls-files", "-u", env=env):
            raise RuntimeError("Unmerged entries; manual source review required")
        changed = set(filter(None, git(root, "diff", "--cached", "--name-only", "-z", main,
                                       env=env).decode().split("\0")))
        if not changed or not changed.issubset(allowed):
            raise RuntimeError("Unexpected/no publication changes")
        issues = []
        for relative in sorted(changed):
            if relative.endswith("/server_sync.py"):
                issues.append(relative + ": private connection helper")
            entry = git(root, "ls-files", "-s", "--", relative, env=env)
            if entry:
                mode = entry.split()[0]
                size = int(git(root, "cat-file", "-s", ":" + relative, env=env))
                reason = forbidden_artifact(relative, size)
                if mode not in (b"100644", b"100755") or reason:
                    issues.append(relative + ": " + (reason or "special tree entry"))
        if issues:
            raise RuntimeError("Unsafe publication paths: " + "; ".join(issues))
        tree = git(root, "write-tree", env=env).decode().strip()
        candidate = git(root, "commit-tree", tree, "-p", main, env=env,
                        data=(message + "\n").encode()).decode().strip()
    index_after = hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None
    if index_before != index_after or head != git(root, "rev-parse", "HEAD").decode().strip():
        raise RuntimeError("User index/HEAD changed concurrently; candidate was not published")
    if main != git(root, "rev-parse", "refs/heads/main").decode().strip():
        raise RuntimeError("main changed concurrently; candidate was not published")
    return {"base_main": main, "candidate_commit": candidate, "source_commits": source_commits,
            "changed_paths": sorted(changed), "working_head_unchanged": True,
            "user_index_unchanged": True, "refs_updated": False, "pushed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="append", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--path", action="append", help="Only reviewed exact paths or prefixes ending with /")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(prepare(root, args.commit, args.message, args.path), indent=2))


if __name__ == "__main__":
    main()
