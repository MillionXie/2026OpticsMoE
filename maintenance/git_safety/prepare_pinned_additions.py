"""Prepare missing-only, SHA-reviewed source additions without changing any checkout.

No source copy, branch creation, checkout, ref update or push. Existing different
files in the target tree are a hard error, even if the source is newer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile

from prepare_scoped_main import git
from review_git import forbidden_artifact


def prepare(root: Path, manifest: dict, message: str, base_commit: str | None = None) -> dict:
    root = root.resolve()
    main = git(root, "rev-parse", "refs/heads/main").decode().strip()
    head = git(root, "rev-parse", "HEAD").decode().strip()
    if "branch refs/heads/main\n" in git(root, "worktree", "list", "--porcelain").decode():
        raise RuntimeError("main is checked out; stop publication")
    index = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").decode().strip())
    before = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    base = git(root, "rev-parse", "--verify", (base_commit or main) + "^{commit}").decode().strip()
    git(root, "merge-base", "--is-ancestor", main, base)
    source = git(root, "rev-parse", "--verify", manifest["source_commit"] + "^{commit}").decode().strip()
    reviewed, additions, identical = set(), [], []
    for row in manifest["paths"]:
        path = row["path"]
        pure = PurePosixPath(path)
        if (pure.is_absolute() or ".." in pure.parts or str(pure) != path
                or "\\" in path or ":" in path or any(ord(c) < 32 for c in path)
                or path in reviewed or pure.name == "server_sync.py"):
            raise RuntimeError("Unsafe/duplicate path: " + path)
        reviewed.add(path)
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise RuntimeError("Missing reviewed SHA: " + path)
        entry = git(root, "ls-tree", "-z", source, "--", path)
        if not entry:
            raise RuntimeError("Missing source: " + path)
        mode, kind, oid = entry.split(b"\t", 1)[0].split()
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise RuntimeError("Special source entry: " + path)
        data = git(root, "cat-file", "blob", oid.decode())
        reason = forbidden_artifact(path, len(data))
        if reason or b"\0" in data:
            raise RuntimeError("Unsafe source: " + path + ": " + (reason or "binary content"))
        data.decode("utf-8")
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise RuntimeError("Reviewed SHA/size mismatch: " + path)
        target = git(root, "ls-tree", "-z", base, "--", path)
        if target:
            if target != entry:
                raise RuntimeError("Existing different target; never overwrite: " + path)
            identical.append(path)
        else:
            additions.append((path, mode.decode(), oid.decode()))
    if not additions:
        raise RuntimeError("No missing additions")
    with tempfile.TemporaryDirectory(prefix="lightgen_pinned_index_") as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))
        git(root, "read-tree", base, env=env)
        for path, mode, oid in additions:
            git(root, "update-index", "--add", "--cacheinfo", mode, oid, path, env=env)
        changed = set(filter(None, git(root, "diff", "--cached", "--name-only", "-z", base, env=env).decode().split("\0")))
        if changed != {row[0] for row in additions} or git(root, "ls-files", "-u", env=env):
            raise RuntimeError("Unexpected candidate tree")
        tree = git(root, "write-tree", env=env).decode().strip()
        candidate = git(root, "commit-tree", tree, "-p", base, env=env,
                        data=(message + "\n").encode()).decode().strip()
    after = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    if (before != after or head != git(root, "rev-parse", "HEAD").decode().strip()
            or main != git(root, "rev-parse", "refs/heads/main").decode().strip()):
        raise RuntimeError("Concurrent index/HEAD/main change; candidate not published")
    return {"base_main": main, "candidate_parent": base, "source_commit": source,
            "candidate_commit": candidate, "added_paths": sorted(changed),
            "identical_paths": identical, "working_head_unchanged": True,
            "user_index_unchanged": True, "refs_updated": False, "pushed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--base-commit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(prepare(root, json.loads(args.manifest.read_text(encoding="utf-8")),
                             args.message, args.base_commit), indent=2))


if __name__ == "__main__":
    main()
