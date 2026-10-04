"""Prepare the specifically reviewed T06 dependency closure without a checkout.

This does not generalize the task-only replacement rule to arbitrary shared
files. Only the release-bound T06 package and its named historical backend are
allowed; newer main lab_runtime fixes remain untouched. Candidate tests and a
separate normal publication are still required. No ref, index or file changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from prepare_scoped_main import git
from review_git import forbidden_artifact

TASK = "LightGenV2/tasks/t06_video_quality_assessment/"
BACKEND = "experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/"
SOURCE = "8e869473787f4ffceb2a6a77f4430b94c206f459"
PRESERVE_MAIN = {TASK + "lab_runtime.py"}


def prepare(root: Path, audit: dict, message: str) -> dict:
    root = root.resolve()
    main = git(root, "rev-parse", "main").decode().strip()
    head = git(root, "rev-parse", "HEAD").decode().strip()
    if main != audit["main_commit"] or audit["source_commit"] != SOURCE:
        raise RuntimeError("Stale main or unreviewed source commit")
    if any(not row["matches"] for row in audit["profile_source_contracts"]):
        raise RuntimeError("Pinned profile/backend source mismatch")
    if "branch refs/heads/main\n" in git(root, "worktree", "list", "--porcelain").decode():
        raise RuntimeError("main is checked out; stop")
    index = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").decode().strip())
    before = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    changes, seen = [], set()
    for row in audit["files"]:
        path = row["path"]
        pure = PurePosixPath(path)
        if (str(pure) != path or pure.is_absolute() or ".." in pure.parts
                or "\\" in path or ":" in path or path in seen
                or any(ord(c) < 32 for c in path)
                or pure.suffix not in {".py", ".yaml"}):
            raise RuntimeError("Unsafe closure path: " + path)
        seen.add(path)
        if not path.startswith((TASK, BACKEND)):
            # Existing VQA-109 utilities are read-only dependencies, never
            # replacement targets in this T06 migration.
            if path not in {"LightGenV2/__init__.py", "LightGenV2/tasks/__init__.py", "experiments/__init__.py",
                            "experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/__init__.py",
                            "experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/modeling.py",
                            "experiments/lgvq_four_stage_optical_electronic_109_no_attention_vqa/settings.py"}:
                raise RuntimeError("Unaudited consumer/shared path: " + path)
            if row["main_relation"] != "identical":
                raise RuntimeError("Shared initializer differs; separate review required")
        source = git(root, "show", SOURCE + ":" + path)
        if len(source) != row["bytes"] or hashlib.sha256(source).hexdigest() != row["source_sha256"]:
            raise RuntimeError("Pinned source SHA mismatch: " + path)
        target_entry = git(root, "ls-tree", "-z", main, "--", path)
        target = git(root, "cat-file", "blob", target_entry.split(b"\t", 1)[0].split()[-1].decode()) if target_entry else None
        target_sha = hashlib.sha256(target).hexdigest() if target is not None else None
        if target_sha != row["main_sha256"]:
            raise RuntimeError("Old main SHA mismatch: " + path)
        if path in PRESERVE_MAIN or target == source:
            continue
        if forbidden_artifact(path, len(source)) or b"\0" in source:
            raise RuntimeError("Forbidden source artifact: " + path)
        if path.endswith(".py"):
            compile(source.decode("utf-8"), path, "exec")
        else:
            source.decode("utf-8")
        entry = git(root, "ls-tree", "-z", SOURCE, "--", path)
        mode, kind, oid = entry.split(b"\t", 1)[0].split()
        if mode not in (b"100644", b"100755") or kind != b"blob":
            raise RuntimeError("Special Git entry")
        changes.append((path, mode.decode(), oid.decode()))
    if not changes:
        raise RuntimeError("No closure changes")
    with tempfile.TemporaryDirectory(prefix="lightgen_t06_index_") as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))
        git(root, "read-tree", main, env=env)
        for path, mode, oid in changes:
            git(root, "update-index", "--add", "--cacheinfo", mode, oid, path, env=env)
        changed = set(filter(None, git(root, "diff", "--cached", "--name-only", "-z", main, env=env).decode().split("\0")))
        if changed != {row[0] for row in changes} or git(root, "ls-files", "-u", env=env):
            raise RuntimeError("Unexpected candidate tree")
        tree = git(root, "write-tree", env=env).decode().strip()
        candidate = git(root, "commit-tree", tree, "-p", main, env=env, data=(message + "\n").encode()).decode().strip()
    after = hashlib.sha256(index.read_bytes()).hexdigest() if index.exists() else None
    if before != after or head != git(root, "rev-parse", "HEAD").decode().strip() or main != git(root, "rev-parse", "main").decode().strip():
        raise RuntimeError("Concurrent Git mutation; candidate not published")
    return {"base_main": main, "candidate_commit": candidate, "source_commit": SOURCE,
            "changed_paths": sorted(changed), "preserved_main_paths": sorted(PRESERVE_MAIN),
            "candidate_runtime_test_required": True, "new_checkout_created": False,
            "working_head_unchanged": True, "user_index_unchanged": True,
            "refs_updated": False, "pushed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[2], json.loads(args.audit.read_text(encoding="utf-8")), args.message), indent=2))
