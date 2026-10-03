"""Append ONLY the reviewed 2026-10-04 payload block to a publication tree.

Other preexisting differences between the working branch and main are never
copied. Creates an unreferenced candidate with a temporary index, not a checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from prepare_scoped_main import git

RULES = {"node_modules/", "/.codex_tmp/", *("/handoffs/**/*." + extension for extension in
          ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "gif", "npy", "npz",
           "safetensors", "pth", "ckpt", "onnx", "mp4", "avi", "pdf", "pptx", "xlsx"))}
MARKER = "# 2026-10-04: installed presentation/frontend dependencies"


def reviewed_block(source: str) -> str:
    if source.count(MARKER) != 1:
        raise RuntimeError("Expected exactly one reviewed payload block")
    block = source[source.index(MARKER):].split("\n# Rebuildable baseline source handoffs;", 1)[0].rstrip() + "\n"
    rules = {line for line in block.splitlines() if line and not line.startswith("#")}
    if rules != RULES:
        raise RuntimeError("Unexpected ignore rules; never hide unreviewed source")
    return block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--expected-main", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    main_ref = git(root, "rev-parse", "main").decode().strip()
    head = git(root, "rev-parse", "HEAD").decode().strip()
    if main_ref != args.expected_main or "branch refs/heads/main\n" in git(root, "worktree", "list", "--porcelain").decode():
        raise RuntimeError("main changed or is checked out; abort")
    index = Path(git(root, "rev-parse", "--path-format=absolute", "--git-path", "index").decode().strip())
    before = hashlib.sha256(index.read_bytes()).hexdigest()
    git(root, "merge-base", "--is-ancestor", main_ref, args.base_commit)
    source = git(root, "show", args.source_commit + ":.gitignore").decode()
    old = git(root, "show", args.base_commit + ":.gitignore").decode()
    if MARKER in old:
        raise RuntimeError("Block already present; inspect instead of duplicating")
    block = reviewed_block(source)
    new = old.rstrip() + "\n\n" + block
    with tempfile.TemporaryDirectory(prefix="lightgen_ignore_index_") as scratch:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(scratch) / "index"))
        git(root, "read-tree", args.base_commit, env=env)
        blob = git(root, "hash-object", "-w", "--stdin", data=new.encode()).decode().strip()
        git(root, "update-index", "--cacheinfo", "100644", blob, ".gitignore", env=env)
        changed = git(root, "diff", "--cached", "--name-only", args.base_commit, env=env).decode().splitlines()
        if changed != [".gitignore"]:
            raise RuntimeError("Unexpected publication paths")
        tree = git(root, "write-tree", env=env).decode().strip()
        candidate = git(root, "commit-tree", tree, "-p", args.base_commit,
                        data=b"Append reviewed local payload ignores without replacing historical source rules\n").decode().strip()
    if (hashlib.sha256(index.read_bytes()).hexdigest() != before
            or git(root, "rev-parse", "HEAD").decode().strip() != head
            or git(root, "rev-parse", "main").decode().strip() != main_ref):
        raise RuntimeError("Concurrent index/HEAD/main change; do not publish")
    print(json.dumps({"candidate_commit": candidate, "parent": args.base_commit,
                      "source_commit": args.source_commit, "old_ignore_sha256": hashlib.sha256(old.encode()).hexdigest(),
                      "new_ignore_sha256": hashlib.sha256(new.encode()).hexdigest(),
                      "only_appended_reviewed_block": True, "user_index_unchanged": True}))


if __name__ == "__main__":
    main()
