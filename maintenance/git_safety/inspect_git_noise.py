"""Read-only counts: distinguish branch names, untracked payloads and version deltas."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode("utf-8")


def inspect(root):
    paths = [p for p in git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0") if p]
    groups = Counter("/".join(p.split("/")[:2]) for p in paths)
    diff = git(root, "diff", "--numstat", "main", "HEAD").splitlines()
    numeric = [row.split("\t", 2) for row in diff if not row.startswith("-\t")]
    return {"schema_version": 1, "audited_on": "2026-10-04", "read_only": True,
            "working_branch": git(root, "branch", "--show-current").strip(),
            "working_head": git(root, "rev-parse", "HEAD").strip(),
            "published_main": git(root, "rev-parse", "main").strip(),
            "local_branch_count": len(git(root, "for-each-ref", "--format=%(refname)", "refs/heads").splitlines()),
            "registered_worktrees": git(root, "worktree", "list", "--porcelain").count("worktree "),
            "tracked_dirty": git(root, "status", "--porcelain", "-uno").splitlines(),
            "untracked_visible_files": len(paths), "untracked_visible_python": sum(p.endswith(".py") for p in paths),
            "largest_untracked_groups": groups.most_common(15),
            "working_commit_vs_main": {"changed_paths": len(diff),
                                       "added_lines": sum(int(r[0]) for r in numeric),
                                       "removed_lines": sum(int(r[1]) for r in numeric)},
            "warning": "Commit-vs-main differences are NOT a deletion receipt or uncommitted-file count. Do not merge whole branches to make this number zero."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.repo.resolve())
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
