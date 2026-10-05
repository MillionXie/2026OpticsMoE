"""Read-only Git reconciliation and staged-artifact checks; never edits Git refs."""
import argparse
import json
from collections import Counter
from pathlib import Path, PurePosixPath
import subprocess
import sys


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args]).decode("utf-8")


def forbidden_artifact(path, size):
    """Conservative staging guard, not a classification of disposable data."""
    relative = PurePosixPath(path.replace("\\", "/"))
    parts = {part.lower() for part in relative.parts}
    if str(relative).lower().endswith(".tar.gz") or relative.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".zip",
                                   ".bundle", ".tar", ".tgz", ".npy", ".npz"}:
        return "weight/cache/transport artifact"
    if parts.intersection({"runs", "data", "datasets", ".codex_tmp", ".worktrees",
                           "rejected_dark", "ccd_raw"}):
        return "runtime/private directory"
    if size > 10 * 1024 * 1024:
        return "larger than 10 MiB; manual source review required"
    return None


def staged_review(root):
    files = filter(None, git(root, "diff", "--cached", "--name-only",
                             "--diff-filter=ACMRT", "-z").split("\0"))
    issues = []
    for path in files:
        size = int(git(root, "cat-file", "-s", ":" + path).strip())
        reason = forbidden_artifact(path, size)
        if reason:
            issues.append({"path": path, "bytes": size, "reason": reason})
    return issues


def tree_difference_scope(root, reference, main="main"):
    """Distinguish old-only content from main additions; not a deletion approval."""
    main_paths = set(filter(None, git(root, "ls-tree", "-r", "--name-only", "-z", main).split("\0")))
    ref_paths = set(filter(None, git(root, "ls-tree", "-r", "--name-only", "-z", reference).split("\0")))
    changed = set(filter(None, git(root, "diff", "--name-only", "--no-renames", "-z", main, reference).split("\0")))
    return {
        "main_only_paths": len(main_paths - ref_paths),
        "reference_only_paths": len(ref_paths - main_paths),
        "shared_paths_with_changed_content": len(changed & main_paths & ref_paths),
        "total_changed_paths": len(changed),
        "working_overlay_included": False,
        "interpretation": "Path differences are not missing commits or evidence of failed synchronization; reference-only and changed shared paths still require source/asset review.",
    }


def audit(root):
    status = git(root, "status", "--porcelain", "-uno").splitlines()
    branch = git(root, "branch", "--show-current").strip() or "(detached)"
    branches = git(root, "for-each-ref", "--format=%(refname)", "refs/heads").splitlines()
    trees = git(root, "worktree", "list", "--porcelain").split("\n\n")
    checked_out = {line[7:] for tree in trees for line in tree.splitlines()
                   if line.startswith("branch ")}
    refs = git(root, "for-each-ref", "--format=%(refname)",
               "refs/archive/server-authoritative-20261002").splitlines()
    authorities = []
    for ref in refs:
        delta = list(filter(None, git(root, "diff", "--name-only", "--no-renames", "-z", "main", ref).split("\0")))
        areas = Counter()
        for path in delta:
            parts = path.split("/")
            area = "/".join(parts[:3] if parts[:2] == ["LightGenV2", "tasks"] else
                            parts[:2] if parts[0] in {"LightGenV2", "experiments", "LightGenPublic",
                                                    "FixedFeedbackSFT", "TransferFromElectricity"} else parts[:1])
            areas[area] += 1
        authorities.append({"ref": ref, "head": git(root, "rev-parse", ref).strip(),
                            "changed_paths_vs_main": len(delta),
                            "areas": dict(areas.most_common()),
                            "requires_overlay_review": True,
                            "difference_scope": tree_difference_scope(root, ref)})
    merged = git(root, "for-each-ref", "--merged=main", "--format=%(refname)",
                 "refs/heads").splitlines()
    return {
        "repo": str(root), "branch": branch, "head": git(root, "rev-parse", "HEAD").strip(),
        "main": git(root, "rev-parse", "main").strip(),
        "working_head_vs_main": tree_difference_scope(root, "HEAD"),
        "tracked_dirty_count": len(status), "tracked_dirty_status": status,
        "local_branch_count": len(branches),
        "registered_worktree_count": sum(tree.startswith("worktree ") for tree in trees),
        "already_merged_unused_branches": [ref for ref in merged
                                           if ref not in checked_out and ref != "refs/heads/main"],
        "server_authorities": authorities, "staged_artifact_issues": staged_review(root),
        "mutations": [],
        "warning": "HEAD refs omit server working edits. Do not merge entire old branches or delete from this report alone.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--check-staged", action="store_true")
    args = parser.parse_args()
    root = Path(git(Path(args.repo).resolve(), "rev-parse", "--show-toplevel").strip())
    if args.check_staged:
        issues = staged_review(root)
        print(json.dumps({"staged_artifact_issues": issues, "mutations": []}, indent=2))
        return 1 if issues else 0
    print(json.dumps(audit(root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
