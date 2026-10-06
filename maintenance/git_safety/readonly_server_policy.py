"""Public read-only management policy; no connections, credentials or mutations."""
import argparse
import json
from pathlib import PurePosixPath


def validate_request(phase, repo, checkout=None, commit=None):
    if phase in {"sync", "publish-bundle"}:
        raise ValueError("Legacy write phase retired: use reviewed main-only Git synchronization; no automatic task worktrees or branches")
    if phase not in {"inspect", "test"}:
        raise ValueError("Only inspect/test are permitted")
    checkout = checkout or repo
    if any(not PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts for p in (repo, checkout)):
        raise ValueError("Require absolute paths without parent traversal")
    if phase == "test" and (not commit or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)):
        raise ValueError("test requires an exact 40-character commit SHA")
    return repo, checkout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--repo", default="/data/repository")
    parser.add_argument("--checkout")
    parser.add_argument("--commit")
    args = parser.parse_args()
    try:
        repo, checkout = validate_request(args.phase, args.repo, args.checkout, args.commit)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"phase": args.phase, "repo": repo, "checkout": checkout,
                      "commit": args.commit, "mutations": [], "connected": False}))


if __name__ == "__main__":
    main()
