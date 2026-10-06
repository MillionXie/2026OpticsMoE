"""Read-only identity audit of untracked source snapshots; never deletion approval."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


SOURCE_SUFFIXES = {".py", ".sh", ".ps1", ".bat", ".cmd", ".yaml", ".yml"}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def blob_id(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def identity(data, tracked):
    """Exact bytes first, then explicit CRLF normalization, never semantic inference."""
    exact = tracked.get(blob_id(data), [])
    if exact:
        return "exact_git_blob", exact
    normalized = tracked.get(blob_id(data.replace(b"\r\n", b"\n")), [])
    if normalized:
        return "crlf_normalized_git_blob", normalized
    return "no_byte_identity_in_reference", []


def audit(root, reference="HEAD"):
    root = Path(root).resolve()
    before = git(root, "rev-parse", "HEAD").decode().strip()
    pinned_reference = git(root, "rev-parse", reference + "^{commit}").decode().strip()
    tracked = defaultdict(list)
    for record in git(root, "ls-tree", "-r", "-z", pinned_reference).split(b"\0"):
        if not record:
            continue
        metadata, name = record.split(b"\t", 1)
        mode, kind, object_id = metadata.split()
        if kind == b"blob":
            tracked[object_id.decode()].append(name.decode("utf-8"))
    rows = []
    skipped = Counter()
    for name in git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not name:
            continue
        relative = name.decode("utf-8")
        path = root / relative
        if path.suffix.lower() not in SOURCE_SUFFIXES:
            skipped["not_selected_source_extension"] += 1
            continue
        if path.is_symlink() or not path.is_file():
            skipped["symlink_or_not_regular_file"] += 1
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            skipped["source_over_2_mib_manual_review"] += 1
            continue
        data = path.read_bytes()
        state, matches = identity(data, tracked)
        rows.append({"path": relative, "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest(),
                     "identity": state, "reference_paths": matches})
    after = git(root, "rev-parse", "HEAD").decode().strip()
    if before != after:
        raise RuntimeError("HEAD changed during audit; discard this snapshot")
    return {"schema": 1, "head": before,
            "reference_commit": pinned_reference,
            "source_file_count": len(rows), "counts": dict(Counter(r["identity"] for r in rows)),
            "top_level_counts": dict(Counter(r["path"].split("/")[0] for r in rows)),
            "top_level_identity_counts": {
                area: dict(Counter(r["identity"] for r in rows if r["path"].split("/")[0] == area))
                for area in sorted({r["path"].split("/")[0] for r in rows})},
            "skipped_counts": dict(skipped), "files": rows, "mutations": [],
            "limitations": ["Identity is not evidence that a runtime path is unused.",
                            "Nonmatching source is not necessarily a new algorithm.",
                            "Data, weights, timing reports and ignored files are outside this audit.",
                            "No file may be removed solely from this report."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--reference", default="HEAD")
    args = parser.parse_args()
    print(json.dumps(audit(args.repo, args.reference), ensure_ascii=False, indent=2))
