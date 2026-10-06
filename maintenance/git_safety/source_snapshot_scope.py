"""Read-only identity audit of untracked source snapshots; never deletion approval."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess


SOURCE_SUFFIXES = {".py", ".sh", ".ps1", ".bat", ".cmd", ".yaml", ".yml"}


def source_path(root, relative):
    """Use Windows extended paths so deep package files are not falsely missing."""
    path = root / relative
    if os.name == "nt":
        text = str(path)
        if not text.startswith("\\\\?\\"):
            text = "\\\\?\\UNC\\" + text[2:] if text.startswith("\\\\") else "\\\\?\\" + text
        return Path(text)
    return path


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
        path = source_path(root, relative)
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


def history_refs(root):
    """Pin existing recovery references only; never create or change refs."""
    records = git(root, "for-each-ref", "--format=%(refname) %(objectname)", "refs/archive/").decode().splitlines()
    refs = dict(line.split(" ", 1) for line in records)
    # Multiple recovery aliases can point to the same commit. Bound actual
    # trees inspected, without deleting aliases or miscounting them as work.
    if len(refs) > 4096 or len(set(refs.values())) > 512:
        raise ValueError("Recovery inventory exceeds 4096 refs or 512 unique commits; choose a reviewed scope")
    return refs


def audit_history(root, reference="HEAD"):
    root = Path(root).resolve()
    refs = history_refs(root)
    current = audit(root, reference)
    index = defaultdict(list)
    by_commit = defaultdict(list)
    for name, commit in refs.items():
        by_commit[commit].append(name)
    for commit, names in by_commit.items():
        for record in git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
            if not record:
                continue
            metadata, path = record.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            relative = path.decode("utf8")
            if mode not in (b"100644", b"100755") or kind != b"blob" or Path(relative).suffix.lower() not in SOURCE_SUFFIXES:
                continue
            index[oid.decode()].append({"commit": commit, "archive_refs": names, "path": relative})
    rows = []
    for row in current["files"]:
        if row["identity"] != "no_byte_identity_in_reference":
            continue
        path = source_path(root, row["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("Source type changed during history audit: " + row["path"])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise RuntimeError("Source changed during history audit: " + row["path"])
        state, matches = identity(data, index)
        rows.append({"path": row["path"], "sha256": row["sha256"],
                     "history_identity": state, "archive_matches": matches})
    if history_refs(root) != refs or git(root, "rev-parse", "HEAD").decode().strip() != current["head"]:
        raise RuntimeError("Git identity changed during history audit; discard snapshot")
    return {"schema": 1, "head": current["head"], "reference_commit": current["reference_commit"],
            "archive_refs_checked": len(refs), "unique_archive_commits": len(by_commit),
            "source_counts_against_main": current["counts"], "nonmatching_sources_checked": len(rows),
            "history_counts": dict(Counter(row["history_identity"] for row in rows)),
            "files": rows, "mutations": [],
            "limitations": ["Matches prove existing local Git blob provenance, not main adoption or runtime compatibility.",
                            "Archive refs are not proof of an independent complete data backup.",
                            "No match does not mean garbage or an unneeded algorithm.",
                            "No deletion, move, ignore or source replacement is authorized by this audit."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--reference", default="HEAD")
    parser.add_argument("--history", action="store_true", help="Check existing recovery history (bounded to 512 unique commits, without creating refs)")
    parser.add_argument("--summary", action="store_true", help="Omit repeated provenance rows; retain counts and unresolved source names")
    args = parser.parse_args()
    result = audit_history(args.repo, args.reference) if args.history else audit(args.repo, args.reference)
    if args.summary:
        rows = result.pop('files')
        result['unresolved_source_paths'] = [row['path'] for row in rows
            if row.get('history_identity', row.get('identity')) == 'no_byte_identity_in_reference']
        result['detailed_rows_omitted'] = len(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
