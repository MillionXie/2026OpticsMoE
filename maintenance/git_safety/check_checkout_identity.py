"""Read-only content identity audit; never checkout, stage, refresh or overwrite files."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def classify(working, head, main):
    def normalized(value):
        return value.replace(b"\r\n", b"\n") if value is not None else None
    if normalized(working) == normalized(head):
        return "head_identical_after_eol_normalization"
    if main is not None and normalized(working) == normalized(main):
        return "already_matches_published_main"
    return "unique_working_change_preserve"


def inspect(root):
    root = root.resolve()
    head = git(root, "rev-parse", "HEAD").decode().strip()
    main = git(root, "rev-parse", "refs/heads/main").decode().strip()
    status = git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no")
    records = status.split(b"\0")
    rows = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        code, name = record[:2].decode(), record[3:].decode("utf-8")
        if "R" in code or "C" in code:
            index += 1
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("path escapes checkout")
        def blob(ref):
            result = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{name}"], capture_output=True)
            return result.stdout if result.returncode == 0 else None
        working = target.read_bytes() if target.is_file() else None
        old, published = blob(head), blob(main)
        def digest(value):
            return hashlib.sha256(value).hexdigest() if value is not None else None
        rows.append({"path": name, "status": code, "classification": classify(working, old, published)
                     if working is not None else "missing_working_file_preserve_history",
                     "working_sha256": digest(working), "head_sha256": digest(old),
                     "main_sha256": digest(published)})
    if head != git(root, "rev-parse", "HEAD").decode().strip() or main != git(root, "rev-parse", "refs/heads/main").decode().strip():
        raise RuntimeError("Git identity changed during audit; rerun")
    if status != git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no"):
        raise RuntimeError("working status changed during audit; rerun")
    return {"head": head, "main": main, "rows": rows,
            "counts": dict(Counter(row["classification"] for row in rows)),
            "read_only": True, "safe_to_overwrite": False,
            "scope": "Tracked working changes only; not full runtime/dependency or untracked-source audit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional audit receipt; no source mutation")
    args = parser.parse_args()
    report = inspect(Path(__file__).resolve().parents[2])
    payload = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError("Never overwrite an earlier receipt")
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
