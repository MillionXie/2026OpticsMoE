"""Read-only, content-hashed inventory of the four legacy sibling exports.

Does not relocate, delete, repair Git metadata, or read credential contents into
the report. Output is a generated inventory, not a scientific version decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

NAMES = (
    "2026OpticsMoE_a100_formal",
    "2026OpticsMoE_abo_handoff",
    "2026OpticsMoE_lsp_handoff",
    "2026OpticsMoE_lsp_refinement",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, repository: Path) -> dict:
    rows = []
    equal = different = absent = 0
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise RuntimeError(f"Symlink requires a separate dependency audit: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix()
        digest = sha256(item)
        counterpart = repository / relative
        if counterpart.is_file():
            relation = "same" if sha256(counterpart) == digest else "different"
        else:
            relation = "absent"
        equal += relation == "same"
        different += relation == "different"
        absent += relation == "absent"
        rows.append({"path": relative, "bytes": item.stat().st_size,
                     "sha256": digest, "main_working_file_relation": relation})
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    return {"original_path": str(root.resolve()),
            "git_metadata_present": (root / ".git").exists(),
            "git_valid": git.returncode == 0,
            "git_head": git.stdout.strip() if git.returncode == 0 else None,
            "files": len(rows), "bytes": sum(row["bytes"] for row in rows),
            "same_as_main_working": equal, "different_from_main_working": different,
            "absent_from_main_working": absent, "entries": rows}


def verify(root: Path, rows: list[dict]) -> None:
    # Relocation adds a directory level. Windows' legacy 260-character API
    # limit must not turn existing long-path evidence into a false "missing".
    if os.name == "nt":
        absolute = str(root.resolve())
        if not absolute.startswith("\\\\?\\"):
            root = Path("\\\\?\\" + absolute)
    expected = {row["path"] for row in rows}
    actual = {item.relative_to(root).as_posix() for item in root.rglob("*")
              if item.is_file()}
    if expected != actual:
        raise RuntimeError(f"File set changed: {root}")
    for row in rows:
        item = root / row["path"]
        if item.stat().st_size != row["bytes"] or sha256(item) != row["sha256"]:
            raise RuntimeError(f"Content changed: {item}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--archive-root", type=Path)
    args = parser.parse_args()
    repo = args.repository.resolve()
    if args.verify_manifest:
        manifest = json.loads(args.verify_manifest.read_text(encoding="utf-8"))
        for name, record in manifest["projects"].items():
            if name not in NAMES:
                raise RuntimeError(f"Unexpected project: {name}")
            root = args.archive_root / name if args.archive_root else repo.parent / name
            verify(root, record["entries"])
        print(json.dumps({"verified": list(manifest["projects"]), "content_unchanged": True}))
        return
    if not args.output:
        parser.error("--output is required for an inventory")
    result = {"schema_version": 1, "audited_on": "2026-10-04",
              "main_repository": str(repo), "operation": "read_only_inventory",
              "comparison_note": "Compared with working files, NOT an approval to replace main.",
              "projects": {name: inventory(repo.parent / name, repo) for name in NAMES}}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in record.items() if k != "entries"}
                      for name, record in result["projects"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
