"""Read-only canonical-entry and source-reference checks. Never moves/deletes data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
import subprocess
from urllib.parse import unquote


def inspect_tree(root: Path, commit: str) -> dict:
    """Check the published tree, independently of a protected dirty checkout."""
    resolved = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--verify",
                                        commit + "^{commit}"], text=True).strip()
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "-z", resolved])
    blobs = {}
    for entry in entries.split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            mode, kind, oid = metadata.split()
            if kind == b"blob" and mode in (b"100644", b"100755"):
                blobs[path.decode()] = oid.decode()
    def read(path):
        return subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", blobs[path]])
    def relative(path):
        normalized = posixpath.normpath("LightGenV2/" + path)
        if normalized.startswith("../") or normalized.startswith("/") or "\\" in normalized:
            raise ValueError("path escapes repository: " + path)
        return normalized
    registry = json.loads(read("LightGenV2/TASK_REGISTRY.json"))
    errors, private, identifiers = [], [], set()
    def required(path, label):
        if path not in blobs:
            if path.startswith("handoffs/"):
                private.append(label + ": " + path)
            else:
                errors.append("missing " + label + ": " + path)
    for task in registry["tasks"]:
        if task["id"] in identifiers:
            errors.append("duplicate task id: " + task["id"])
        identifiers.add(task["id"])
        for path in [task["entry"], *task.get("evidence", []), *([task["reproduction"]] if task.get("reproduction") else [])]:
            required(relative(path), "entry/evidence")
        for weight in task.get("weights", []):
            if not re.fullmatch(r"[0-9a-f]{64}", weight["sha256"]):
                errors.append("invalid weight identity: " + task["id"])
    for document in registry.get("navigation_documents", []):
        path = relative(document)
        required(path, "navigation document")
        if path in blobs:
            for link in re.findall(r"\]\(([^\s)]+)\)", read(path).decode("utf-8")):
                if ":" not in link and not link.startswith("#"):
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(link.split("#", 1)[0])))
                    if target.startswith("../") or target.startswith("/"):
                        errors.append("navigation escapes repository: " + link)
                    elif target not in blobs and not any(name.startswith(target.rstrip("/") + "/") for name in blobs):
                        required(target, "navigation target")
    for name in registry.get("source_import_manifests", []):
        path = relative(name)
        required(path, "source manifest")
        if path not in blobs:
            continue
        manifest = json.loads(read(path))
        for row in [*manifest["files"], *manifest.get("additional_dependency_hashes", [])]:
            required(row["path"], "imported source")
            if row["path"] in blobs:
                payload = read(row["path"])
                normalization = row.get("hash_normalization")
                if normalization == "crlf_to_lf":
                    payload = payload.replace(b"\r\n", b"\n")
                elif normalization is not None:
                    errors.append("unknown hash normalization: " + row["path"])
                if hashlib.sha256(payload).hexdigest() != row["source_blob_sha256"]:
                    errors.append("imported source SHA mismatch: " + row["path"])
        for dependency in manifest.get("additional_existing_dependencies", []):
            required(dependency, "compatibility dependency")
    for manifest in registry.get("artifact_manifests", []):
        required(relative(manifest), "artifact identity manifest")
    return {"commit": resolved, "entries_checked": len(identifiers), "errors": errors,
            "private_artifact_validation_required": private, "read_only": True,
            "working_files_used": False, "migration_complete": registry["migration_complete"]}


def inspect(root: Path, verify_weights: bool = False, verify_private: bool = False) -> dict:
    base = root / "LightGenV2"
    registry = json.loads((base / "TASK_REGISTRY.json").read_text(encoding="utf-8"))
    errors, pending, checked, unavailable_private = [], [], [], []
    verify_private = verify_private or verify_weights
    identifiers = set()
    for task in registry["tasks"]:
        task_id = task["id"]
        if task_id in identifiers:
            errors.append(f"duplicate task id: {task_id}")
        identifiers.add(task_id)
        paths = [task["entry"], *task.get("evidence", [])]
        if task.get("reproduction"):
            paths.append(task["reproduction"])
        for relative in paths:
            target = (base / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                errors.append(f"entry escapes repository: {task_id}: {relative}")
            elif not target.is_file():
                if relative.startswith("../handoffs/") and not verify_private:
                    unavailable_private.append(f"{task_id}: {relative}")
                else:
                    errors.append(f"missing entry/evidence: {task_id}: {relative}")
        ref = task.get("protected_git_ref")
        if ref:
            result = subprocess.run(["git", "-C", str(root), "rev-parse", "--verify", ref],
                                    capture_output=True, text=True, encoding="utf-8")
            if result.returncode and not verify_private:
                unavailable_private.append(f"protected source archive ref: {task_id}")
            elif result.returncode or result.stdout.strip() != task["source_head"]:
                errors.append(f"source ref mismatch: {task_id}: {ref}")
        for weight in task.get("weights", []):
            if not re.fullmatch(r"[0-9a-f]{64}", weight["sha256"]):
                errors.append(f"invalid SHA256: {task_id}")
            if verify_weights and weight.get("path"):
                path = (base / weight["path"]).resolve()
                if not path.is_relative_to(root.resolve()):
                    errors.append(f"weight path escapes repository: {task_id}")
                elif not path.is_file():
                    errors.append(f"missing local weight: {task_id}: {path}")
                else:
                    with path.open("rb") as handle:
                        actual = hashlib.file_digest(handle, "sha256").hexdigest()
                    if actual != weight["sha256"]:
                        errors.append(f"weight SHA mismatch: {task_id}: {path.name}")
        status = task["source_status"]
        if any(word in status for word in ("pending", "not_yet", "not_started")):
            pending.append({"id": task_id, "status": status})
        checked.append(task_id)
    for document in registry.get("navigation_documents", []):
        path = base / document
        if not path.is_file():
            errors.append(f"missing navigation document: {document}")
            continue
        for relative in re.findall(r"\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if ":" in relative or relative.startswith("#"):
                continue
            target = path.parent / unquote(relative.split("#", 1)[0])
            if not target.resolve().is_relative_to(root.resolve()):
                errors.append(f"navigation escapes repository: {document}: {relative}")
            elif not target.exists():
                if "handoffs" in target.parts and not verify_private:
                    unavailable_private.append(f"navigation: {document}: {relative}")
                else:
                    errors.append(f"broken navigation link: {document}: {relative}")
    for source_manifest in registry.get("source_import_manifests", []):
        manifest_path = base / source_manifest
        if not manifest_path.resolve().is_relative_to(root.resolve()) or not manifest_path.is_file():
            errors.append(f"invalid source manifest: {source_manifest}")
            continue
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in [*source["files"], *source.get("additional_dependency_hashes", [])]:
            path = root / row["path"]
            if not path.resolve().is_relative_to(root.resolve()):
                errors.append(f"source escapes repository: {row['path']}")
                continue
            if not path.is_file():
                errors.append(f"missing imported source: {row['path']}")
                continue
            payload = path.read_bytes()
            normalization = row.get("hash_normalization")
            if normalization == "crlf_to_lf":
                payload = payload.replace(b"\r\n", b"\n")
            elif normalization is not None:
                errors.append(f"unknown source hash normalization: {row['path']}")
            if hashlib.sha256(payload).hexdigest() != row["source_blob_sha256"]:
                errors.append(f"imported source SHA mismatch: {row['path']}")
        for relative in source.get("additional_existing_dependencies", []):
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()):
                errors.append(f"dependency escapes repository: {relative}")
            elif not target.is_file():
                errors.append(f"missing compatibility dependency: {relative}")
    for artifact_manifest in registry.get("artifact_manifests", []):
        path = (base / artifact_manifest).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            errors.append(f"invalid artifact manifest: {artifact_manifest}")
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        for report in artifact["reports"]:
            target = (root / report["repository_relative_path"]).resolve()
            if not target.is_relative_to(root.resolve()):
                errors.append(f"report escapes repository: {target}")
            elif not target.is_file():
                if "handoffs" in target.parts and not verify_private:
                    unavailable_private.append(f"original report: {target.name}")
                else:
                    errors.append(f"missing original report: {target.name}")
            elif hashlib.sha256(target.read_bytes()).hexdigest() != report["sha256"]:
                errors.append(f"original report SHA mismatch: {target.name}")
    return {"entries_checked": len(checked), "errors": errors, "pending_source_integration": pending,
            "migration_complete": registry["migration_complete"], "weights_checked": verify_weights,
            "read_only": True, "private_artifact_validation_required": unavailable_private,
            "private_artifacts_required": verify_private}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-local-weights", action="store_true",
                        help="Stream only explicitly registered local PTs to verify SHA; never load a model")
    parser.add_argument("--verify-local-artifacts", action="store_true",
                        help="Require private report files and protected source archive refs locally")
    parser.add_argument("--commit", help="Check a Git tree without changing the working directory")
    args = parser.parse_args()
    if args.commit and (args.verify_local_weights or args.verify_local_artifacts):
        parser.error("Git-tree checks do not load private working-directory artifacts")
    root = Path(__file__).resolve().parents[2]
    report = inspect_tree(root, args.commit) if args.commit else inspect(root, args.verify_local_weights,
                                                                      args.verify_local_artifacts)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
