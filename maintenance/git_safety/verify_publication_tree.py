"""Verify source provenance in a candidate Git tree, without checking it out."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def read(root, ref, path):
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe source identity path")
    return subprocess.check_output(["git", "-C", str(root), "show", f"{ref}:{path}"])


def verify(root, ref, manifests):
    errors, checked = [], []
    for name in manifests:
        manifest = json.loads(read(root, ref, name))
        for row in [*manifest["files"], *manifest.get("additional_dependency_hashes", [])]:
            payload = read(root, ref, row["path"])
            normalization = row.get("hash_normalization")
            if normalization == "crlf_to_lf":
                payload = payload.replace(b"\r\n", b"\n")
            elif normalization is not None:
                errors.append("unknown hash normalization: " + row["path"])
            if hashlib.sha256(payload).hexdigest() != row["source_blob_sha256"]:
                errors.append("source provenance mismatch: " + row["path"])
            checked.append(row["path"])
    teacher_root = "LightGenV2/tasks/t13_temporal_robust_training/"
    teacher = json.loads(read(root, ref, teacher_root + "reference/source_manifest.json"))
    for path, digest in teacher["files"].items():
        if hashlib.sha256(read(root, ref, teacher_root + path)).hexdigest() != digest:
            errors.append("teacher runtime mismatch: " + path)
    return {"commit": ref, "source_hashes_checked": len(checked),
            "teacher_snapshot_hashes_checked": len(teacher["files"]),
            "errors": errors, "read_only": True, "runtime_tests_rerun": False}


def verify_reviewed_publications(root, ref, manifests):
    """Enforce already-reviewed T03/T10 imports as well as T13/T16 snapshots."""
    checked, errors = [], []
    evolution_path = 'maintenance/storage/GOVERNANCE_TOOL_EVOLUTION_20261004.json'
    present = subprocess.check_output(['git','-C',str(root),'ls-tree',ref,'--',evolution_path])
    evolutions = (json.loads(read(root,ref,evolution_path))['paths'] if present else [])
    evolved = {row['path']:row for row in evolutions}
    for path,row in evolved.items():
        assert path.startswith('maintenance/git_safety/') and row['review_reason']
        assert hashlib.sha256(read(root,row['historical_source_commit'],path)).hexdigest() == row['historical_sha256']
        assert hashlib.sha256(read(root,row['source_commit'],path)).hexdigest() == row['sha256']
    for name in manifests:
        manifest = json.loads(read(root, ref, name))
        for row in manifest["paths"]:
            path = row.get("published_path", row.get("path"))
            expected = (row.get("published_sha256") if "published_path" in row
                        else row.get("source_sha256", row.get("sha256")))
            if not expected:
                raise ValueError("Missing published identity: " + str(path))
            if path in evolved:
                if expected != evolved[path]['historical_sha256']:
                    raise ValueError('Unreviewed governance history: '+path)
                expected = evolved[path]['sha256']
            actual = hashlib.sha256(read(root, ref, path)).hexdigest()
            if actual != expected:
                errors.append("reviewed publication mismatch: " + path)
            checked.append(path)
    return {"reviewed_publication_hashes_checked": len(checked), "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = verify(root, args.commit, [
        "LightGenV2/tasks/t16_zero_phase_ccd_lifelong/source_import_20261002.json",
        "LightGenV2/tasks/t11_lifelong_optics/source_import_20261004.json",
        "LightGenV2/tasks/t06_video_quality_assessment/source_import_20261004.json",
        "LightGenV2/tasks/t06_video_quality_assessment/adaptation_source_import_20261004.json",
        "LightGenV2/tasks/t09_multimodal_matching/source_import_20261004.json",
        "LightGenV2/tasks/t13_temporal_robust_training/source_import_20261003.json"])
    reviewed = verify_reviewed_publications(root, args.commit, [
        "maintenance/storage/T03_REVIEWED_CORE_20261003.json",
        "maintenance/storage/T03_REVIEWED_ENTRY_20261003.json",
        "maintenance/storage/T03_PINNED_ADDITIONS_20261003.json",
        "maintenance/storage/T03_ENTRY_ADDITIONS_20261003.json",
        "maintenance/storage/T10_RUNTIME_ADDITIONS_20261003.json",
        "maintenance/storage/T08_PHYSICAL_TOOLS_IMPORT_20261004.json",
        "maintenance/storage/T12_PRESERVED_ENTRY_ADDITIONS_20261004.json",
        "maintenance/storage/T12_PRESERVED_ENTRY_DEPENDENCIES_20261004.json",
        "maintenance/storage/T07_RANK72_SOURCE_ADDITIONS_20261004.json",
        "maintenance/storage/T07_REVIEWED_MAIN_ENTRY_20261004.json"])
    report["reviewed_publication_hashes_checked"] = reviewed["reviewed_publication_hashes_checked"]
    report["errors"].extend(reviewed["errors"])
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
