from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from .datasets import TASK_SPECS, Vtab1kClassificationDataset


EXPECTED_ARCHIVE_SHA256 = "dca579dac0ac3d285ec8d898033795f2eb55fe83cbd8d0c10b074455a9d2aa1c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_selected(archive: Path, output_root: Path, tasks: tuple[str, ...]) -> dict:
    archive_digest = sha256_file(archive)
    if archive_digest != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            f"VTAB archive SHA-256 mismatch: {archive_digest}; "
            f"expected {EXPECTED_ARCHIVE_SHA256}"
        )
    selected_directories = {TASK_SPECS[task].directory for task in tasks}
    provenance_path = output_root / "P14_DATA_PROVENANCE.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not provenance_path.is_file():
            raise FileExistsError(
                f"Refusing to overwrite a non-empty unverified data root: {output_root}"
            )
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            existing.get("archive_sha256") != archive_digest
            or existing.get("tasks") != list(tasks)
        ):
            raise RuntimeError(
                f"Existing VTAB provenance does not match this request: {provenance_path}"
            )
        for task in tasks:
            for split in ("train800", "val200", "train800val200", "test"):
                dataset = Vtab1kClassificationDataset(output_root, task, split)  # type: ignore[arg-type]
                missing = [str(path) for path, _ in dataset.samples if not path.is_file()]
                if missing:
                    raise RuntimeError(
                        f"{task}/{split} is incomplete; first missing file: {missing[0]}"
                    )
        return existing
    output_root.mkdir(parents=True, exist_ok=True)
    extracted_files = 0
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            parts = PurePosixPath(member.name).parts
            if "vtab-1k" not in parts:
                continue
            prefix = parts.index("vtab-1k")
            relative_parts = parts[prefix + 1 :]
            if not relative_parts or relative_parts[0] not in selected_directories:
                continue
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"Unsafe non-regular VTAB archive entry: {member.name}")
            destination = output_root.joinpath(*relative_parts).resolve()
            if output_root.resolve() not in destination.parents and destination != output_root.resolve():
                raise RuntimeError(f"VTAB archive entry escapes output root: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read VTAB archive member: {member.name}")
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            extracted_files += 1
    for task in tasks:
        for split in ("train800", "val200", "train800val200", "test"):
            dataset = Vtab1kClassificationDataset(output_root, task, split)  # type: ignore[arg-type]
            missing = [str(path) for path, _ in dataset.samples if not path.is_file()]
            if missing:
                raise RuntimeError(f"{task}/{split} is incomplete; first missing file: {missing[0]}")
    provenance = {
        "format": "p14-vtab1k-data-provenance-v1",
        "source": "https://huggingface.co/datasets/anhth/vtab",
        "revision": "102b19d8a1e23c47fb3941e736ca8b1d49d2552c",
        "archive": str(archive.resolve()),
        "archive_sha256": archive_digest,
        "tasks": list(tasks),
        "dataset_directories": sorted(selected_directories),
        "extracted_files": extracted_files,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the six selected VTAB-1k tasks")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASK_SPECS), default=tuple(TASK_SPECS))
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    result = extract_selected(args.archive.resolve(), args.output_root.resolve(), tuple(args.tasks))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
