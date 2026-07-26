from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PREPARATION_SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def preparation_metadata_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".metadata.json")


def ensure_cc3m_dataset(settings: Any) -> dict[str, Any]:
    """Prepare the configured public CC3M WebDataset snapshot when needed.

    A manually supplied JSONL remains supported. Automatic preparation is only
    attempted when the manifest is absent, or when a previous automatic smoke
    preparation contains fewer shards than the current formal configuration.
    """

    manifest_path = Path(settings.manifest_path)
    metadata_path = preparation_metadata_path(manifest_path)
    if manifest_path.is_file():
        if not metadata_path.is_file():
            return {
                "status": "using_manual_manifest",
                "manifest_path": str(manifest_path),
            }
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_existing_source(metadata, settings)
        prepared = int(metadata.get("prepared_shards", 0))
        requested = settings.dataset_download_max_shards
        enough = (
            bool(metadata.get("complete_source_split"))
            if requested is None
            else prepared >= int(requested)
        )
        if enough:
            return {"status": "already_prepared", **metadata}
    if not settings.dataset_auto_prepare:
        raise FileNotFoundError(
            f"CC3M JSONL manifest does not exist: {manifest_path}. "
            "Set dataset.prepare.auto_if_missing=true or create the manifest manually."
        )
    return prepare_cc3m_webdataset(settings)


def prepare_cc3m_webdataset(settings: Any) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise RuntimeError(
            "Automatic CC3M preparation requires huggingface_hub. "
            "Install it or provide dataset.manifest_path manually."
        ) from error

    endpoint = (
        settings.dataset_source_endpoint
        or os.environ.get("HF_ENDPOINT")
        or "https://huggingface.co"
    )
    api = HfApi(endpoint=endpoint)
    try:
        info = api.dataset_info(
            settings.dataset_source_repo_id,
            revision=settings.dataset_source_revision,
            files_metadata=True,
            timeout=60,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not inspect the configured CC3M WebDataset source "
            f"{settings.dataset_source_repo_id}@{settings.dataset_source_revision} "
            f"through {endpoint}. This snapshot is public and normally needs no "
            f"authorization. Original error: {type(error).__name__}: {error}"
        ) from error
    if getattr(info, "gated", False):
        raise RuntimeError(
            f"Dataset {settings.dataset_source_repo_id} is gated. Request access on "
            f"https://huggingface.co/datasets/{settings.dataset_source_repo_id} "
            "and run `hf auth login` on this server."
        )

    prefix = f"cc3m-{settings.dataset_source_split}-"
    candidates = sorted(
        sibling.rfilename
        for sibling in (info.siblings or [])
        if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith(".tar")
    )
    if not candidates:
        raise RuntimeError(
            f"No {prefix}*.tar shards were found in "
            f"{settings.dataset_source_repo_id}@{settings.dataset_source_revision}"
        )
    available_count = len(candidates)
    requested = settings.dataset_download_max_shards
    selected = candidates if requested is None else candidates[: int(requested)]
    size_by_name = {
        sibling.rfilename: int(getattr(sibling, "size", 0) or 0)
        for sibling in (info.siblings or [])
    }

    archive_dir = Path(settings.dataset_archive_dir)
    image_dir = Path(settings.dataset_image_dir)
    state_dir = Path(settings.manifest_path).parent / ".cc3m_prepare"
    shard_manifest_dir = state_dir / "shard_manifests"
    marker_dir = state_dir / "markers"
    for directory in (archive_dir, image_dir, shard_manifest_dir, marker_dir):
        directory.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    workers = min(int(settings.dataset_download_workers), len(selected))
    print(
        f"[prepare_data] source={settings.dataset_source_repo_id}@{info.sha} "
        f"endpoint={endpoint} split={settings.dataset_source_split} "
        f"shards={len(selected)}/{available_count} workers={workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_name = {
            executor.submit(
                _prepare_remote_shard,
                name,
                size_by_name.get(name, 0),
                settings,
                endpoint,
                str(info.sha),
                archive_dir,
                image_dir,
                shard_manifest_dir,
                marker_dir,
            ): name
            for name in selected
        }
        for number, future in enumerate(as_completed(future_to_name), start=1):
            name = future_to_name[future]
            try:
                result = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"CC3M preparation failed at shard {name}. Completed shards are "
                    "preserved; rerun the same command to resume. "
                    f"Original error: {type(error).__name__}: {error}"
                ) from error
            completed.append(result)
            elapsed = max(time.perf_counter() - started, 1e-6)
            samples = sum(int(item["samples"]) for item in completed)
            print(
                f"[prepare_data] shards={number}/{len(selected)} "
                f"latest={name} latest_samples={result['samples']} "
                f"completed_samples={samples:,} elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    ordered = [
        json.loads((marker_dir / f"{Path(name).stem}.json").read_text(encoding="utf-8"))
        for name in selected
    ]
    manifest_digest, total_samples = _combine_shard_manifests(
        selected,
        shard_manifest_dir,
        Path(settings.manifest_path),
    )
    metadata = {
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "dataset": "cc3m_jsonl",
        "source_repo_id": settings.dataset_source_repo_id,
        "configured_revision": settings.dataset_source_revision,
        "resolved_revision": str(info.sha),
        "source_endpoint": endpoint,
        "source_split": settings.dataset_source_split,
        "public_source_gated": bool(getattr(info, "gated", False)),
        "available_source_shards": available_count,
        "prepared_shards": len(selected),
        "complete_source_split": len(selected) == available_count,
        "source_archive_bytes": sum(size_by_name.get(name, 0) for name in selected),
        "samples": total_samples,
        "manifest_path": str(Path(settings.manifest_path)),
        "manifest_sha256": manifest_digest,
        "image_dir": str(image_dir),
        "archive_dir": str(archive_dir),
        "archives_retained": bool(settings.dataset_keep_archives),
        "download_workers": workers,
        "elapsed_sec": time.perf_counter() - started,
        "shards": ordered,
    }
    metadata_path = preparation_metadata_path(Path(settings.manifest_path))
    _atomic_write_json(metadata_path, metadata)
    print(
        f"[prepare_data] complete manifest={settings.manifest_path} "
        f"samples={total_samples:,} sha256={manifest_digest}",
        flush=True,
    )
    return metadata


def extract_webdataset_shard(
    archive_path: Path,
    output_dir: Path,
    shard_manifest_path: Path,
    *,
    source_split: str,
    manifest_root: Path,
) -> dict[str, Any]:
    """Extract one WebDataset shard without using unsafe tar.extractall()."""

    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    temporary_dir = output_dir.with_name(output_dir.name + ".partial")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    captions: dict[str, str] = {}
    image_members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_path = Path(member.name)
            key = member_path.stem
            suffix = member_path.suffix.lower()
            if suffix == ".txt":
                handle = archive.extractfile(member)
                if handle is not None:
                    captions[key] = handle.read().decode("utf-8", "replace").strip()
            elif suffix in IMAGE_EXTENSIONS:
                image_members[key] = member
        valid_keys = sorted(set(captions) & set(image_members))
        if not valid_keys:
            raise RuntimeError(f"No paired image/text samples found in {archive_path}")
        lines: list[str] = []
        for key in valid_keys:
            caption = captions[key]
            if not caption:
                continue
            member = image_members[key]
            suffix = Path(member.name).suffix.lower()
            target = temporary_dir / f"{key}{suffix}"
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            final_target = output_dir / target.name
            relative_path = os.path.relpath(final_target, manifest_root)
            lines.append(
                json.dumps(
                    {
                        "sample_id": f"{archive_path.stem}:{key}",
                        "image_path": Path(relative_path).as_posix(),
                        "caption": caption,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temporary_dir.replace(output_dir)
    shard_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(shard_manifest_path, "\n".join(lines) + "\n")
    return {
        "archive": archive_path.name,
        "samples": len(lines),
        "images_without_caption": len(set(image_members) - set(captions)),
        "captions_without_image": len(set(captions) - set(image_members)),
    }


def _prepare_remote_shard(
    name: str,
    expected_size: int,
    settings: Any,
    endpoint: str,
    resolved_revision: str,
    archive_dir: Path,
    image_dir: Path,
    shard_manifest_dir: Path,
    marker_dir: Path,
) -> dict[str, Any]:
    marker_path = marker_dir / f"{Path(name).stem}.json"
    shard_manifest_path = shard_manifest_dir / f"{Path(name).stem}.jsonl"
    output_dir = image_dir / settings.dataset_source_split / Path(name).stem
    if marker_path.is_file() and shard_manifest_path.is_file() and output_dir.is_dir():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("source_repo_id") == settings.dataset_source_repo_id
            and marker.get("resolved_revision") == resolved_revision
            and int(marker.get("source_archive_bytes", -1)) == int(expected_size)
        ):
            return marker

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required for CC3M download") from error
    last_error: Exception | None = None
    archive_path: Path | None = None
    for attempt in range(1, 4):
        try:
            archive_path = Path(
                hf_hub_download(
                    repo_id=settings.dataset_source_repo_id,
                    filename=name,
                    repo_type="dataset",
                    revision=settings.dataset_source_revision,
                    local_dir=archive_dir,
                    endpoint=endpoint,
                )
            )
            break
        except Exception as error:  # pragma: no cover - network-dependent retry
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    if archive_path is None:
        assert last_error is not None
        raise last_error
    actual_size = archive_path.stat().st_size
    if expected_size and actual_size != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {name}: {actual_size} != {expected_size}"
        )
    result = extract_webdataset_shard(
        archive_path,
        output_dir,
        shard_manifest_path,
        source_split=settings.dataset_source_split,
        manifest_root=Path(settings.manifest_path).parent,
    )
    result.update(
        {
            "source_repo_id": settings.dataset_source_repo_id,
            "resolved_revision": resolved_revision,
            "source_archive_bytes": actual_size,
        }
    )
    _atomic_write_json(marker_path, result)
    if not settings.dataset_keep_archives:
        archive_path.unlink(missing_ok=True)
    return result


def _combine_shard_manifests(
    shard_names: list[str],
    shard_manifest_dir: Path,
    output_path: Path,
) -> tuple[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    digest = hashlib.sha256()
    samples = 0
    with temporary.open("wb") as destination:
        for name in shard_names:
            source_path = shard_manifest_dir / f"{Path(name).stem}.jsonl"
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing prepared shard manifest: {source_path}")
            with source_path.open("rb") as source:
                while True:
                    block = source.read(8 * 1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    digest.update(block)
                    samples += block.count(b"\n")
    temporary.replace(output_path)
    return digest.hexdigest(), samples


def _validate_existing_source(metadata: dict[str, Any], settings: Any) -> None:
    expected = {
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "source_repo_id": settings.dataset_source_repo_id,
        "configured_revision": settings.dataset_source_revision,
        "source_split": settings.dataset_source_split,
    }
    changed = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if changed:
        raise RuntimeError(
            "Existing automatically prepared CC3M manifest has incompatible source "
            f"metadata fields {changed}. Use a separate manifest_path/data directory "
            "or remove the old generated dataset before changing its source."
        )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
    )
