from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

from .settings import Settings


TABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_spaq_dataset(settings: Settings) -> dict[str, Any]:
    """Download and extract SPAQ when the configured root is not already prepared."""

    ready = inspect_spaq_root(settings.data_root)
    if ready["has_annotations"] and ready["image_count"] > 0:
        return {"action": "reuse", **ready}
    if not settings.download:
        raise FileNotFoundError(
            f"SPAQ is not prepared under {settings.data_root} and download=false. "
            "Set download=true or point data_root to an existing SPAQ directory. "
            f"Current inspection: {ready}"
        )
    settings.data_root.mkdir(parents=True, exist_ok=True)
    downloads = settings.data_root / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    print(
        f"[SPAQ] dataset missing; downloading via {settings.download_source} "
        f"into {settings.data_root}",
        flush=True,
    )
    if settings.download_source == "huggingface":
        archive = _download_huggingface(settings, downloads)
    else:
        archive = _download_google_drive(settings, downloads)
    extracted = _extract_downloads(settings.data_root, downloads)
    ready = inspect_spaq_root(settings.data_root)
    if not ready["has_annotations"] or ready["image_count"] == 0:
        raise RuntimeError(
            "SPAQ download completed but a usable annotation table and image files were not found. "
            f"Downloaded artifact: {archive}. Extracted archives: {extracted}. Inspection: {ready}"
        )
    if not settings.keep_download_archive:
        _remove_download_artifacts(downloads)
    return {
        "action": "download",
        "source": settings.download_source,
        "archive": str(archive),
        "extracted_archives": extracted,
        **ready,
    }


def inspect_spaq_root(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {
            "data_root": str(root),
            "exists": False,
            "has_annotations": False,
            "annotation_files": [],
            "image_count": 0,
        }
    tables = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES
    )
    image_count = sum(
        1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return {
        "data_root": str(root),
        "exists": True,
        "has_annotations": bool(tables),
        "annotation_files": [str(path) for path in tables],
        "image_count": image_count,
    }


def _download_huggingface(settings: Settings, downloads: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic SPAQ download. "
            "Install this experiment's requirements.txt."
        ) from exc
    print(
        f"[SPAQ] Hugging Face dataset={settings.download_repo_id} "
        f"file={settings.download_filename}. This is a large download and is resumable.",
        flush=True,
    )
    path = hf_hub_download(
        repo_id=settings.download_repo_id,
        repo_type="dataset",
        filename=settings.download_filename,
        local_dir=str(downloads),
        cache_dir=str(settings.cache_dir) if settings.cache_dir else None,
        endpoint=settings.download_endpoint,
    )
    return Path(path).resolve()


def _download_google_drive(settings: Settings, downloads: Path) -> Path:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for Google Drive SPAQ download. "
            "Install this experiment's requirements.txt."
        ) from exc
    url = str(settings.download_url)
    if "/folders/" in url:
        outputs = gdown.download_folder(
            url=url,
            output=str(downloads),
            quiet=False,
            remaining_ok=True,
            use_cookies=False,
        )
        if not outputs:
            raise RuntimeError(f"Google Drive folder download returned no files: {url}")
        return downloads
    output = downloads / (Path(settings.download_filename).name or "spaq_download")
    result = gdown.download(url=url, output=str(output), quiet=False, resume=True, fuzzy=True)
    if result is None:
        raise RuntimeError(f"Google Drive download failed: {url}")
    return Path(result).resolve()


def _extract_downloads(data_root: Path, downloads: Path) -> list[str]:
    archives = sorted(
        path
        for path in downloads.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".zip", ".tgz", ".tar"}
            or path.name.lower().endswith((".tar.gz", ".tar.bz2", ".tar.xz"))
        )
    )
    extracted: list[str] = []
    for archive in archives:
        print(f"[SPAQ] extracting {archive}", flush=True)
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as handle:
                _safe_zip_extract(handle, data_root)
        else:
            with tarfile.open(archive, mode="r:*") as handle:
                _safe_tar_extract(handle, data_root)
        extracted.append(str(archive))
    return extracted


def _safe_zip_extract(handle: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in handle.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"Unsafe path in SPAQ zip archive: {member.filename}")
    handle.extractall(destination)


def _safe_tar_extract(handle: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in handle.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"Unsafe path in SPAQ tar archive: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Links are not allowed in SPAQ archive: {member.name}")
    handle.extractall(destination)


def _remove_download_artifacts(downloads: Path) -> None:
    if downloads.is_dir():
        shutil.rmtree(downloads)
