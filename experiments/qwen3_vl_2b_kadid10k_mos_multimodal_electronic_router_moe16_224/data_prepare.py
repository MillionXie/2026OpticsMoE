from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .settings import Settings


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_kadid10k_dataset(settings: Settings) -> dict[str, Any]:
    """Reuse or automatically download and safely extract official KADID-10k."""

    ready = inspect_kadid10k_root(settings.data_root)
    if ready["has_dmos_csv"] and ready["distorted_image_count"] > 0:
        return {"action": "reuse", **ready}
    if not settings.download:
        raise FileNotFoundError(
            f"KADID-10k is not prepared under {settings.data_root} and download=false. "
            "Set download=true or point data_root to an extracted KADID-10k directory. "
            f"Current inspection: {ready}"
        )
    settings.data_root.mkdir(parents=True, exist_ok=True)
    downloads = settings.data_root / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / settings.download_filename
    print(
        "[KADID-10k] dataset missing; downloading the official approximately "
        f"3.1 GB archive from {settings.download_url}",
        flush=True,
    )
    _download_resumable(str(settings.download_url), archive)
    print(f"[KADID-10k] extracting {archive} into {settings.data_root}", flush=True)
    _safe_extract_zip(archive, settings.data_root)
    ready = inspect_kadid10k_root(settings.data_root)
    if not ready["has_dmos_csv"] or ready["distorted_image_count"] == 0:
        raise RuntimeError(
            "KADID-10k download/extraction completed but dmos.csv and distorted PNG "
            f"images were not both found. Inspection: {ready}"
        )
    if settings.require_official_counts and (
        ready["distorted_image_count"] < settings.official_distorted_images
    ):
        raise RuntimeError(
            "KADID-10k extraction appears incomplete: "
            f"found {ready['distorted_image_count']} distorted images, expected at least "
            f"{settings.official_distorted_images}."
        )
    if not settings.keep_download_archive:
        shutil.rmtree(downloads)
    return {
        "action": "download",
        "source": "official_url",
        "download_url": str(settings.download_url),
        "archive": str(archive),
        **ready,
    }


def inspect_kadid10k_root(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        return {
            "data_root": str(root),
            "exists": False,
            "has_dmos_csv": False,
            "metadata_files": [],
            "image_roots": [],
            "image_count": 0,
            "distorted_image_count": 0,
        }
    metadata = sorted(path for path in root.rglob("*.csv") if path.name.lower() == "dmos.csv")
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    distorted = [
        path
        for path in images
        if len(path.stem.split("_")) == 3 and path.stem.lower().startswith("i")
    ]
    image_roots = sorted({str(path.parent) for path in images})
    return {
        "data_root": str(root),
        "exists": True,
        "has_dmos_csv": bool(metadata),
        "metadata_files": [str(path) for path in metadata],
        "image_roots": image_roots,
        "image_count": len(images),
        "distorted_image_count": len(distorted),
    }


def _download_resumable(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(
            f"[KADID-10k] reusing downloaded archive {destination} "
            f"({destination.stat().st_size / 2**30:.2f} GiB)",
            flush=True,
        )
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "2026OpticsMoE-KADID10k/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
        print(
            f"[KADID-10k] resuming partial archive at {downloaded / 2**30:.2f} GiB",
            flush=True,
        )
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"KADID-10k download failed with HTTP {exc.code} for {url}"
        ) from exc
    status = getattr(response, "status", response.getcode())
    if downloaded and status != 206:
        print(
            "[KADID-10k] server did not honor Range; restarting the archive download",
            flush=True,
        )
        downloaded = 0
    mode = "ab" if downloaded and status == 206 else "wb"
    remaining = int(response.headers.get("Content-Length", "0") or 0)
    expected = downloaded + remaining if remaining else None
    next_report = downloaded + 256 * 2**20
    partial.parent.mkdir(parents=True, exist_ok=True)
    with response, partial.open(mode) as handle:
        total = downloaded
        while True:
            chunk = response.read(8 * 2**20)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
            if total >= next_report:
                suffix = f"/{expected / 2**30:.2f}" if expected else ""
                print(
                    f"[KADID-10k] downloaded {total / 2**30:.2f}{suffix} GiB",
                    flush=True,
                )
                next_report = total + 256 * 2**20
    if expected is not None and partial.stat().st_size != expected:
        raise RuntimeError(
            "KADID-10k download ended at an unexpected size: "
            f"{partial.stat().st_size} bytes, expected {expected} bytes. "
            "The .part file is retained for resume."
        )
    partial.replace(destination)


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(f"Downloaded KADID-10k artifact is not a valid ZIP: {archive}")
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe path in KADID-10k ZIP: {member.filename}")
        handle.extractall(destination)
