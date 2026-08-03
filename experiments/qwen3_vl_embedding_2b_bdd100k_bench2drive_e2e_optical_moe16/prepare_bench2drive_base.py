from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_REPO_ID = "rethinklab/Bench2Drive"
KEEP_PARTS = ("anno", "camera/rgb_front")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official Bench2Drive Base archives one at a time and "
            "retain only front RGB frames and expert annotations."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/bench2drive/base"))
    parser.add_argument(
        "--archive-dir", type=Path, default=Path("data/bench2drive/_downloads/base")
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--archive-list",
        type=Path,
        default=None,
        help=(
            "Official bench2drive_base_1000.json. Using it avoids a paginated "
            "Hub API listing, which some mirrors redirect to huggingface.co."
        ),
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent archive download/extraction workers (recommended: 4).",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def list_archives(
    repo_id: str,
    *,
    revision: str,
    endpoint: str | None,
    token: str | None,
) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install this experiment's requirements first."
        ) from exc
    api = HfApi(endpoint=endpoint, token=token)
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    archives = sorted(
        name for name in files if name.lower().endswith((".tar.gz", ".tgz"))
    )
    if not archives:
        raise RuntimeError(
            f"No .tar.gz clips were found in dataset {repo_id}@{revision}."
        )
    return archives


def archives_from_official_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Bench2Drive archive manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Bench2Drive archive manifest must be a JSON object: {path}")
    archives = sorted(
        str(name)
        for name in payload
        if str(name).lower().endswith((".tar.gz", ".tgz"))
    )
    if not archives:
        raise RuntimeError(f"No tar archives are listed in {path}")
    return archives


def download_archive(
    repo_id: str,
    filename: str,
    *,
    revision: str,
    endpoint: str | None,
    token: str | None,
    archive_dir: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required") from exc
    archive_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        endpoint=endpoint,
        token=token,
        local_dir=archive_dir,
    )
    return Path(path)


def extract_front_rgb_and_annotations(archive_path: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = 0
    images = 0
    annotations = 0
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or not _keep_member(member.name):
                continue
            relative = _safe_relative_path(member.name)
            target = output_dir.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {member.name} from {archive_path}")
            temporary = target.with_name(target.name + ".partial")
            with source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            temporary.replace(target)
            selected += 1
            posix = relative.as_posix()
            if "/camera/rgb_front/" in f"/{posix}":
                images += 1
            elif "/anno/" in f"/{posix}":
                annotations += 1
    if images == 0 or annotations == 0:
        raise RuntimeError(
            f"Archive {archive_path} did not contain both camera/rgb_front and anno "
            f"files (images={images}, annotations={annotations})."
        )
    return {"selected_files": selected, "rgb_front_images": images, "annotations": annotations}


def _keep_member(name: str) -> bool:
    normalized = "/" + name.replace("\\", "/").strip("/") + "/"
    return "/anno/" in normalized or "/camera/rgb_front/" in normalized


def _safe_relative_path(name: str) -> PurePosixPath:
    value = PurePosixPath(name.replace("\\", "/"))
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise RuntimeError(f"Unsafe archive member path: {name!r}")
    return value


def marker_path(output_dir: Path, archive_name: str) -> Path:
    digest = hashlib.sha256(archive_name.encode("utf-8")).hexdigest()[:20]
    return output_dir / ".completed_archives" / f"{digest}.json"


def staging_path(output_dir: Path, archive_name: str) -> Path:
    digest = hashlib.sha256(archive_name.encode("utf-8")).hexdigest()[:20]
    return output_dir / ".extracting" / digest


def commit_staging(staging: Path, output_dir: Path) -> None:
    children = [path for path in staging.iterdir() if path.name not in {".", ".."}]
    if not children:
        raise RuntimeError(f"No extracted route directory was produced in {staging}")
    for source in children:
        target = output_dir / source.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        source.replace(target)
    staging.rmdir()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_archives(output_dir: Path, archives: Iterable[str]) -> int:
    return sum(marker_path(output_dir, name).is_file() for name in archives)


def main() -> int:
    args = parse_args()
    if args.max_clips is not None and args.max_clips <= 0:
        raise ValueError("--max-clips must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.archive_list is not None:
        archives = archives_from_official_manifest(args.archive_list)
    else:
        archives = list_archives(
            args.repo_id,
            revision=args.revision,
            endpoint=args.endpoint,
            token=args.token,
        )
    if args.max_clips is not None:
        archives = archives[: args.max_clips]
    done = completed_archives(args.output_dir, archives)
    print(
        f"Bench2Drive archives selected={len(archives)} completed={done} "
        f"remaining={len(archives) - done}"
    )
    if args.dry_run:
        for name in archives[:10]:
            print(name)
        return 0

    remaining = [
        (index, filename)
        for index, filename in enumerate(archives, start=1)
        if not marker_path(args.output_dir, filename).is_file()
    ]

    def process_one(item: tuple[int, str]) -> str:
        index, filename = item
        marker = marker_path(args.output_dir, filename)
        print(f"[bench2drive_download] {index}/{len(archives)} downloading {filename}", flush=True)
        archive_path = download_archive(
            args.repo_id,
            filename,
            revision=args.revision,
            endpoint=args.endpoint,
            token=args.token,
            archive_dir=args.archive_dir,
        )
        staging = staging_path(args.output_dir, filename)
        if staging.exists():
            shutil.rmtree(staging)
        print(f"[bench2drive_download] extracting front RGB + anno from {archive_path}", flush=True)
        counts = extract_front_rgb_and_annotations(archive_path, staging)
        commit_staging(staging, args.output_dir)
        write_json(
            marker,
            {
                "repo_id": args.repo_id,
                "revision": args.revision,
                "archive": filename,
                **counts,
            },
        )
        if not args.keep_archives:
            archive_path.unlink(missing_ok=True)
        print(
            f"[bench2drive_download] completed {filename}: "
            f"images={counts['rgb_front_images']} annotations={counts['annotations']}",
            flush=True,
        )
        return filename

    if args.workers == 1:
        for item in remaining:
            process_one(item)
    else:
        print(
            f"[bench2drive_download] running {args.workers} concurrent workers; "
            "each completed archive is independently resumable",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, item) for item in remaining]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    write_json(
        args.output_dir / "download_manifest.json",
        {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "selected_archives": len(archives),
            "completed_archives": completed_archives(args.output_dir, archives),
            "retained_modalities": list(KEEP_PARTS),
            "archives_retained": bool(args.keep_archives),
            "workers": args.workers,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
