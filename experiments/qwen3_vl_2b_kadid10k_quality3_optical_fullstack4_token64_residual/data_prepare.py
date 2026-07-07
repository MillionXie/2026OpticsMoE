from __future__ import annotations

import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_KADID10K_DOWNLOAD_URL=(
    "https://files.osf.io/v1/resources/xkqjh/providers/osfstorage/"
    "5eafe5bf0ffc0500ec6f6c94/?zip="
)


def ensure_kadid10k_dataset(root:Path,metadata_csv:str,image_dir:str,download_url:str|None=None)->dict[str,Any]:
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    located=_locate_dataset(root,metadata_csv,image_dir)
    if located is not None:return {**located,"prepared":False,"source":"existing"}
    url=str(download_url or DEFAULT_KADID10K_DOWNLOAD_URL).strip()
    if not url:raise RuntimeError("KADID-10k is missing and dataset_download_url is empty")
    downloads=root/"_downloads";raw=root/"_raw";downloads.mkdir(parents=True,exist_ok=True);raw.mkdir(parents=True,exist_ok=True)
    archive=downloads/"KADID-10k.zip";marker=raw/".kadid10k_extracted"
    _download_with_resume(url,archive)
    if not marker.is_file():
        print(f"[dataset] validating and extracting {archive}",flush=True)
        _extract_safely(archive,raw);marker.write_text("ok\n",encoding="utf-8")
    located=_locate_dataset(root,metadata_csv,image_dir)
    if located is None:
        raise FileNotFoundError(
            f"KADID-10k archive was extracted under {raw}, but dmos.csv and an image/images folder "
            "could not be located. Inspect the extracted archive structure."
        )
    return {**located,"prepared":True,"source":url,"archive":str(archive)}


def _locate_dataset(root:Path,metadata_csv:str,image_dir:str)->dict[str,str]|None:
    configured_csv=Path(metadata_csv).expanduser();configured_csv=configured_csv if configured_csv.is_absolute() else root/configured_csv
    configured_images=Path(image_dir).expanduser();configured_images=configured_images if configured_images.is_absolute() else root/configured_images
    if configured_csv.is_file() and _contains_images(configured_images):
        return {"metadata_csv":str(configured_csv.resolve()),"image_dir":str(configured_images.resolve())}
    csv_name=Path(metadata_csv).name or "dmos.csv"
    csv_candidates=[path for path in root.rglob(csv_name) if path.is_file() and "_downloads" not in path.parts]
    if csv_name.lower()!="dmos.csv":csv_candidates.extend(path for path in root.rglob("dmos.csv") if path.is_file() and "_downloads" not in path.parts)
    image_names={Path(image_dir).name.lower(),"image","images"}
    image_candidates=[path for path in root.rglob("*") if path.is_dir() and path.name.lower() in image_names and _contains_images(path)]
    if not csv_candidates or not image_candidates:return None
    csv_path=sorted(csv_candidates,key=lambda path:(len(path.parts),str(path)))[0]
    image_path=sorted(image_candidates,key=lambda path:(0 if path.parent==csv_path.parent else 1,len(path.parts),str(path)))[0]
    return {"metadata_csv":str(csv_path.resolve()),"image_dir":str(image_path.resolve())}


def _contains_images(path:Path)->bool:
    if not path.is_dir():return False
    return any(path.glob("*.png")) or any(path.glob("*.jpg")) or any(path.glob("*.jpeg"))


def _download_with_resume(url:str,destination:Path)->None:
    if destination.is_file() and _valid_zip(destination):return
    partial=destination.with_suffix(".zip.part");offset=partial.stat().st_size if partial.is_file() else 0
    headers={"User-Agent":"2026OpticsMoE-KADID10k/1.0"}
    if offset:headers["Range"]=f"bytes={offset}-"
    print(f"[dataset] downloading KADID-10k from {url}",flush=True)
    try:response=urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"KADID-10k download failed: {exc.reason}. You may download it manually and place "
            f"the archive at {destination}, then rerun prepare_data."
        ) from exc
    status=getattr(response,"status",None);append=offset>0 and status==206
    if offset and not append:print("[dataset] server does not support resume for this generated ZIP; restarting download",flush=True)
    downloaded=offset if append else 0;next_report=((downloaded//(256*1024**2))+1)*(256*1024**2)
    with response,partial.open("ab" if append else "wb") as handle:
        while True:
            chunk=response.read(8*1024**2)
            if not chunk:break
            handle.write(chunk);downloaded+=len(chunk)
            if downloaded>=next_report:
                print(f"[dataset] downloaded {downloaded/1024**3:.2f} GiB",flush=True);next_report+=256*1024**2
    partial.replace(destination)
    if not _valid_zip(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded KADID-10k archive is not a valid ZIP: {destination}. Rerun prepare_data.")


def _valid_zip(path:Path)->bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names=[name.lower() for name in archive.namelist()]
            return any(name.endswith("dmos.csv") for name in names) and any(".png" in name for name in names)
    except (OSError,zipfile.BadZipFile):return False


def _extract_safely(archive_path:Path,destination:Path)->None:
    base=destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target=(destination/member.filename).resolve()
            if target!=base and base not in target.parents:raise RuntimeError(f"Unsafe KADID archive path: {member.filename}")
        archive.extractall(destination)


__all__=["DEFAULT_KADID10K_DOWNLOAD_URL","ensure_kadid10k_dataset"]
