from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .datasets import _as_rgb_image, _restore_env, _stable_hash
from .io_utils import write_json


GENERIC_MANIFEST_SCHEMA_VERSION = 1
IMAGE_COLUMNS = ("image", "img")
CAPTION_COLUMNS = ("caption", "captions", "sentences", "answer", "answers")
ID_COLUMNS = ("image_id", "img_id", "imgid", "id", "cocoid")
FILENAME_COLUMNS = ("filename", "file_name", "image_name")
SPLIT_COLUMNS = ("split", "image_split")


@dataclass(frozen=True)
class GenericCaptionRecord:
    image_id: str
    filename: str
    caption: str
    caption_index: int
    source_split: str
    source_index: int
    local_path: str | None = None


@dataclass
class GenericDatasetBundle:
    train: "GenericCaptionDataset"
    metadata: dict[str, Any]
    manifest_digest: str
    dataset_fingerprint: str


class GenericCaptionDataset(Dataset[Any]):
    def __init__(self, records: Sequence[GenericCaptionRecord], sources: Mapping[str, Any],
                 image_columns: Mapping[str, str], prompt_template: str) -> None:
        self.records = list(records); self.sources = dict(sources)
        self.image_columns = dict(image_columns); self.prompt_template = prompt_template
        # Generic hidden-state distillation has no downstream class target.
        self.targets = [0.0] * len(self.records)

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int) -> tuple[Image.Image, str, float]:
        record = self.records[index]
        if record.local_path is not None:
            image = _as_rgb_image(record.local_path)
        else:
            source = self.sources[record.source_split]
            image = _as_rgb_image(source[record.source_index][self.image_columns[record.source_split]])
        return image, self.prompt_template.format(caption=record.caption), 0.0

    def sample_metadata(self, index: int) -> dict[str, Any]: return asdict(self.records[index])


def load_generic_coco(settings: Any, persist_manifest: bool = True,
                      raw_dataset: Any | None = None) -> GenericDatasetBundle:
    if raw_dataset is not None:
        records, sources, image_columns, fingerprint = _scan_huggingface(raw_dataset, settings)
    elif settings.generic_dataset_source == "local_coco2017":
        records, sources, image_columns, fingerprint = _load_local(settings)
    else:
        loaded = _load_huggingface(settings)
        records, sources, image_columns, fingerprint = _scan_huggingface(loaded, settings)
    records.sort(key=lambda row: (row.image_id, row.filename))
    if settings.generic_image_limit is not None:
        records = records[: int(settings.generic_image_limit)]
    if not records:
        raise RuntimeError("Generic COCO pretraining selection is empty")
    if len({row.image_id for row in records}) != len(records):
        raise RuntimeError("Generic pretraining manifest must contain one deterministic caption per image")
    root = settings.output_dir / "generic_pretrain" / "manifests"
    records, digest = _persist_or_validate(root, records, settings, fingerprint) if persist_manifest else (records, _digest(records))
    settings.generic_manifest_digest = digest; settings.generic_dataset_fingerprint = fingerprint
    dataset = GenericCaptionDataset(records, sources, image_columns, settings.generic_prompt_template)
    metadata = {
        "dataset": settings.generic_dataset, "source": settings.generic_dataset_source,
        "repo_id": settings.generic_dataset_repo_id, "revision": settings.generic_dataset_revision,
        "sample_count": len(dataset), "unique_images": len(dataset), "captions_per_image": 1,
        "caption_selection": "stable_sha256_one_caption_per_image_v1",
        "split_values": list(settings.generic_split_values), "prompt_template": settings.generic_prompt_template,
        "manifest_sha256": digest, "dataset_fingerprint": fingerprint,
        "teacher_fine_tuned": False, "downstream_labels_used": False,
    }
    return GenericDatasetBundle(dataset, metadata, digest, fingerprint)


def _scan_huggingface(loaded: Any, settings: Any) -> tuple[list[GenericCaptionRecord], dict[str, Any], dict[str, str], str]:
    sources = _source_mapping(loaded); records: list[GenericCaptionRecord] = []
    image_columns: dict[str, str] = {}; fingerprints: list[str] = []
    accepted = {str(value).strip().lower() for value in settings.generic_split_values}
    for source_name, source in sources.items():
        columns = list(getattr(source, "column_names", [])) or (list(source[0].keys()) if len(source) else [])
        image_col = _find(columns, IMAGE_COLUMNS, "image")
        caption_col = _find(columns, CAPTION_COLUMNS, "caption")
        id_col = _find(columns, ID_COLUMNS, "image id", required=False)
        filename_col = _find(columns, FILENAME_COLUMNS, "filename", required=False)
        split_col = _find(columns, SPLIT_COLUMNS, "internal split", required=False)
        image_columns[source_name] = image_col
        fingerprints.append(f"{source_name}:{getattr(source, '_fingerprint', 'unknown')}")
        metadata_cols = [caption_col] + [value for value in (id_col, filename_col, split_col) if value]
        metadata_source = source.select_columns(metadata_cols) if hasattr(source, "select_columns") else source
        for index in range(len(metadata_source)):
            row = metadata_source[index]
            split_value = str(row[split_col]).strip().lower() if split_col else source_name.lower()
            if accepted and split_value not in accepted and source_name.lower() not in accepted:
                continue
            image_id = str(row[id_col]) if id_col and row.get(id_col) is not None else f"{source_name}:{index}"
            captions = _captions(row[caption_col], image_id)
            selected = _stable_hash(settings.seed, image_id, "generic-caption") % len(captions)
            filename = str(row[filename_col]) if filename_col and row.get(filename_col) is not None else image_id
            records.append(GenericCaptionRecord(image_id, filename, captions[selected], selected, source_name, index))
    fingerprint = hashlib.sha256("|".join(sorted(fingerprints)).encode()).hexdigest()
    return records, sources, image_columns, fingerprint


def _load_local(settings: Any) -> tuple[list[GenericCaptionRecord], dict[str, Any], dict[str, str], str]:
    annotation = settings.generic_data_root / settings.generic_annotations_file
    image_root = settings.generic_data_root / settings.generic_images_dir
    if not annotation.is_file() or not image_root.is_dir():
        raise FileNotFoundError(
            f"Local COCO is incomplete. Expected annotations={annotation} and images={image_root}. "
            "Set generic_pretrain.dataset.source=huggingface for automatic Hugging Face loading."
        )
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    images = {str(row["id"]): str(row["file_name"]) for row in payload.get("images", [])}
    captions: dict[str, list[str]] = {key: [] for key in images}
    for row in payload.get("annotations", []):
        key = str(row["image_id"]); value = str(row.get("caption", "")).strip()
        if key in captions and value: captions[key].append(value)
    records: list[GenericCaptionRecord] = []
    for image_id, filename in images.items():
        values = captions.get(image_id, [])
        if not values: continue
        selected = _stable_hash(settings.seed, image_id, "generic-caption") % len(values)
        path = image_root / filename
        if not path.is_file(): raise FileNotFoundError(f"COCO image listed in annotations is missing: {path}")
        records.append(GenericCaptionRecord(image_id, filename, values[selected], selected, "local", len(records), str(path)))
    fingerprint = _sha256_file(annotation)
    return records, {}, {}, fingerprint


def _load_huggingface(settings: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for generic COCO pretraining") from exc
    settings.generic_data_root.mkdir(parents=True, exist_ok=True)
    previous_endpoint = os.environ.get("HF_ENDPOINT"); previous_offline = os.environ.get("HF_DATASETS_OFFLINE")
    try:
        if settings.hf_endpoint: os.environ["HF_ENDPOINT"] = str(settings.hf_endpoint)
        if not settings.download or settings.local_files_only: os.environ["HF_DATASETS_OFFLINE"] = "1"
        kwargs: dict[str, Any] = {"cache_dir": str(settings.generic_data_root), "trust_remote_code": True}
        if settings.generic_dataset_revision: kwargs["revision"] = settings.generic_dataset_revision
        return load_dataset(settings.generic_dataset_repo_id, **kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load generic dataset {settings.generic_dataset_repo_id} at {settings.generic_data_root}. "
            "Use a reachable HF_ENDPOINT/cache or configure local_coco2017. No fallback dataset is used. "
            f"Original error: {exc}"
        ) from exc
    finally:
        _restore_env("HF_ENDPOINT", previous_endpoint); _restore_env("HF_DATASETS_OFFLINE", previous_offline)


def _source_mapping(loaded: Any) -> dict[str, Any]:
    columns = getattr(loaded, "column_names", None)
    if isinstance(loaded, Mapping) or isinstance(columns, Mapping):
        return {str(name): loaded[name] for name in loaded.keys()}
    return {"train": loaded}


def _captions(value: Any, image_id: str) -> tuple[str, ...]:
    if isinstance(value, str): values = [value]
    elif isinstance(value, Mapping) and "raw" in value:
        raw = value["raw"]; values = [raw] if isinstance(raw, str) else list(raw)
    else:
        try: values = list(value)
        except TypeError as exc: raise RuntimeError(f"Captions for image {image_id} are not iterable") from exc
    expanded = []
    for item in values:
        text = item.get("raw") if isinstance(item, Mapping) and "raw" in item else item
        if str(text).strip(): expanded.append(str(text).strip())
    if not expanded: raise RuntimeError(f"Image {image_id} has no usable caption")
    return tuple(expanded)


def _find(columns: Sequence[str], candidates: Sequence[str], purpose: str, required: bool = True) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup: return lookup[candidate.lower()]
    if required: raise RuntimeError(f"Generic dataset lacks {purpose}; accepted={list(candidates)}, found={list(columns)}")
    return None


def _persist_or_validate(root: Path, records: list[GenericCaptionRecord], settings: Any,
                         fingerprint: str) -> tuple[list[GenericCaptionRecord], str]:
    root.mkdir(parents=True, exist_ok=True); path = root / "train.jsonl"; meta_path = root / "train_metadata.json"
    digest = _digest(records)
    identity = {"schema_version": GENERIC_MANIFEST_SCHEMA_VERSION, "dataset": settings.generic_dataset,
                "source": settings.generic_dataset_source, "repo_id": settings.generic_dataset_repo_id,
                "revision": settings.generic_dataset_revision, "dataset_fingerprint": fingerprint,
                "seed": settings.seed, "prompt_template": settings.generic_prompt_template,
                "split_values": list(settings.generic_split_values), "sample_count": len(records), "sha256": digest}
    if path.is_file() or meta_path.is_file():
        if not path.is_file() or not meta_path.is_file(): raise RuntimeError(f"Incomplete generic manifest under {root}")
        saved = json.loads(meta_path.read_text(encoding="utf-8")); actual = _sha256_file(path)
        changed = [key for key, value in identity.items() if saved.get(key) != value]
        if changed or actual != digest:
            raise RuntimeError(f"Generic manifest mismatch: {changed}. Delete {root} and rebuild; stale cache reuse is forbidden.")
        return [GenericCaptionRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line], digest
    text = "".join(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)
    # Byte-exact write keeps the persisted SHA platform independent.
    path.write_bytes(text.encode("utf-8")); write_json(meta_path, identity); return records, digest


def _digest(records: Sequence[GenericCaptionRecord]) -> str:
    text = "".join(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()
