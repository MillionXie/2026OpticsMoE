from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .io_utils import write_json


IMAGE_COLUMN_CANDIDATES = ("image", "img")
CAPTION_COLUMN_CANDIDATES = ("caption", "captions", "sentences")
SPLIT_COLUMN_CANDIDATES = ("split", "image_split")
ID_COLUMN_CANDIDATES = ("img_id", "image_id", "id")
FILENAME_COLUMN_CANDIDATES = ("filename", "file_name", "image_name")
REQUESTED_STANDARD_IMAGE_COUNTS = {"train": 29_783, "validation": 1_000, "test": 1_000}
KNOWN_REPOSITORY_IMAGE_COUNTS = {"train": 29_000, "validation": 1_014, "test": 1_000}
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FlickrImageRecord:
    image_id: str
    filename: str
    captions: tuple[str, ...]
    split: str
    source_split: str
    source_index: int


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    split: str
    image_id: str
    filename: str
    caption: str
    caption_index: int
    label: int
    caption_source_image_id: str
    negative_sampling_type: str


@dataclass
class DatasetBundle:
    train: "FlickrPairDataset"
    test: "FlickrPairDataset"
    metadata: dict[str, Any]
    cache_identity: dict[str, Any]


class FlickrPairDataset(Dataset[Any]):
    def __init__(self, pairs: Sequence[PairRecord], images: Mapping[str, FlickrImageRecord],
                 sources: Mapping[str, Any], image_columns: Mapping[str, str], prompt_template: str) -> None:
        self.pairs = list(pairs)
        self.images = dict(images)
        self.sources = dict(sources)
        self.image_columns = dict(image_columns)
        self.prompt_template = prompt_template
        self.targets = [int(pair.label) for pair in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Image.Image, str, float]:
        pair = self.pairs[index]
        image_record = self.images[pair.image_id]
        source = self.sources[image_record.source_split]
        value = source[image_record.source_index][self.image_columns[image_record.source_split]]
        image = _as_rgb_image(value)
        return image, self.prompt_template.format(caption=pair.caption), float(pair.label)

    def sample_metadata(self, index: int) -> dict[str, Any]:
        return asdict(self.pairs[index])


class IndexedDataset(Dataset[Any]):
    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, str, float, int]:
        image, prompt, target = self.dataset[index]
        return image, prompt, float(target), index


def indexed_collate(batch: Sequence[tuple[Any, str, float, int]]) -> tuple[list[Any], list[str], torch.Tensor, torch.Tensor]:
    images, prompts, labels, indices = zip(*batch)
    return list(images), list(prompts), torch.tensor(labels, dtype=torch.float32), torch.tensor(indices, dtype=torch.long)


def make_indexed_loader(dataset: Dataset[Any], batch_size: int, num_workers: int, shuffle: bool,
                        seed: int) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(IndexedDataset(dataset), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=indexed_collate, generator=generator,
                      pin_memory=False, persistent_workers=num_workers > 0)


def targets_of(dataset: Dataset[Any]) -> list[float]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return [float(dataset[index][2]) for index in range(len(dataset))]
    return [float(value) for value in targets]


def sample_metadata(dataset: Dataset[Any], index: int) -> dict[str, Any]:
    if hasattr(dataset, "sample_metadata"):
        return dict(dataset.sample_metadata(index))
    wrapped = getattr(dataset, "dataset", None)
    indices = getattr(dataset, "indices", None)
    if wrapped is not None and indices is not None:
        return sample_metadata(wrapped, int(indices[index]))
    return {"sample_index": int(index)}


def load_flickr30k(settings: Any, persist_manifest: bool = True,
                   raw_dataset: Any | None = None) -> DatasetBundle:
    loaded = raw_dataset if raw_dataset is not None else _load_huggingface(settings)
    sources = _source_mapping(loaded)
    records_by_split, image_columns, fingerprints = _scan_sources(sources)
    count_profile = _validate_image_splits(records_by_split, bool(settings.validate_standard_counts))

    # Flickr30k's official validation images are intentionally not used in this user-requested protocol.
    train_images = _limit_images(records_by_split["train"], settings.train_image_limit)
    test_images = _limit_images(records_by_split["test"], settings.test_image_limit)
    selected = {"train": train_images, "test": test_images}
    overlap = set(row.image_id for row in train_images) & set(row.image_id for row in test_images)
    if overlap:
        raise RuntimeError(f"Flickr30k train/test image_id leakage detected: {sorted(overlap)[:5]}")

    manifest_root = settings.output_dir / "pair_manifests"
    pairs: dict[str, list[PairRecord]] = {}
    digests: dict[str, str] = {}
    for split, images in selected.items():
        built = build_fixed_pairs(images, split, settings.captions_per_image,
                                  settings.negatives_per_positive, settings.seed)
        if persist_manifest:
            built, digest = _persist_or_validate_manifest(manifest_root, split, built, settings, fingerprints)
        else:
            digest = _pairs_digest(built)
        pairs[split] = built
        digests[split] = digest

    settings.resolved_dataset_fingerprints = fingerprints
    settings.pair_manifest_digests = digests
    image_lookup = {row.image_id: row for rows in records_by_split.values() for row in rows}
    train = FlickrPairDataset(pairs["train"], image_lookup, sources, image_columns, settings.prompt_template)
    test = FlickrPairDataset(pairs["test"], image_lookup, sources, image_columns, settings.prompt_template)
    metadata = {
        "dataset": "flickr30k_image_text_matching",
        "repo_id": settings.dataset_repo_id,
        "revision": settings.dataset_revision,
        "dataset_fingerprints": fingerprints,
        "standard_image_counts": {name: len(rows) for name, rows in records_by_split.items()},
        "image_count_profile": count_profile,
        "requested_standard_image_counts": REQUESTED_STANDARD_IMAGE_COUNTS,
        "known_repository_image_counts": KNOWN_REPOSITORY_IMAGE_COUNTS,
        "train_images": len(train_images),
        "validation_images": len(records_by_split["validation"]),
        "test_images": len(test_images),
        "validation_usage": "official split validated but intentionally unused",
        "train_pairs": len(train),
        "test_pairs": len(test),
        "positive_pairs": {name: sum(row.label == 1 for row in pairs[name]) for name in pairs},
        "negative_pairs": {name: sum(row.label == 0 for row in pairs[name]) for name in pairs},
        "captions_per_image": settings.captions_per_image,
        "negatives_per_positive": settings.negatives_per_positive,
        "negative_sampling_algorithm": settings.negative_sampling_algorithm,
        "seed": settings.seed,
        "prompt_template": settings.prompt_template,
        "pair_manifest_digests": digests,
        "split_policy": "official image-level train/test; no validation; test evaluated every epoch",
        "class_names": ["not_match", "match"],
    }
    identity = {
        "dataset": metadata["dataset"], "repo_id": settings.dataset_repo_id,
        "revision": settings.dataset_revision, "dataset_fingerprints": fingerprints,
        "pair_manifest_digests": digests, "prompt_template": settings.prompt_template,
        "negative_sampling_algorithm": settings.negative_sampling_algorithm,
        "captions_per_image": settings.captions_per_image,
        "negatives_per_positive": settings.negatives_per_positive, "seed": settings.seed,
    }
    if persist_manifest:
        write_json(manifest_root / "summary.json", metadata)
    return DatasetBundle(train=train, test=test, metadata=metadata, cache_identity=identity)


def build_fixed_pairs(images: Sequence[FlickrImageRecord], split: str, captions_per_image: int,
                      negatives_per_positive: int, seed: int) -> list[PairRecord]:
    ordered = sorted(images, key=lambda row: row.image_id)
    if len(ordered) < 2:
        raise RuntimeError(f"Flickr30k {split} requires at least two images for negative pairing")
    selected: list[list[int]] = []
    for image in ordered:
        if len(image.captions) < captions_per_image:
            raise RuntimeError(f"Image {image.image_id} has {len(image.captions)} captions, fewer than requested {captions_per_image}")
        ranked = sorted(range(len(image.captions)), key=lambda index: _stable_hash(seed, split, image.image_id, index))
        selected.append(ranked[:captions_per_image])

    rows: list[PairRecord] = []
    for caption_slot in range(captions_per_image):
        for image_index, image in enumerate(ordered):
            caption_index = selected[image_index][caption_slot]
            rows.append(PairRecord(
                pair_id=f"{split}:{image.image_id}:p{caption_slot}", split=split,
                image_id=image.image_id, filename=image.filename,
                caption=image.captions[caption_index], caption_index=caption_index, label=1,
                caption_source_image_id=image.image_id, negative_sampling_type="positive_ground_truth",
            ))
        for negative_rank in range(negatives_per_positive):
            permutation = _valid_derangement(ordered, selected, caption_slot, seed, split, negative_rank)
            for image_index, image in enumerate(ordered):
                source_index = permutation[image_index]
                source = ordered[source_index]
                caption_index = selected[source_index][caption_slot]
                rows.append(PairRecord(
                    pair_id=f"{split}:{image.image_id}:n{caption_slot}_{negative_rank}", split=split,
                    image_id=image.image_id, filename=image.filename,
                    caption=source.captions[caption_index], caption_index=caption_index, label=0,
                    caption_source_image_id=source.image_id,
                    negative_sampling_type="deterministic_split_derangement",
                ))
    rows.sort(key=lambda row: row.pair_id)
    positives = sum(row.label == 1 for row in rows); negatives = len(rows) - positives
    if negatives != positives * negatives_per_positive:
        raise RuntimeError("Internal error: Flickr30k fixed pair manifest is not balanced as configured")
    return rows


def _valid_derangement(images: Sequence[FlickrImageRecord], selected: Sequence[Sequence[int]], caption_slot: int,
                       seed: int, split: str, negative_rank: int) -> list[int]:
    size = len(images)
    for attempt in range(4096):
        order = list(range(size))
        rng = random.Random(_stable_hash(seed, split, caption_slot, negative_rank, attempt))
        # Sattolo's algorithm produces one cycle and therefore no fixed points.
        for index in range(size - 1, 0, -1):
            other = rng.randrange(index)
            order[index], order[other] = order[other], order[index]
        valid = True
        for target_index, source_index in enumerate(order):
            target, source = images[target_index], images[source_index]
            caption = source.captions[selected[source_index][caption_slot]]
            if source.image_id == target.image_id or caption in target.captions:
                valid = False
                break
        if valid:
            return order
    raise RuntimeError(
        f"Could not construct a collision-free deterministic negative-caption derangement for {split}. "
        "The split may be too small or contain duplicate captions."
    )


def _load_huggingface(settings: Any) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for Flickr30k. Install requirements.txt.") from exc
    settings.data_root.mkdir(parents=True, exist_ok=True)
    previous_endpoint = os.environ.get("HF_ENDPOINT")
    previous_offline = os.environ.get("HF_DATASETS_OFFLINE")
    try:
        if settings.hf_endpoint:
            os.environ["HF_ENDPOINT"] = str(settings.hf_endpoint)
        if not settings.download or settings.local_files_only:
            os.environ["HF_DATASETS_OFFLINE"] = "1"
        kwargs: dict[str, Any] = {
            "cache_dir": str(settings.data_root),
            # nlphuji/flickr30k is currently distributed through its repository
            # dataset script. datasets>=3 removed the legacy task_templates API
            # used by that script, so this experiment pins datasets<3 and
            # explicitly trusts this configured repository.
            "trust_remote_code": True,
        }
        if settings.dataset_revision:
            kwargs["revision"] = settings.dataset_revision
        return load_dataset(settings.dataset_repo_id, **kwargs)
    except Exception as exc:
        mode = "offline cache" if (not settings.download or settings.local_files_only) else "Hugging Face download/cache"
        raise RuntimeError(
            f"Unable to load {settings.dataset_repo_id} using {mode} at {settings.data_root}. "
            "No fallback dataset is allowed. Check network/HF_ENDPOINT or pre-populate the configured cache. "
            f"Original error: {exc}"
        ) from exc
    finally:
        _restore_env("HF_ENDPOINT", previous_endpoint)
        _restore_env("HF_DATASETS_OFFLINE", previous_offline)


def _source_mapping(loaded: Any) -> dict[str, Any]:
    column_names = getattr(loaded, "column_names", None)
    if isinstance(loaded, Mapping) or isinstance(column_names, Mapping):
        return {str(name): loaded[name] for name in loaded.keys()}
    return {"test": loaded}


def _scan_sources(sources: Mapping[str, Any]) -> tuple[dict[str, list[FlickrImageRecord]], dict[str, str], dict[str, str]]:
    records = {"train": [], "validation": [], "test": []}
    image_columns: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    seen_ids: set[str] = set()
    for source_name, source in sources.items():
        columns = list(getattr(source, "column_names", []))
        if not columns and len(source):
            columns = list(source[0].keys())
        image_column = _find_column(columns, IMAGE_COLUMN_CANDIDATES, "image")
        caption_column = _find_column(columns, CAPTION_COLUMN_CANDIDATES, "caption list")
        split_column = _find_column(columns, SPLIT_COLUMN_CANDIDATES, "internal split")
        id_column = _find_column(columns, ID_COLUMN_CANDIDATES, "image id")
        filename_column = _find_column(columns, FILENAME_COLUMN_CANDIDATES, "filename", required=False)
        image_columns[source_name] = image_column
        fingerprints[source_name] = str(getattr(source, "_fingerprint", "unknown"))
        metadata_columns = [caption_column, split_column, id_column] + ([filename_column] if filename_column else [])
        metadata_source = source.select_columns(metadata_columns) if hasattr(source, "select_columns") else source
        for index in range(len(metadata_source)):
            row = metadata_source[index]
            split = _normalize_split(row[split_column])
            image_id = str(row[id_column])
            if image_id in seen_ids:
                raise RuntimeError(f"Duplicate Flickr30k image_id across source rows: {image_id}")
            seen_ids.add(image_id)
            captions = _normalize_captions(row[caption_column], image_id)
            filename = str(row[filename_column]) if filename_column and row.get(filename_column) is not None else image_id
            records[split].append(FlickrImageRecord(image_id, filename, captions, split, source_name, index))
    for split in records:
        records[split].sort(key=lambda row: row.image_id)
    return records, image_columns, fingerprints


def _validate_image_splits(records: Mapping[str, Sequence[FlickrImageRecord]], strict_counts: bool) -> str:
    actual = {split: len(records.get(split, ())) for split in ("train", "validation", "test")}
    for split in ("train", "validation", "test"):
        if not records.get(split):
            raise RuntimeError(f"Flickr30k internal split '{split}' was not found or is empty")
    if strict_counts and actual not in (REQUESTED_STANDARD_IMAGE_COUNTS, KNOWN_REPOSITORY_IMAGE_COUNTS):
        raise RuntimeError(
            f"Unexpected Flickr30k internal image counts: {actual}. Expected either the requested profile "
            f"{REQUESTED_STANDARD_IMAGE_COUNTS} or the known nlphuji/flickr30k Karpathy profile "
            f"{KNOWN_REPOSITORY_IMAGE_COUNTS}. Refusing an unknown dataset revision."
        )
    if actual == REQUESTED_STANDARD_IMAGE_COUNTS:
        profile = "requested_31783"
    elif actual == KNOWN_REPOSITORY_IMAGE_COUNTS:
        profile = "nlphuji_current_karpathy_31014"
        print(
            "WARNING: nlphuji/flickr30k currently exposes 29,000 train, 1,014 validation, and 1,000 test "
            "images (31,014 total), not the requested 29,783/1,000/1,000 profile. The exact repository "
            "split is retained and this discrepancy is recorded in dataset.json.", flush=True,
        )
    else:
        profile = "custom_counts_validation_disabled"
    for split, rows in records.items():
        invalid = [row.image_id for row in rows if len(row.captions) != 5]
        if strict_counts and invalid:
            raise RuntimeError(f"Flickr30k {split} contains images without exactly five captions: {invalid[:5]}")
    sets = {name: {row.image_id for row in rows} for name, rows in records.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise RuntimeError(f"Flickr30k image_id leakage between {left} and {right}: {sorted(overlap)[:5]}")
    return profile


def _persist_or_validate_manifest(root: Path, split: str, pairs: Sequence[PairRecord], settings: Any,
                                  fingerprints: Mapping[str, str]) -> tuple[list[PairRecord], str]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{split}.jsonl"
    metadata_path = root / f"{split}_metadata.json"
    expected_identity = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION, "split": split,
        "dataset": "flickr30k_image_text_matching", "repo_id": settings.dataset_repo_id,
        "revision": settings.dataset_revision, "dataset_fingerprints": dict(fingerprints),
        "seed": settings.seed, "prompt_template": settings.prompt_template,
        "negative_sampling_algorithm": settings.negative_sampling_algorithm,
        "captions_per_image": settings.captions_per_image,
        "negatives_per_positive": settings.negatives_per_positive,
        "sample_count": len(pairs),
    }
    expected_digest = _pairs_digest(pairs)
    if path.is_file() or metadata_path.is_file():
        if not path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Incomplete Flickr30k pair manifest for {split}; remove {root} and rebuild")
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        changed = [key for key, value in expected_identity.items() if existing_metadata.get(key) != value]
        actual_digest = _file_sha256(path)
        if changed or actual_digest != existing_metadata.get("sha256") or actual_digest != expected_digest:
            raise RuntimeError(
                f"Flickr30k pair manifest mismatch for {split}: fields={changed}, "
                f"saved_sha256={existing_metadata.get('sha256')}, expected_sha256={expected_digest}. "
                f"Delete {root} before rebuilding; stale manifests/caches are never silently reused."
            )
        loaded = [PairRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return loaded, actual_digest
    text = "".join(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in pairs)
    path.write_text(text, encoding="utf-8")
    digest = _file_sha256(path)
    write_json(metadata_path, {**expected_identity, "sha256": digest,
                               "positive_count": sum(row.label == 1 for row in pairs),
                               "negative_count": sum(row.label == 0 for row in pairs)})
    return list(pairs), digest


def _pairs_digest(pairs: Sequence[PairRecord]) -> str:
    payload = "".join(json.dumps(asdict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in pairs)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _limit_images(records: Sequence[FlickrImageRecord], limit: int | None) -> list[FlickrImageRecord]:
    return list(records if limit is None else records[:limit])


def _find_column(columns: Sequence[str], candidates: Sequence[str], purpose: str, required: bool = True) -> str | None:
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    if required:
        raise RuntimeError(f"Flickr30k is missing the {purpose} column. Accepted={list(candidates)}; found={list(columns)}")
    return None


def _normalize_split(value: Any) -> str:
    text = str(value).strip().lower()
    aliases = {"train": "train", "training": "train", "val": "validation",
               "valid": "validation", "validation": "validation", "test": "test"}
    if text not in aliases:
        raise RuntimeError(f"Unsupported Flickr30k internal split value: {value!r}")
    return aliases[text]


def _normalize_captions(value: Any, image_id: str) -> tuple[str, ...]:
    if isinstance(value, str):
        captions = [value]
    elif isinstance(value, Mapping) and "raw" in value:
        captions = list(value["raw"])
    else:
        try:
            captions = list(value)
        except TypeError as exc:
            raise RuntimeError(f"Flickr30k captions for image {image_id} are not a sequence") from exc
    cleaned = tuple(str(caption).strip() for caption in captions if str(caption).strip())
    if not cleaned:
        raise RuntimeError(f"Flickr30k image {image_id} has no usable caption")
    return cleaned


def _as_rgb_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, Mapping):
        if value.get("path"):
            with Image.open(value["path"]) as image:
                return image.convert("RGB")
        if value.get("bytes") is not None:
            import io
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return image.convert("RGB")
    raise RuntimeError(f"Unsupported Flickr30k image payload type: {type(value).__name__}")


def _stable_hash(*values: Any) -> int:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
