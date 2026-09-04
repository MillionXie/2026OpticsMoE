from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from .settings import (
    FEATURE_CONTRACT,
    LANGUAGE_CONTRACT,
    QUALITY_CONTRACT,
    TARGET_PROMPTS,
    feature_contract_for_grid,
    quality_contract_for_grid,
    ExperimentSettings,
)


SPATIAL_COLUMNS = (
    "spatial",
    "spatial_score",
    "spatial_quality",
    "spatial_mos",
    "spatial_quality_score",
)
TEMPORAL_COLUMNS = (
    "temporal",
    "temporal_score",
    "temporal_quality",
    "temporal_mos",
    "temporal_quality_score",
)
ID_COLUMNS = ("sample_id", "video_id", "id", "name", "video_name")
PATH_COLUMNS = ("video_path", "path", "file_path", "filepath", "video")
SPLIT_COLUMNS = ("split", "subset", "partition")

# These records bind the two independently stored cache files to one exact
# frozen Qwen front.  In particular, a matching tensor shape or model name is
# not sufficient: every record is content addressed and the pair record names
# both component digests.
QWEN_SOURCE_IDENTITY_CONTRACT = "qwen3_vl_local_checkpoint_identity_v1"
QWEN_COMPONENT_FINGERPRINT_CONTRACT = "qwen3_vl_front_component_fingerprint_v1"
QWEN_FRONT_PAIR_CONTRACT = "qwen3_vl_vision_language_front_pair_v1"


def _canonical_record_sha256(record: Mapping[str, Any]) -> str:
    payload = {str(key): value for key, value in record.items() if key != "sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hashed_record(
    raw: Any,
    *,
    contract: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError(f"{label} is missing or is not a mapping")
    record = dict(raw)
    if record.get("contract") != contract:
        raise RuntimeError(
            f"{label} contract mismatch: expected {contract!r}, "
            f"got {record.get('contract')!r}"
        )
    expected = record.get("sha256")
    actual = _canonical_record_sha256(record)
    if not isinstance(expected, str) or len(expected) != 64 or expected != actual:
        raise RuntimeError(
            f"{label} content digest is invalid; the cache provenance record "
            "was modified or incompletely written"
        )
    return record


def _validate_cache_front_identity(
    vision: Mapping[str, Any], language: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate that Vision and Language came from one exact Qwen front.

    A source model name alone cannot prevent an old Vision cache from being
    silently paired with new text embeddings.  The shared pair digest binds
    the checkpoint/config identity, patch+position tensor digest, and
    ``embed_tokens`` tensor digest.  Both cache files must carry the same
    self-validating records.
    """

    vision_source = _validate_hashed_record(
        vision.get("qwen_source_identity"),
        contract=QWEN_SOURCE_IDENTITY_CONTRACT,
        label="Vision Qwen source identity",
    )
    language_source = _validate_hashed_record(
        language.get("qwen_source_identity"),
        contract=QWEN_SOURCE_IDENTITY_CONTRACT,
        label="Language Qwen source identity",
    )
    vision_component = _validate_hashed_record(
        vision.get("qwen_vision_front_fingerprint"),
        contract=QWEN_COMPONENT_FINGERPRINT_CONTRACT,
        label="Vision patch+position fingerprint",
    )
    language_component = _validate_hashed_record(
        language.get("qwen_language_embedding_fingerprint"),
        contract=QWEN_COMPONENT_FINGERPRINT_CONTRACT,
        label="Language embed_tokens fingerprint",
    )
    vision_pair = _validate_hashed_record(
        vision.get("qwen_front_pair_identity"),
        contract=QWEN_FRONT_PAIR_CONTRACT,
        label="Vision front-pair identity",
    )
    language_pair = _validate_hashed_record(
        language.get("qwen_front_pair_identity"),
        contract=QWEN_FRONT_PAIR_CONTRACT,
        label="Language front-pair identity",
    )
    if vision_source != language_source:
        raise RuntimeError(
            "Vision and Language caches do not originate from the same Qwen "
            "checkpoint/revision; regenerating both caches together is required"
        )
    if vision_pair != language_pair:
        raise RuntimeError(
            "Vision and Language caches belong to different Qwen front bundles; "
            "an old Vision cache must not be mixed with new text embeddings"
        )
    if vision_component.get("component") != "vision_patch_embed_and_position":
        raise RuntimeError("Vision cache has the wrong Qwen component fingerprint")
    if language_component.get("component") != "language_embed_tokens":
        raise RuntimeError("Language cache has the wrong Qwen component fingerprint")
    expected_links = {
        "source_identity_sha256": vision_source["sha256"],
        "vision_front_sha256": vision_component["sha256"],
        "language_embedding_sha256": language_component["sha256"],
    }
    for key, expected in expected_links.items():
        if vision_pair.get(key) != expected:
            raise RuntimeError(
                f"Qwen front-pair identity does not bind the loaded {key}: "
                f"expected {expected}, got {vision_pair.get(key)}"
            )
    for label, component in (
        ("Vision", vision_component),
        ("Language", language_component),
    ):
        if component.get("source_identity_sha256") != vision_source["sha256"]:
            raise RuntimeError(
                f"{label} component fingerprint is linked to a different Qwen checkpoint"
            )
    vision_contract = vision.get("qwen_front_contract")
    language_contract = language.get("qwen_front_contract")
    if not isinstance(vision_contract, Mapping) or dict(vision_contract) != dict(
        language_contract or {}
    ):
        raise RuntimeError("Vision and Language Qwen architecture contracts differ")
    return {
        "source": vision_source,
        "vision_component": vision_component,
        "language_component": language_component,
        "pair": vision_pair,
    }


def _first(row: Mapping[str, Any], names: Iterable[str], *, required: bool) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    if required:
        raise KeyError(f"None of the required columns exist: {tuple(names)}")
    return None


def file_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestRow:
    sample_id: str
    video_path: str
    split: str
    spatial: float
    temporal: float

    def target(self, name: str) -> float:
        if name == "spatial":
            return self.spatial
        if name == "temporal":
            return self.temporal
        raise ValueError(f"Unknown target name {name!r}")


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest = Path(path).expanduser().resolve()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise RuntimeError(f"LGVQ manifest is empty: {manifest}")
    result: list[ManifestRow] = []
    seen: set[str] = set()
    for row_number, row in enumerate(raw_rows, 2):
        video_path = str(_first(row, PATH_COLUMNS, required=True))
        identifier = _first(row, ID_COLUMNS, required=False)
        sample_id = str(identifier or Path(video_path).stem)
        if sample_id in seen:
            raise RuntimeError(f"Duplicate sample_id {sample_id!r} at row {row_number}")
        seen.add(sample_id)
        split = str(_first(row, SPLIT_COLUMNS, required=True)).strip().lower()
        split = {"val": "validation", "valid": "validation", "dev": "validation"}.get(split, split)
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split {split!r} at row {row_number}")
        result.append(
            ManifestRow(
                sample_id=sample_id,
                video_path=video_path,
                split=split,
                spatial=float(_first(row, SPATIAL_COLUMNS, required=True)),
                temporal=float(_first(row, TEMPORAL_COLUMNS, required=True)),
            )
        )
    return result


def split_counts(rows: Iterable[ManifestRow]) -> dict[str, int]:
    result = {"train": 0, "validation": 0, "test": 0}
    for row in rows:
        result[row.split] += 1
    return result


def _load_torch(path: Path, *, mmap: bool = False) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def load_vision_cache(
    path: str | Path, *, frame_count: int, token_grid: int = 7
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = _load_torch(source, mmap=True)
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise RuntimeError(f"Unsupported Vision cache schema: {source}")
    expected_feature_contract = feature_contract_for_grid(token_grid)
    expected_quality_contract = quality_contract_for_grid(token_grid)
    if payload.get("feature_contract") != expected_feature_contract:
        raise RuntimeError(
            f"Vision cache contract mismatch: expected {expected_feature_contract!r}, "
            f"got {payload.get('feature_contract')!r}"
        )
    required = {
        "vision_tokens",
        "quality_tokens",
        "quality_contract",
        "sample_ids",
        "video_paths",
        "splits",
        "qwen_front_contract",
        "qwen_source_identity",
        "qwen_vision_front_fingerprint",
        "qwen_front_pair_identity",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Vision cache is missing {sorted(missing)}")
    tokens = payload["vision_tokens"]
    quality = payload["quality_tokens"]
    count = len(payload["sample_ids"])
    if int(payload.get("frame_count", -1)) != frame_count:
        raise ValueError(
            f"Vision cache frame_count={payload.get('frame_count')} does not match "
            f"configured frame_count={frame_count}"
        )
    expected_shape = (count, frame_count, token_grid * token_grid, 1024)
    if not torch.is_tensor(tokens) or tokens.dtype != torch.float16 or tuple(tokens.shape) != expected_shape:
        raise ValueError(
            f"vision_tokens must be float16 {expected_shape}; got "
            f"{getattr(tokens, 'dtype', None)} {getattr(tokens, 'shape', None)}"
        )
    if payload["quality_contract"] != expected_quality_contract:
        raise RuntimeError("Quality-side cache contract does not match this experiment")
    expected_quality_shape = (count, frame_count, token_grid * token_grid, 14)
    if (
        not torch.is_tensor(quality)
        or quality.dtype != torch.float16
        or tuple(quality.shape) != expected_quality_shape
    ):
        raise ValueError(
            f"quality_tokens must be float16 {expected_quality_shape}; got "
            f"{getattr(quality, 'dtype', None)} {getattr(quality, 'shape', None)}"
        )
    if any(len(payload[name]) != count for name in ("video_paths", "splits")):
        raise ValueError("Vision-cache metadata lengths do not match vision_tokens")
    if len(set(map(str, payload["sample_ids"]))) != count:
        raise ValueError("Vision-cache sample_ids contain duplicates")
    # Keep validation bounded: a formal Vision tensor is about 4.5 GiB and a
    # single torch.isfinite over all rows would allocate another giant mask.
    for start in range(0, count, 16):
        if not bool(torch.isfinite(tokens[start : start + 16]).all()):
            raise ValueError(f"Vision cache contains non-finite values near row {start}")
        if not bool(torch.isfinite(quality[start : start + 16]).all()):
            raise ValueError(f"Quality cache contains non-finite values near row {start}")
    return payload


def load_language_cache(
    path: str | Path,
    *,
    target_name: str,
    prompt: str,
    maximum_tokens: int,
    frame_count: int,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = _load_torch(source)
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise RuntimeError(f"Unsupported Language cache schema: {source}")
    if payload.get("language_contract") != LANGUAGE_CONTRACT:
        raise RuntimeError("Language cache contract does not match this experiment")
    if payload.get("target_name") != target_name:
        raise RuntimeError(
            f"Cannot use {payload.get('target_name')!r} prompt embeddings for a {target_name!r} run"
        )
    if payload.get("prompt") != prompt or prompt != TARGET_PROMPTS[target_name]:
        raise RuntimeError("Cached text and configured target-specific prompt differ")
    required = {
        "language_tokens",
        "attention_mask",
        "input_ids",
        "qwen_front_contract",
        "qwen_source_identity",
        "qwen_language_embedding_fingerprint",
        "qwen_front_pair_identity",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Language cache is missing {sorted(missing)}")
    tokens = payload["language_tokens"]
    mask = payload["attention_mask"]
    input_ids = payload["input_ids"]
    if not torch.is_tensor(tokens) or tokens.dtype != torch.float16 or tokens.ndim != 3:
        raise ValueError("language_tokens must be float16 [1,L,2048]")
    if tokens.shape[0] != 1 or tokens.shape[-1] != 2048:
        raise ValueError(f"language_tokens must be [1,L,2048], got {tuple(tokens.shape)}")
    if tuple(mask.shape) != tuple(tokens.shape[:2]) or tuple(input_ids.shape) != tuple(tokens.shape[:2]):
        raise ValueError("attention_mask and input_ids must both be [1,L]")
    if mask.dtype != torch.bool or input_ids.dtype != torch.int64:
        raise ValueError("attention_mask must be bool and input_ids must be int64")
    if frame_count + tokens.shape[1] > maximum_tokens:
        raise ValueError(
            f"{frame_count} frame tokens + {tokens.shape[1]} prompt tokens exceed the "
            f"configured serial limit {maximum_tokens}"
        )
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("Language cache contains non-finite values")
    return payload


def load_quality_feature_cache(
    path: str | Path,
    *,
    sample_ids: Sequence[str],
    frame_count: int,
    token_grid: int,
    width: int,
) -> dict[str, Any]:
    """Load an independently auditable, frozen convolutional input-head cache."""

    source = Path(path).expanduser().resolve()
    payload = _load_torch(source, mmap=True)
    if not isinstance(payload, dict) or payload.get("contract") != "lgvq_quality_conv5_feature_cache_v1":
        raise RuntimeError(f"Unsupported quality-feature cache: {source}")
    if list(map(str, payload.get("sample_ids", []))) != list(map(str, sample_ids)):
        raise RuntimeError("Quality-feature cache sample order differs from the manifest")
    tokens = payload.get("quality_tokens")
    expected = (len(sample_ids), frame_count, token_grid * token_grid, width)
    if not torch.is_tensor(tokens) or tokens.dtype != torch.float16 or tuple(tokens.shape) != expected:
        raise ValueError(f"quality-feature tokens must be float16 {expected}")
    for start in range(0, len(sample_ids), 16):
        if not bool(torch.isfinite(tokens[start : start + 16]).all()):
            raise ValueError(f"Quality-feature cache contains non-finite values near row {start}")
    return payload


def load_raw_frame_cache(
    path: str | Path, *, sample_ids: Sequence[str], frame_count: int
) -> dict[str, Any]:
    """Load the exact uint8 frames used to train the optional conv5 stem."""

    source = Path(path).expanduser().resolve()
    payload = _load_torch(source, mmap=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported raw-frame cache: {source}")
    if list(map(str, payload.get("sample_ids", []))) != list(map(str, sample_ids)):
        raise RuntimeError("Raw-frame cache sample order differs from the manifest")
    frames = payload.get("frames")
    expected_prefix = (len(sample_ids), frame_count, 3)
    if (
        not torch.is_tensor(frames)
        or frames.dtype != torch.uint8
        or frames.ndim != 5
        or tuple(frames.shape[:3]) != expected_prefix
        or tuple(frames.shape[-2:]) != (224, 224)
    ):
        raise ValueError(
            f"raw frames must be uint8 [N,{frame_count},3,224,224], got "
            f"{None if not torch.is_tensor(frames) else tuple(frames.shape)}"
        )
    return payload


def load_vgg_feature_cache(
    path: str | Path, *, sample_ids: Sequence[str], frame_count: int, token_grid: int
) -> dict[str, Any]:
    """Load frozen plain-VGG16 Conv/ReLU/Pool tokens in manifest order."""

    source = Path(path).expanduser().resolve()
    payload = _load_torch(source, mmap=True)
    expected_contract = "lgvq_frozen_plain_vgg16_4f_14x14x512_v1"
    if not isinstance(payload, dict) or payload.get("contract") != expected_contract:
        raise RuntimeError(f"Unsupported plain-VGG feature cache: {source}")
    if list(map(str, payload.get("sample_ids", []))) != list(map(str, sample_ids)):
        raise RuntimeError("Plain-VGG feature cache sample order differs from the manifest")
    tokens = payload.get("tokens")
    expected = (len(sample_ids), frame_count, token_grid * token_grid, 512)
    if not torch.is_tensor(tokens) or tokens.dtype != torch.float16 or tuple(tokens.shape) != expected:
        raise ValueError(f"plain-VGG tokens must be float16 {expected}")
    # The cache builder validates every generated batch before persisting it.
    # Touching all ~2.1 GB here once per parallel trial causes severe CPU/I/O
    # contention, so loading performs representative boundary/midpoint checks.
    probe_indices = sorted({0, len(sample_ids) // 2, len(sample_ids) - 1})
    if any(not bool(torch.isfinite(tokens[index]).all()) for index in probe_indices):
        raise ValueError("Plain-VGG cache contains non-finite values in a probe sample")
    return payload


def _align_soft_targets(
    path: Path,
    *,
    rows: Sequence[ManifestRow],
    target_name: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    raw = _load_torch(path)
    if not isinstance(raw, dict):
        raise ValueError("Training soft targets must be a .pt mapping")
    required = {"sample_ids", "predictions", "target_names"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Training soft-target file is missing {sorted(missing)}")
    names = list(raw["target_names"])
    if target_name not in names:
        raise ValueError(f"Soft-target file has no {target_name!r} column: {names}")
    ids = [str(value) for value in raw["sample_ids"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Training soft-target sample_ids contain duplicates")
    predictions = torch.as_tensor(raw["predictions"], dtype=torch.float32).cpu()
    if predictions.ndim != 2 or tuple(predictions.shape) != (len(ids), len(names)):
        raise ValueError("Soft-target predictions do not match sample_ids/target_names")
    train_ids = [row.sample_id for row in rows if row.split == "train"]
    if set(ids) != set(train_ids):
        raise ValueError("Soft-target IDs must exactly match the manifest train split")
    column = names.index(target_name)
    lookup = {sample_id: predictions[index, column] for index, sample_id in enumerate(ids)}
    values = torch.zeros(len(rows), dtype=torch.float32)
    present = torch.zeros(len(rows), dtype=torch.bool)
    for index, row in enumerate(rows):
        if row.split == "train":
            values[index] = lookup[row.sample_id]
            present[index] = True
    return values, present, {
        "usage": "training_only_single_scalar_soft_target",
        "target_name": target_name,
        "path": str(path),
        "sha256": file_sha256(path),
        "sample_count": len(train_ids),
        "full_teacher_loaded_during_training_or_inference": False,
    }


def load_single_metric_cache(settings: ExperimentSettings) -> dict[str, Any]:
    if settings.manifest_path is None:
        raise ValueError("data.manifest is required")
    if settings.vision_cache_path is None or settings.language_cache_path is None:
        raise ValueError("data.vision_cache and data.language_cache are required")
    rows = read_manifest(settings.manifest_path)
    vision = load_vision_cache(
        settings.vision_cache_path,
        frame_count=settings.frame_count,
        token_grid=settings.token_grid,
    )
    language = load_language_cache(
        settings.language_cache_path,
        target_name=settings.target_name,
        prompt=settings.prompt,
        maximum_tokens=settings.maximum_language_tokens,
        frame_count=settings.frame_count,
    )
    front_identity = _validate_cache_front_identity(vision, language)
    manifest_ids = [row.sample_id for row in rows]
    cache_ids = [str(value) for value in vision["sample_ids"]]
    if manifest_ids != cache_ids:
        raise RuntimeError(
            "Manifest and Vision cache sample order differ; positional joining is forbidden"
        )
    manifest_splits = [row.split for row in rows]
    if manifest_splits != list(vision["splits"]):
        raise RuntimeError("Manifest and Vision cache split assignments differ")
    targets = torch.tensor(
        [row.target(settings.target_name) for row in rows], dtype=torch.float32
    )
    if not bool(torch.isfinite(targets).all()):
        raise ValueError("Manifest contains non-finite target values")
    result: dict[str, Any] = {
        "schema_version": 1,
        "target_name": settings.target_name,
        "prompt": settings.prompt,
        "vision_tokens": vision["vision_tokens"],
        "quality_tokens": vision["quality_tokens"],
        "language_tokens": language["language_tokens"],
        "language_mask": language["attention_mask"],
        "input_ids": language["input_ids"],
        "sample_ids": manifest_ids,
        "video_paths": [row.video_path for row in rows],
        "splits": manifest_splits,
        "targets": targets,
        "manifest_path": str(settings.manifest_path),
        "manifest_sha256": file_sha256(settings.manifest_path),
        "vision_cache_path": str(settings.vision_cache_path),
        "language_cache_path": str(settings.language_cache_path),
        "qwen_front_identity": front_identity,
    }
    if settings.quality_feature_cache_path is not None:
        auxiliary = load_quality_feature_cache(
            settings.quality_feature_cache_path,
            sample_ids=manifest_ids,
            frame_count=settings.frame_count,
            token_grid=settings.token_grid,
            width=settings.quality_input_width,
        )
        result["quality_tokens"] = auxiliary["quality_tokens"]
        result["quality_feature_provenance"] = {
            key: auxiliary.get(key)
            for key in (
                "contract",
                "source_checkpoint",
                "source_checkpoint_sha256",
                "source_frame_cache",
                "source_frame_cache_sha256",
            )
        }
    if settings.raw_frame_cache_path is not None:
        raw_frames = load_raw_frame_cache(
            settings.raw_frame_cache_path,
            sample_ids=manifest_ids,
            frame_count=settings.frame_count,
        )
        result["raw_frames"] = raw_frames["frames"]
        result["raw_frame_cache_path"] = str(settings.raw_frame_cache_path)
        result["raw_frame_cache_sha256"] = file_sha256(settings.raw_frame_cache_path)
    if settings.vgg_feature_cache_path is not None:
        vgg = load_vgg_feature_cache(
            settings.vgg_feature_cache_path,
            sample_ids=manifest_ids,
            frame_count=settings.frame_count,
            token_grid=settings.token_grid,
        )
        result["vgg_tokens"] = vgg["tokens"]
        result["vgg_feature_cache_path"] = str(settings.vgg_feature_cache_path)
        result["vgg_feature_cache_sha256"] = file_sha256(settings.vgg_feature_cache_path)
    if settings.training_soft_targets_path is not None:
        soft, present, provenance = _align_soft_targets(
            settings.training_soft_targets_path,
            rows=rows,
            target_name=settings.target_name,
        )
        result["soft_targets"] = soft
        result["soft_target_present"] = present
        result["training_soft_target_provenance"] = provenance
    return result


def cache_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    tokens = payload["vision_tokens"]
    return {
        "target_name": payload["target_name"],
        "vision_shape": list(tokens.shape),
        "vision_dtype": str(tokens.dtype),
        "language_shape": list(payload["language_tokens"].shape),
        "quality_shape": list(payload["quality_tokens"].shape),
        "raw_frame_shape": None
        if "raw_frames" not in payload
        else list(payload["raw_frames"].shape),
        "vgg_shape": None
        if "vgg_tokens" not in payload
        else list(payload["vgg_tokens"].shape),
        "language_dtype": str(payload["language_tokens"].dtype),
        "input_ids_shape": list(payload["input_ids"].shape),
        "split_counts": {
            split: list(payload["splits"]).count(split)
            for split in ("train", "validation", "test")
        },
        "target_shape": list(payload["targets"].shape),
        "one_scalar_target": True,
        "qwen_checkpoint_identity_sha256": payload["qwen_front_identity"]["source"][
            "sha256"
        ],
        "qwen_vision_front_sha256": payload["qwen_front_identity"][
            "vision_component"
        ]["sha256"],
        "qwen_language_embedding_sha256": payload["qwen_front_identity"][
            "language_component"
        ]["sha256"],
        "qwen_front_pair_sha256": payload["qwen_front_identity"]["pair"]["sha256"],
    }


class LGVQSingleMetricDataset(Dataset[dict[str, Any]]):
    def __init__(self, payload: Mapping[str, Any], split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        self.payload = payload
        self.indices = [
            index for index, value in enumerate(payload["splits"]) if value == split
        ]
        if not self.indices:
            raise RuntimeError(f"No samples for split={split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.indices[index]
        item: dict[str, Any] = {
            "vision_tokens": self.payload["vision_tokens"][source].float(),
            "quality_tokens": self.payload["quality_tokens"][source].float(),
            "language_tokens": self.payload["language_tokens"][0].float(),
            "language_mask": self.payload["language_mask"][0].bool(),
            "input_ids": self.payload["input_ids"][0].long(),
            "target": self.payload["targets"][source].float(),
            "target_name": self.payload["target_name"],
            "sample_id": self.payload["sample_ids"][source],
            "video_path": self.payload["video_paths"][source],
        }
        if "soft_target_present" in self.payload and bool(self.payload["soft_target_present"][source]):
            item["soft_target"] = self.payload["soft_targets"][source].float()
        if "raw_frames" in self.payload:
            item["raw_frames"] = self.payload["raw_frames"][source]
        if "vgg_tokens" in self.payload:
            item["vgg_tokens"] = self.payload["vgg_tokens"][source].float()
        return item


__all__ = [
    "LGVQSingleMetricDataset",
    "ManifestRow",
    "cache_report",
    "file_sha256",
    "load_language_cache",
    "load_raw_frame_cache",
    "load_vgg_feature_cache",
    "load_single_metric_cache",
    "load_vision_cache",
    "read_manifest",
    "split_counts",
]
