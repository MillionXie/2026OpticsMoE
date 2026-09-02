from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch.utils.data import Dataset


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


def _first(row: Mapping[str, Any], names: Iterable[str], *, required: bool) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    if required:
        raise KeyError(f"None of the required columns exist: {tuple(names)}")
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest = Path(path).expanduser().resolve()
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"LGVQ manifest is empty: {manifest}")
    result: list[ManifestRow] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        video_path = str(_first(row, PATH_COLUMNS, required=True))
        sample_id_value = _first(row, ID_COLUMNS, required=False)
        sample_id = str(sample_id_value or Path(video_path).stem)
        if sample_id in seen:
            raise RuntimeError(f"Duplicate LGVQ sample id {sample_id!r}")
        seen.add(sample_id)
        split = str(_first(row, SPLIT_COLUMNS, required=True)).strip().lower()
        split = {"val": "validation", "valid": "validation", "dev": "validation"}.get(
            split, split
        )
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split {split!r} at manifest row {index + 2}")
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
    counts = {"train": 0, "validation": 0, "test": 0}
    for row in rows:
        counts[row.split] += 1
    return counts


def _find_tensor(payload: Any) -> torch.Tensor:
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, Mapping):
        preferred = (
            "frame_tokens",
            "features",
            "latents",
            "input_latents",
            "parallel_input_latents",
            "vision_hidden_states",
        )
        for key in preferred:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                return value
        tensors = [value for value in payload.values() if isinstance(value, torch.Tensor)]
        candidates = [value for value in tensors if value.ndim in (3, 4)]
        if len(candidates) == 1:
            return candidates[0]
    raise RuntimeError(
        "Could not identify a feature tensor. Expected one of frame_tokens/features/"
        "latents/input_latents/parallel_input_latents/vision_hidden_states."
    )


def _find_language_tensor(payload: Any) -> torch.Tensor | None:
    if not isinstance(payload, Mapping):
        return None
    for key in (
        "language_tokens",
        "text_hidden_states",
        "language_hidden_states",
        "prompt_hidden_states",
    ):
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def canonicalize_feature_tensor(
    value: torch.Tensor,
    *,
    sample_count: int,
    frame_count: int,
    token_count: int,
) -> torch.Tensor:
    """Return CPU [N,4,T,D] without silently truncating samples or tokens."""

    value = value.detach().cpu()
    if value.ndim == 3 and value.shape[0] == sample_count * frame_count:
        value = value.reshape(sample_count, frame_count, *value.shape[1:])
    if value.ndim != 4:
        raise ValueError(f"Feature cache must be [N,4,T,D], got {tuple(value.shape)}")
    expected = (sample_count, frame_count, token_count)
    if tuple(value.shape[:3]) != expected:
        raise ValueError(
            f"Feature cache prefix must be {expected}; got {tuple(value.shape[:3])}. "
            "No crop/repeat/reorder is permitted."
        )
    if value.shape[-1] <= 0 or not torch.isfinite(value.float()).all():
        raise ValueError("Feature cache width must be positive and values finite")
    return value.contiguous()


def build_canonical_cache(
    *,
    manifest_path: Path,
    source_feature_cache: Path,
    source_language_cache: Path | None,
    output_path: Path,
    frame_count: int,
    token_count: int,
    prompt: str,
    language_token_count: int,
) -> dict[str, Any]:
    rows = read_manifest(manifest_path)
    source = torch.load(source_feature_cache, map_location="cpu", weights_only=False)
    source_features = _find_tensor(source)
    source_ids = source.get("sample_ids") if isinstance(source, Mapping) else None
    if source_ids is None:
        raise RuntimeError(
            "Source cache must include sample_ids. MOS.txt and prompt_cls.json have "
            "different row order, so positional joining is forbidden."
        )
    source_index = {str(sample_id): index for index, sample_id in enumerate(source_ids)}
    desired_ids = [row.sample_id for row in rows]
    if len(source_index) != len(source_ids) or set(source_index) != set(desired_ids):
        raise RuntimeError("Source cache sample_ids do not match the manifest exactly")
    order = torch.tensor([source_index[sample_id] for sample_id in desired_ids])
    # Canonicalize flattened [N*4,T,D] sources before applying the N-sample
    # permutation. Indexing the flattened tensor with sample indices would
    # silently select only the first frame block.
    source_features = canonicalize_feature_tensor(
        source_features,
        sample_count=len(source_ids),
        frame_count=frame_count,
        token_count=token_count,
    )
    features = source_features.index_select(0, order).contiguous()
    language_source = (
        source
        if source_language_cache is None
        else torch.load(source_language_cache, map_location="cpu", weights_only=False)
    )
    language_tokens = _find_language_tensor(language_source)
    if language_tokens is None:
        raise RuntimeError(
            "No cached Qwen language hidden states were found. Formal training must "
            "cache the fixed five-level prompt through Qwen together with the four "
            "video frames; SPAQ auxiliary features are not a substitute."
        )
    language_tokens = language_tokens.detach().cpu()
    if language_tokens.ndim != 3 or language_tokens.shape[0] not in (1, len(rows)):
        raise ValueError(
            "language_tokens must be singleton [1,L,D] for the fixed prompt or "
            "[N,L,D] with the same manifest row order, got "
            f"{tuple(language_tokens.shape)}"
        )
    if language_tokens.shape[1] > language_token_count:
        raise ValueError(
            f"Cached language length {language_tokens.shape[1]} exceeds configured "
            f"maximum {language_token_count}; no silent crop is allowed"
        )
    if language_tokens.shape[0] == len(rows):
        language_ids = (
            language_source.get("sample_ids")
            if isinstance(language_source, Mapping)
            else None
        )
        if language_ids is None:
            if language_source is not source:
                raise RuntimeError(
                    "A per-sample separate language cache must include sample_ids"
                )
            language_order = order
        else:
            language_index = {
                str(sample_id): index for index, sample_id in enumerate(language_ids)
            }
            if len(language_index) != len(language_ids) or set(language_index) != set(desired_ids):
                raise RuntimeError("Language-cache sample_ids do not match manifest")
            language_order = torch.tensor(
                [language_index[sample_id] for sample_id in desired_ids]
            )
        language_tokens = language_tokens.index_select(0, language_order)
    else:
        language_order = None
    language_mask = None
    if isinstance(language_source, Mapping):
        candidate_mask = language_source.get("language_mask")
        if not isinstance(candidate_mask, torch.Tensor):
            candidate_mask = language_source.get("attention_mask")
        if isinstance(candidate_mask, torch.Tensor):
            language_mask = candidate_mask.detach().cpu().bool()
    if language_mask is None:
        language_mask = torch.ones(language_tokens.shape[:2], dtype=torch.bool)
    elif language_mask.shape[0] == len(rows) and language_order is not None:
        language_mask = language_mask.index_select(0, language_order)
    if tuple(language_mask.shape) != tuple(language_tokens.shape[:2]):
        raise ValueError("Source language mask shape does not match language tokens")
    payload = {
        "schema_version": 1,
        "feature_contract": "qwen3vl_four_frames_no_crop_v1",
        "frame_tokens": features,
        "language_tokens": language_tokens.contiguous(),
        "language_mask": language_mask,
        "sample_ids": [row.sample_id for row in rows],
        "video_paths": [row.video_path for row in rows],
        "splits": [row.split for row in rows],
        "targets": torch.tensor(
            [[row.spatial, row.temporal] for row in rows], dtype=torch.float32
        ),
        "target_names": ["spatial", "temporal"],
        "alignment_target_present": False,
        "qwen_prompt": prompt,
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "source_feature_cache": str(source_feature_cache),
        "source_feature_cache_sha256": file_sha256(source_feature_cache),
        "source_language_cache": (
            None if source_language_cache is None else str(source_language_cache)
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return cache_report(payload, output_path)


def load_canonical_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path).expanduser().resolve()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise RuntimeError(f"Unsupported canonical LGVQ cache schema: {cache_path}")
    required = {
        "frame_tokens",
        "language_tokens",
        "language_mask",
        "sample_ids",
        "video_paths",
        "splits",
        "targets",
        "target_names",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Canonical cache is missing {sorted(missing)}")
    features = payload["frame_tokens"]
    language_tokens = payload["language_tokens"]
    language_mask = payload["language_mask"]
    targets = payload["targets"]
    if not isinstance(features, torch.Tensor) or features.ndim != 4:
        raise ValueError("frame_tokens must be a tensor [N,4,T,D]")
    count = features.shape[0]
    if not isinstance(language_tokens, torch.Tensor) or language_tokens.ndim != 3:
        raise ValueError("language_tokens must be [N,L,D]")
    if tuple(language_mask.shape) != tuple(language_tokens.shape[:2]):
        raise ValueError("language_mask must be [N,L]")
    if language_tokens.shape[0] not in (1, count):
        raise ValueError("Language cache must be singleton fixed prompt or per sample")
    if tuple(targets.shape) != (count, 2):
        raise ValueError("targets must be [N,2] in spatial,temporal order")
    if list(payload["target_names"]) != ["spatial", "temporal"]:
        raise ValueError("Only spatial and temporal targets are allowed")
    if any(len(payload[key]) != count for key in ("sample_ids", "video_paths", "splits")):
        raise ValueError("Canonical cache metadata lengths do not match frame_tokens")
    if payload.get("alignment_target_present", False):
        raise ValueError("Alignment values must not be stored in the canonical cache")
    return payload


def cache_report(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    features = payload["frame_tokens"]
    return {
        "path": str(path),
        "shape": list(features.shape),
        "language_shape": list(payload["language_tokens"].shape),
        "dtype": str(features.dtype),
        "split_counts": {
            split: list(payload["splits"]).count(split)
            for split in ("train", "validation", "test")
        },
        "targets": ["spatial", "temporal"],
        "alignment_excluded": True,
        "prompt": payload.get("qwen_prompt"),
    }


class LGVQFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(self, payload: Mapping[str, Any], split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split {split!r}")
        self.payload = payload
        self.indices = [
            index for index, value in enumerate(payload["splits"]) if value == split
        ]
        if not self.indices:
            raise RuntimeError(f"LGVQ cache has no samples in split={split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source = self.indices[index]
        return {
            "features": self.payload["frame_tokens"][source].float(),
            "language_tokens": self.payload["language_tokens"][
                0 if self.payload["language_tokens"].shape[0] == 1 else source
            ].float(),
            "language_mask": self.payload["language_mask"][
                0 if self.payload["language_mask"].shape[0] == 1 else source
            ].bool(),
            "target": self.payload["targets"][source].float(),
            "sample_id": self.payload["sample_ids"][source],
            "video_path": self.payload["video_paths"][source],
        }


__all__ = [
    "LGVQFeatureDataset",
    "ManifestRow",
    "build_canonical_cache",
    "cache_report",
    "canonicalize_feature_tensor",
    "file_sha256",
    "load_canonical_cache",
    "read_manifest",
    "split_counts",
]
