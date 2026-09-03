from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F

from .data import (
    QWEN_COMPONENT_FINGERPRINT_CONTRACT,
    QWEN_FRONT_PAIR_CONTRACT,
    QWEN_SOURCE_IDENTITY_CONTRACT,
    _canonical_record_sha256,
    file_sha256,
    read_manifest,
)
from .prepare_manifest import prepare_manifest
from .settings import (
    FEATURE_CONTRACT,
    LANGUAGE_CONTRACT,
    QUALITY_CONTRACT,
    TARGET_PROMPTS,
    load_settings,
)


FRAME_COUNT = 16
FRAME_FRACTIONS = tuple(0.10 + index * 0.80 / (FRAME_COUNT - 1) for index in range(FRAME_COUNT))
PREMERGER_GRID = 28
PREMERGER_TOKENS = PREMERGER_GRID * PREMERGER_GRID
QWEN_MERGE_GRID = 14
QWEN_MERGED_TOKENS = QWEN_MERGE_GRID * QWEN_MERGE_GRID
OUTPUT_GRID = 7
OUTPUT_TOKENS = OUTPUT_GRID * OUTPUT_GRID
VISION_WIDTH = 1024
LANGUAGE_WIDTH = 2048
PART_SCHEMA_VERSION = 1
FINGERPRINT_CHUNK_BYTES = 8 * 1024 * 1024


def _hashed_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    if "sha256" in record:
        raise ValueError("Hashed-record payload must not supply its own sha256")
    record["sha256"] = _canonical_record_sha256(record)
    return record


def _tensor_stream_sha256(
    tensor: torch.Tensor, *, chunk_bytes: int = FINGERPRINT_CHUNK_BYTES
) -> str:
    """Hash a tensor without materializing the whole tensor on CPU.

    ``embed_tokens`` is hundreds of MiB.  Hashing row chunks avoids the former
    failure mode where ``.cpu().numpy().tobytes()`` temporarily duplicated the
    entire embedding matrix.  Shape and dtype are included so equal raw bytes
    with a different interpretation cannot collide at the component level.
    """

    if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
        raise TypeError("Only dense strided tensors can be fingerprinted")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    value = tensor.detach()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    if value.ndim == 0:
        chunks = (value.reshape(1),)
    else:
        elements_per_row = max(1, int(value[0].numel()) if value.shape[0] else 1)
        element_size = max(1, int(value.element_size()))
        rows_per_chunk = max(1, chunk_bytes // (elements_per_row * element_size))
        chunks = (
            value[start : start + rows_per_chunk]
            for start in range(0, int(value.shape[0]), rows_per_chunk)
        )
    for chunk in chunks:
        # view(uint8) preserves bfloat16 bit patterns and avoids NumPy's lack
        # of a native bfloat16 dtype.  At most ``chunk_bytes`` is copied from a
        # CUDA tensor at a time.
        raw = chunk.to(device="cpu").contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _tensor_descriptor(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": str(name),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": _tensor_stream_sha256(tensor),
    }


def _config_mapping(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        raw = config.to_dict()
    elif isinstance(config, Mapping):
        raw = dict(config)
    else:
        raw = {
            key: value
            for key, value in vars(config).items()
            if not str(key).startswith("_")
        }
    # The canonical JSON round-trip both verifies serializability and removes
    # incidental mapping subclasses from transformers configs.
    return json.loads(json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str))


def _checkpoint_revision(model: torch.nn.Module, model_path: Path) -> str | None:
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "config", None), "vision_config", None),
    ):
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
    parts = model_path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return str(parts[index + 1])
    return None


def _checkpoint_artifact_manifest(model_path: Path) -> dict[str, Any]:
    """Build a relocation-stable, lightweight checkpoint file manifest."""

    if not model_path.is_dir():
        return {"files": [], "note": "model_path_is_not_a_directory"}
    lightweight_names = {
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    }
    records: list[dict[str, Any]] = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(model_path).as_posix()
        lower = path.name.lower()
        is_weight = lower.endswith((".safetensors", ".bin"))
        is_index = lower.endswith(".index.json")
        is_lightweight = lower in lightweight_names or is_index
        if not (is_weight or is_lightweight):
            continue
        record: dict[str, Any] = {
            "path": relative,
            "size_bytes": int(path.stat().st_size),
        }
        # Hash small contract/index files.  Gigabyte weight shards are not
        # duplicated or reread here: the exact front tensors are content
        # hashed below, and their hashes are bound into the pair identity.
        if is_lightweight:
            record["sha256"] = file_sha256(path)
        records.append(record)
    return {"files": records}


def _source_identity(
    model: torch.nn.Module,
    *,
    model_path: Path,
    transformers_version: str,
) -> dict[str, Any]:
    config = _config_mapping(getattr(model, "config", None))
    config_bytes = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    artifacts = _checkpoint_artifact_manifest(model_path)
    artifact_bytes = json.dumps(
        artifacts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _hashed_record(
        {
            "contract": QWEN_SOURCE_IDENTITY_CONTRACT,
            "requested_model": "Qwen/Qwen3-VL-2B-Instruct",
            "resolved_model_path": str(model_path),
            "checkpoint_revision": _checkpoint_revision(model, model_path),
            "model_class": type(model).__qualname__,
            "transformers_version": str(transformers_version),
            "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "checkpoint_artifact_manifest_sha256": hashlib.sha256(
                artifact_bytes
            ).hexdigest(),
            "checkpoint_artifacts": artifacts,
        }
    )


def _vision_front_fingerprint(
    visual: torch.nn.Module,
    *,
    source_identity_sha256: str,
    vision_config: Any | None = None,
) -> dict[str, Any]:
    named = list(visual.named_parameters()) + list(visual.named_buffers())
    selected: list[tuple[str, torch.Tensor]] = []
    seen: set[str] = set()
    for name, tensor in named:
        canonical = str(name)
        is_patch = canonical == "patch_embed" or canonical.startswith("patch_embed.")
        is_position = (
            canonical == "pos_embed"
            or canonical.startswith("pos_embed.")
            or "position_embedding" in canonical
        )
        if (is_patch or is_position) and canonical not in seen:
            selected.append((canonical, tensor))
            seen.add(canonical)
    if not any(name.startswith("patch_embed") for name, _ in selected):
        raise RuntimeError("Vision fingerprint found no patch_embed tensors")
    if not any(
        name.startswith("pos_embed") or "position_embedding" in name
        for name, _ in selected
    ):
        raise RuntimeError("Vision fingerprint found no learned position-embedding tensors")
    vision_config_payload = _config_mapping(
        vision_config if vision_config is not None else getattr(visual, "config", None)
    )
    try:
        position_method_source = inspect.getsource(visual.fast_pos_embed_interpolate)
    except (OSError, TypeError):
        position_method_source = repr(visual.fast_pos_embed_interpolate)
    return _hashed_record(
        {
            "contract": QWEN_COMPONENT_FINGERPRINT_CONTRACT,
            "component": "vision_patch_embed_and_position",
            "source_identity_sha256": source_identity_sha256,
            "vision_config": vision_config_payload,
            "position_interpolation_code_sha256": hashlib.sha256(
                position_method_source.encode("utf-8")
            ).hexdigest(),
            "tensors": [
                _tensor_descriptor(name, tensor)
                for name, tensor in sorted(selected, key=lambda pair: pair[0])
            ],
        }
    )


def _language_embedding_fingerprint(
    embed_tokens: torch.nn.Module,
    *,
    source_identity_sha256: str,
    language_config: Any,
) -> dict[str, Any]:
    tensors = list(embed_tokens.named_parameters()) + list(embed_tokens.named_buffers())
    if not tensors:
        raise RuntimeError("Language fingerprint found no embed_tokens tensors")
    return _hashed_record(
        {
            "contract": QWEN_COMPONENT_FINGERPRINT_CONTRACT,
            "component": "language_embed_tokens",
            "source_identity_sha256": source_identity_sha256,
            "language_config": _config_mapping(language_config),
            "tensors": [
                _tensor_descriptor(name, tensor)
                for name, tensor in sorted(tensors, key=lambda pair: pair[0])
            ],
        }
    )


def _front_pair_identity(
    *,
    source_identity: Mapping[str, Any],
    vision_fingerprint: Mapping[str, Any],
    language_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    return _hashed_record(
        {
            "contract": QWEN_FRONT_PAIR_CONTRACT,
            "source_identity_sha256": source_identity["sha256"],
            "vision_front_sha256": vision_fingerprint["sha256"],
            "language_embedding_sha256": language_fingerprint["sha256"],
        }
    )


def _sample_positions(frame_total: int) -> tuple[int, ...]:
    """Return the fixed 16 temporal landmarks in the central 10--90% span."""

    if frame_total <= 0:
        raise ValueError("frame_total must be positive")
    return tuple(
        min(frame_total - 1, max(0, round((frame_total - 1) * fraction)))
        for fraction in FRAME_FRACTIONS
    )


def decode_sixteen_frames(path: Path) -> list[Image.Image]:
    """Decode 16 uniformly stratified central-time frames and center crop them.

    The 0.10--0.90 landmarks avoid unstable first/last decoder frames.  The
    same deterministic 65% short-side crop used by the audited LGVQ pipeline
    is resized to 448x448 before the official Qwen processor sees it.
    """

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required to decode LGVQ MP4 files") from error
    path = Path(path).expanduser().resolve()
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {path}")
    frames: list[Image.Image] = []
    try:
        for position in _sample_positions(count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to decode frame {position} from {path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            side = max(2, round(min(height, width) * 0.65))
            top, left = (height - side) // 2, (width - side) // 2
            square = cv2.resize(
                rgb[top : top + side, left : left + side],
                (448, 448),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(Image.fromarray(square))
    finally:
        capture.release()
    if len(frames) != FRAME_COUNT:
        raise RuntimeError(f"Expected {FRAME_COUNT} decoded frames, got {len(frames)}")
    return frames


def qwen_patch_with_position(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Run only frozen Qwen patch embedding plus official position embedding."""

    if not hasattr(visual, "patch_embed"):
        raise RuntimeError("Qwen visual module does not expose patch_embed")
    if not hasattr(visual, "fast_pos_embed_interpolate"):
        raise RuntimeError(
            "Qwen visual.fast_pos_embed_interpolate is required for the official "
            "position embedding path"
        )
    hidden = visual.patch_embed(pixel_values)
    positional = visual.fast_pos_embed_interpolate(grid_thw)
    if tuple(hidden.shape) != tuple(positional.shape):
        raise RuntimeError(
            f"Patch and position shapes differ: {tuple(hidden.shape)} vs {tuple(positional.shape)}"
        )
    return hidden + positional.to(device=hidden.device, dtype=hidden.dtype)


def pool_qwen_front_tokens(hidden: torch.Tensor, *, image_count: int) -> torch.Tensor:
    """Map official 784 patch tokens to a target-neutral 7x7 front cache.

    Qwen's processor places each spatial 2x2 merge group contiguously.  The
    first mean therefore maps 784x1024 to the Qwen merger's 14x14 token order,
    without calling its learned merger.  A second spatial 2x2 mean maps 14x14
    to 7x7.  No Vision block, merger, attention, or trainable operation runs.
    """

    expected = int(image_count) * PREMERGER_TOKENS
    if hidden.ndim != 2 or hidden.shape != (expected, VISION_WIDTH):
        raise RuntimeError(
            f"Expected [{expected},{VISION_WIDTH}] patch+position tokens, got {tuple(hidden.shape)}"
        )
    merged = hidden.reshape(image_count, QWEN_MERGED_TOKENS, 4, VISION_WIDTH).mean(2)
    grid = merged.reshape(image_count, QWEN_MERGE_GRID, QWEN_MERGE_GRID, VISION_WIDTH)
    pooled = grid.reshape(
        image_count,
        OUTPUT_GRID,
        2,
        OUTPUT_GRID,
        2,
        VISION_WIDTH,
    ).mean((2, 4))
    return pooled.reshape(image_count, OUTPUT_TOKENS, VISION_WIDTH).contiguous()


def quality_tokens_from_images(
    images: Sequence[Image.Image], *, video_count: int
) -> torch.Tensor:
    """Create the fixed 14-channel quality side input at the same 7x7 grid.

    This deterministic bank complements rather than replaces Qwen patch and
    position embeddings.  It has no learned parameters and is cached so the
    training graph does not repeatedly calculate Sobel/local-statistics maps.
    """

    if len(images) != int(video_count) * FRAME_COUNT:
        raise ValueError("images must contain exactly 16 frames per video")
    arrays = [
        torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy())
        .permute(2, 0, 1)
        for image in images
    ]
    spatial_shapes = {tuple(value.shape[-2:]) for value in arrays}
    if len(spatial_shapes) != 1:
        raise ValueError("All decoded frames must have the same spatial size")
    height, width = next(iter(spatial_shapes))
    frames = torch.stack(arrays).reshape(
        video_count, FRAME_COUNT, 3, height, width
    )
    rgb = frames.float().div(255.0)
    luminance = (
        0.2989 * rgb[:, :, 0:1]
        + 0.5870 * rgb[:, :, 1:2]
        + 0.1140 * rgb[:, :, 2:3]
    )
    flat = luminance.flatten(0, 1)
    sobel_x_kernel = torch.tensor(
        ((-1.0, 0.0, 1.0), (-2.0, 0.0, 2.0), (-1.0, 0.0, 1.0))
    ).view(1, 1, 3, 3) / 4.0
    sobel_y_kernel = sobel_x_kernel.transpose(-1, -2).contiguous()
    laplacian_kernel = torch.tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
    ).view(1, 1, 3, 3) / 4.0
    padded3 = F.pad(flat, (1, 1, 1, 1), mode="reflect")
    sobel_x = F.conv2d(padded3, sobel_x_kernel)
    sobel_y = F.conv2d(padded3, sobel_y_kernel)
    gradient = torch.sqrt(sobel_x.square() + sobel_y.square() + 1.0e-12)
    laplacian = F.conv2d(padded3, laplacian_kernel).abs()
    padded5 = F.pad(flat, (2, 2, 2, 2), mode="reflect")
    local_mean = F.avg_pool2d(padded5, 5, stride=1)
    local_square_mean = F.avg_pool2d(padded5.square(), 5, stride=1)
    local_std = (local_square_mean - local_mean.square()).clamp_min(0.0).sqrt()
    shape = (video_count, FRAME_COUNT, 1, height, width)
    saturation = rgb.amax(2, keepdim=True) - rgb.amin(2, keepdim=True)
    temporal = torch.zeros_like(luminance)
    temporal[:, 1:] = (luminance[:, 1:] - luminance[:, :-1]).abs()
    y = torch.linspace(-1.0, 1.0, height).view(1, 1, 1, height, 1).expand(
        video_count, FRAME_COUNT, 1, height, width
    )
    x = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, 1, width).expand(
        video_count, FRAME_COUNT, 1, height, width
    )
    time = torch.linspace(-1.0, 1.0, FRAME_COUNT).view(
        1, FRAME_COUNT, 1, 1, 1
    ).expand(video_count, FRAME_COUNT, 1, height, width)
    channels = torch.cat(
        (
            rgb,
            luminance,
            sobel_x.reshape(shape),
            sobel_y.reshape(shape),
            gradient.reshape(shape),
            laplacian.reshape(shape),
            local_std.reshape(shape),
            saturation,
            temporal,
            x,
            y,
            time,
        ),
        2,
    )
    if channels.shape[2] != 14:
        raise RuntimeError(f"Quality bank must have 14 channels, got {channels.shape[2]}")
    pooled = F.adaptive_avg_pool2d(channels.flatten(0, 1), (OUTPUT_GRID, OUTPUT_GRID))
    return (
        pooled.reshape(video_count, FRAME_COUNT, 14, OUTPUT_TOKENS)
        .permute(0, 1, 3, 2)
        .half()
        .contiguous()
    )


def render_prompt(tokenizer: Any, *, target_name: str, prompt: str) -> str:
    expected = TARGET_PROMPTS.get(target_name)
    if expected is None:
        raise ValueError(f"Unknown target_name={target_name!r}")
    if prompt.strip() != expected:
        raise ValueError(f"Prompt does not match the {target_name} target contract")
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def _resolve_embed_tokens(model: torch.nn.Module) -> torch.nn.Module:
    core = getattr(model, "model", model)
    language = getattr(core, "language_model", None)
    candidates = (
        getattr(language, "embed_tokens", None),
        getattr(getattr(language, "model", None), "embed_tokens", None),
        getattr(core, "embed_tokens", None),
        model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None,
    )
    for candidate in candidates:
        if isinstance(candidate, torch.nn.Module):
            weight = getattr(candidate, "weight", None)
            if torch.is_tensor(weight) and weight.ndim == 2 and weight.shape[1] == LANGUAGE_WIDTH:
                return candidate
    raise RuntimeError("Could not locate Qwen Language embed_tokens with width 2048")


def _sample_order_sha256(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _model_contract(model: torch.nn.Module, visual: torch.nn.Module, model_path: Path) -> dict[str, Any]:
    evidence = "\n".join(
        str(value)
        for value in (
            model_path,
            getattr(model.config, "_name_or_path", ""),
            getattr(model.config, "name_or_path", ""),
        )
    ).lower().replace("_", "-")
    if "embedding" in evidence:
        raise RuntimeError("Use Qwen3-VL-2B-Instruct, not the Embedding checkpoint")
    vision_config = getattr(model.config, "vision_config", getattr(visual, "config", None))
    text_config = getattr(model.config, "text_config", None)
    vision_width = int(getattr(vision_config, "hidden_size", -1))
    language_width = int(getattr(text_config, "hidden_size", -1))
    merge_size = int(getattr(vision_config, "spatial_merge_size", -1))
    if (vision_width, language_width, merge_size) != (VISION_WIDTH, LANGUAGE_WIDTH, 2):
        raise RuntimeError(
            "Qwen architecture contract changed: expected Vision/Language/merge "
            f"1024/2048/2, got {vision_width}/{language_width}/{merge_size}"
        )
    return {
        "requested_model": "Qwen/Qwen3-VL-2B-Instruct",
        "resolved_model_path": str(model_path),
        "vision_hidden_size": vision_width,
        "language_hidden_size": language_width,
        "spatial_merge_size": merge_size,
        "vision_blocks_executed": 0,
        "vision_merger_executed": False,
        "language_blocks_executed": 0,
        "lm_head_executed": False,
    }


def _part_path(parts_dir: Path, start: int, stop: int) -> Path:
    return parts_dir / f"vision_{start:06d}_{stop:06d}.pt"


def _load_part(
    path: Path,
    *,
    start: int,
    stop: int,
    sample_ids: Sequence[str],
    manifest_sha256: str,
    source_identity_sha256: str,
    vision_front_sha256: str,
    front_pair_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("feature_contract") != FEATURE_CONTRACT
            or payload.get("quality_contract") != QUALITY_CONTRACT
            or int(payload.get("start", -1)) != start
            or int(payload.get("stop", -1)) != stop
            or list(payload.get("sample_ids", [])) != list(sample_ids)
            or payload.get("manifest_sha256") != manifest_sha256
            or payload.get("qwen_source_identity_sha256")
            != source_identity_sha256
            or payload.get("qwen_vision_front_sha256") != vision_front_sha256
            or payload.get("qwen_front_pair_sha256") != front_pair_sha256
        ):
            return None
        value = payload.get("vision_tokens")
        quality = payload.get("quality_tokens")
        shape = (stop - start, FRAME_COUNT, OUTPUT_TOKENS, VISION_WIDTH)
        quality_shape = (stop - start, FRAME_COUNT, OUTPUT_TOKENS, 14)
        if not torch.is_tensor(value) or value.dtype != torch.float16 or tuple(value.shape) != shape:
            return None
        if not torch.is_tensor(quality) or quality.dtype != torch.float16 or tuple(quality.shape) != quality_shape:
            return None
        if not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(quality).all()):
            return None
        return value.contiguous(), quality.contiguous()
    except Exception:
        return None


def _vision_cache_identity_matches(
    path: Path,
    *,
    source_identity: Mapping[str, Any],
    vision_fingerprint: Mapping[str, Any],
    front_pair_identity: Mapping[str, Any],
    model_contract: Mapping[str, Any],
) -> bool:
    """Check provenance stored inside the large cache without copying tensors."""

    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
        "mmap": True,
    }
    try:
        try:
            payload = torch.load(path, **kwargs)
        except TypeError:
            kwargs.pop("mmap", None)
            payload = torch.load(path, **kwargs)
        return bool(
            isinstance(payload, Mapping)
            and payload.get("qwen_source_identity") == dict(source_identity)
            and payload.get("qwen_vision_front_fingerprint")
            == dict(vision_fingerprint)
            and payload.get("qwen_front_pair_identity") == dict(front_pair_identity)
            and payload.get("qwen_front_contract") == dict(model_contract)
        )
    except Exception:
        return False


def _extract_vision_rows(
    *,
    rows: Sequence[Any],
    processor: Any,
    visual: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    result: list[torch.Tensor] = []
    quality_result: list[torch.Tensor] = []
    visual_dtype = next(visual.patch_embed.parameters()).dtype
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = [
            frame
            for row in batch_rows
            for frame in decode_sixteen_frames(Path(row.video_path))
        ]
        # This is the same processor boundary used by the audited four-frame
        # Qwen cache.  Qwen3VLProcessor 4.57 requires non-None text during
        # image collation, hence one disposable "x" per image.  Its input_ids
        # are never consumed here; only pixel_values/image_grid_thw enter the
        # frozen patch+position front, so no image-placeholder/model-forward
        # correspondence is required or silently relied upon.
        processed = processor(
            text=["x"] * len(images),
            images=images,
            return_tensors="pt",
        )
        quality_result.append(
            quality_tokens_from_images(images, video_count=len(batch_rows))
        )
        pixel_values = processed["pixel_values"].to(device=device, dtype=visual_dtype)
        grid_thw = processed["image_grid_thw"].to(device)
        expected_grid = torch.tensor([1, PREMERGER_GRID, PREMERGER_GRID], device=device)
        if grid_thw.shape != (len(images), 3) or not bool(torch.all(grid_thw == expected_grid)):
            raise RuntimeError(
                "Official Qwen processor contract changed; expected one [1,28,28] "
                f"grid per frame, got {grid_thw.detach().cpu().tolist()}"
            )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            hidden = qwen_patch_with_position(visual, pixel_values, grid_thw)
            pooled = pool_qwen_front_tokens(hidden, image_count=len(images))
        result.append(
            pooled.reshape(len(batch_rows), FRAME_COUNT, OUTPUT_TOKENS, VISION_WIDTH)
            .detach()
            .cpu()
            .half()
            .contiguous()
        )
    return torch.cat(result, dim=0), torch.cat(quality_result, dim=0)


def _cache_language(
    *,
    tokenizer: Any,
    embed_tokens: torch.nn.Module,
    device: torch.device,
    target_name: str,
    prompt: str,
    output: Path,
    model_contract: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    language_fingerprint: Mapping[str, Any],
    front_pair_identity: Mapping[str, Any],
) -> dict[str, Any]:
    rendered = render_prompt(tokenizer, target_name=target_name, prompt=prompt)
    encoded = tokenizer(
        [rendered],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    embed_dtype = next(embed_tokens.parameters()).dtype
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        tokens = embed_tokens(input_ids).to(embed_dtype)
    tokens = tokens.detach().cpu().half().contiguous()
    input_ids = input_ids.detach().cpu().long().contiguous()
    attention_mask = attention_mask.detach().cpu().bool().contiguous()
    if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[-1] != LANGUAGE_WIDTH:
        raise RuntimeError(f"Unexpected Qwen embed_tokens output {tuple(tokens.shape)}")
    payload = {
        "schema_version": 1,
        "language_contract": LANGUAGE_CONTRACT,
        "target_name": target_name,
        "prompt": prompt,
        "rendered_chat_template": rendered,
        "language_tokens": tokens,
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "broadcast": "singleton target-specific prompt is broadcast to every video",
        "qwen_front_contract": dict(model_contract),
        "qwen_source_identity": dict(source_identity),
        "qwen_language_embedding_fingerprint": dict(language_fingerprint),
        "qwen_front_pair_identity": dict(front_pair_identity),
    }
    _atomic_torch_save(output, payload)
    return {
        "path": str(output),
        "target_name": target_name,
        "shape": list(tokens.shape),
        "token_count": int(tokens.shape[1]),
        "dtype": str(tokens.dtype),
        "qwen_source_identity_sha256": source_identity["sha256"],
        "qwen_language_embedding_sha256": language_fingerprint["sha256"],
        "qwen_front_pair_sha256": front_pair_identity["sha256"],
    }


def build_cache(
    *,
    dataset_root: Path,
    model_path: Path,
    vision_output: Path,
    language_output: Path,
    target_name: str,
    manifest: Path | None,
    batch_size: int,
    chunk_rows: int,
    device_name: str,
    overwrite_vision: bool = False,
) -> dict[str, Any]:
    if target_name not in TARGET_PROMPTS:
        raise ValueError(f"target_name must be one of {sorted(TARGET_PROMPTS)}")
    if batch_size <= 0 or chunk_rows <= 0:
        raise ValueError("batch_size and chunk_rows must be positive")
    try:
        import transformers
    except ImportError as error:
        raise RuntimeError("transformers with Qwen3-VL support is required") from error

    dataset_root = dataset_root.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    vision_output = vision_output.expanduser().resolve()
    language_output = language_output.expanduser().resolve()
    manifest = vision_output.with_suffix(".manifest.csv") if manifest is None else manifest.expanduser().resolve()
    if not manifest.exists():
        prepare_manifest(dataset_root, manifest)
    rows = read_manifest(manifest)
    sample_ids = [row.sample_id for row in rows]
    manifest_digest = file_sha256(manifest)
    sample_order_digest = _sample_order_sha256(sample_ids)
    device = torch.device(
        device_name
        if not device_name.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )

    processor = transformers.AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=448 * 448,
        max_pixels=448 * 448,
        local_files_only=True,
        trust_remote_code=True,
    )
    model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
    if model_cls is None:
        model_cls = transformers.AutoModelForImageTextToText
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = (
        model_cls.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .requires_grad_(False)
        .eval()
    )
    core = getattr(model, "model", model)
    visual = getattr(core, "visual", None)
    if visual is None:
        raise RuntimeError("Loaded Qwen model does not expose model.visual")
    visual.requires_grad_(False).eval()
    embed_tokens = _resolve_embed_tokens(model).requires_grad_(False).eval()
    model_contract = _model_contract(model, visual, model_path)
    source_identity = _source_identity(
        model,
        model_path=model_path,
        transformers_version=str(getattr(transformers, "__version__", "unknown")),
    )
    vision_fingerprint = _vision_front_fingerprint(
        visual,
        source_identity_sha256=source_identity["sha256"],
        vision_config=getattr(model.config, "vision_config", None),
    )
    language_fingerprint = _language_embedding_fingerprint(
        embed_tokens,
        source_identity_sha256=source_identity["sha256"],
        language_config=getattr(model.config, "text_config", None),
    )
    front_pair_identity = _front_pair_identity(
        source_identity=source_identity,
        vision_fingerprint=vision_fingerprint,
        language_fingerprint=language_fingerprint,
    )

    language_report = _cache_language(
        tokenizer=processor.tokenizer,
        embed_tokens=embed_tokens,
        device=device,
        target_name=target_name,
        prompt=TARGET_PROMPTS[target_name],
        output=language_output,
        model_contract=model_contract,
        source_identity=source_identity,
        language_fingerprint=language_fingerprint,
        front_pair_identity=front_pair_identity,
    )

    metadata_path = vision_output.with_suffix(vision_output.suffix + ".json")
    reusable = False
    if vision_output.is_file() and metadata_path.is_file() and not overwrite_vision:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            reusable = (
                metadata.get("feature_contract") == FEATURE_CONTRACT
                and metadata.get("manifest_sha256") == manifest_digest
                and metadata.get("sample_order_sha256") == sample_order_digest
                and metadata.get("shape") == [len(rows), FRAME_COUNT, OUTPUT_TOKENS, VISION_WIDTH]
                and metadata.get("quality_contract") == QUALITY_CONTRACT
                and metadata.get("quality_shape") == [len(rows), FRAME_COUNT, OUTPUT_TOKENS, 14]
                and int(metadata.get("file_size_bytes", -1)) == vision_output.stat().st_size
                and metadata.get("qwen_source_identity_sha256")
                == source_identity["sha256"]
                and metadata.get("qwen_vision_front_sha256")
                == vision_fingerprint["sha256"]
                and metadata.get("qwen_front_pair_sha256")
                == front_pair_identity["sha256"]
                and _vision_cache_identity_matches(
                    vision_output,
                    source_identity=source_identity,
                    vision_fingerprint=vision_fingerprint,
                    front_pair_identity=front_pair_identity,
                    model_contract=model_contract,
                )
            )
        except Exception:
            reusable = False

    vision_report: dict[str, Any]
    if reusable:
        vision_report = {
            "path": str(vision_output),
            "reused": True,
            "shape": [len(rows), FRAME_COUNT, OUTPUT_TOKENS, VISION_WIDTH],
            "quality_shape": [len(rows), FRAME_COUNT, OUTPUT_TOKENS, 14],
            "dtype": "torch.float16",
            "qwen_source_identity_sha256": source_identity["sha256"],
            "qwen_vision_front_sha256": vision_fingerprint["sha256"],
            "qwen_front_pair_sha256": front_pair_identity["sha256"],
        }
    else:
        parts_dir = Path(str(vision_output) + ".parts")
        parts_dir.mkdir(parents=True, exist_ok=True)
        part_specs: list[tuple[Path, int, int]] = []
        for start in range(0, len(rows), chunk_rows):
            stop = min(len(rows), start + chunk_rows)
            path = _part_path(parts_dir, start, stop)
            loaded = _load_part(
                path,
                start=start,
                stop=stop,
                sample_ids=sample_ids[start:stop],
                manifest_sha256=manifest_digest,
                source_identity_sha256=source_identity["sha256"],
                vision_front_sha256=vision_fingerprint["sha256"],
                front_pair_sha256=front_pair_identity["sha256"],
            )
            if loaded is None:
                value, quality = _extract_vision_rows(
                    rows=rows[start:stop],
                    processor=processor,
                    visual=visual,
                    device=device,
                    batch_size=batch_size,
                )
                _atomic_torch_save(
                    path,
                    {
                        "schema_version": PART_SCHEMA_VERSION,
                        "feature_contract": FEATURE_CONTRACT,
                        "start": start,
                        "stop": stop,
                        "sample_ids": sample_ids[start:stop],
                        "manifest_sha256": manifest_digest,
                        "qwen_source_identity_sha256": source_identity["sha256"],
                        "qwen_vision_front_sha256": vision_fingerprint["sha256"],
                        "qwen_front_pair_sha256": front_pair_identity["sha256"],
                        "vision_tokens": value,
                        "quality_contract": QUALITY_CONTRACT,
                        "quality_tokens": quality,
                    },
                )
                print(f"[qwen-front] saved {stop}/{len(rows)} -> {path.name}", flush=True)
            else:
                print(f"[qwen-front] resumed {stop}/{len(rows)} <- {path.name}", flush=True)
            part_specs.append((path, start, stop))

        # The full Qwen checkpoint is no longer needed.  Release it before the
        # approximately 4.5-GiB contiguous target-neutral cache is assembled.
        del embed_tokens, core, visual, model, processor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        features = torch.empty(
            len(rows), FRAME_COUNT, OUTPUT_TOKENS, VISION_WIDTH, dtype=torch.float16
        )
        quality_features = torch.empty(
            len(rows), FRAME_COUNT, OUTPUT_TOKENS, 14, dtype=torch.float16
        )
        for path, start, stop in part_specs:
            loaded = _load_part(
                path,
                start=start,
                stop=stop,
                sample_ids=sample_ids[start:stop],
                manifest_sha256=manifest_digest,
                source_identity_sha256=source_identity["sha256"],
                vision_front_sha256=vision_fingerprint["sha256"],
                front_pair_sha256=front_pair_identity["sha256"],
            )
            if loaded is None:
                raise RuntimeError(f"Vision shard became invalid during assembly: {path}")
            value, quality = loaded
            features[start:stop].copy_(value)
            quality_features[start:stop].copy_(quality)
        payload = {
            "schema_version": 1,
            "feature_contract": FEATURE_CONTRACT,
            "quality_contract": QUALITY_CONTRACT,
            "vision_tokens": features,
            "quality_tokens": quality_features,
            "quality_channel_order": [
                "R",
                "G",
                "B",
                "luminance",
                "sobel_x",
                "sobel_y",
                "gradient_magnitude",
                "absolute_laplacian",
                "local_std_5x5",
                "saturation",
                "previous_frame_luminance_abs_difference",
                "x_coordinate",
                "y_coordinate",
                "time_coordinate",
            ],
            "sample_ids": sample_ids,
            "video_paths": [row.video_path for row in rows],
            "splits": [row.split for row in rows],
            "frame_sampling_fractions": FRAME_FRACTIONS,
            "center_crop_short_side_fraction": 0.65,
            "preprocessor_intermediate_size": [448, 448],
            "qwen_premerger_grid_thw": [1, 28, 28],
            "pooling": (
                "Qwen contiguous block-major 2x2 mean: 784->196; then spatial "
                "2x2 mean: 14x14->7x7=49"
            ),
            "target_neutral_shared_vision_asset": True,
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_digest,
            "sample_order_sha256": sample_order_digest,
            "qwen_front_contract": model_contract,
            "qwen_source_identity": source_identity,
            "qwen_vision_front_fingerprint": vision_fingerprint,
            "qwen_front_pair_identity": front_pair_identity,
        }
        _atomic_torch_save(vision_output, payload)
        metadata = {
            "schema_version": 1,
            "feature_contract": FEATURE_CONTRACT,
            "quality_contract": QUALITY_CONTRACT,
            "manifest_sha256": manifest_digest,
            "sample_order_sha256": sample_order_digest,
            "shape": list(features.shape),
            "quality_shape": list(quality_features.shape),
            "dtype": str(features.dtype),
            "file_size_bytes": vision_output.stat().st_size,
            "target_neutral_shared_vision_asset": True,
            "vision_blocks_executed": 0,
            "vision_merger_executed": False,
            "qwen_source_identity_sha256": source_identity["sha256"],
            "qwen_vision_front_sha256": vision_fingerprint["sha256"],
            "qwen_front_pair_sha256": front_pair_identity["sha256"],
        }
        _atomic_json(metadata_path, metadata)
        vision_report = {
            "path": str(vision_output),
            "reused": False,
            "shape": list(features.shape),
            "quality_shape": list(quality_features.shape),
            "dtype": str(features.dtype),
            "parts_directory": str(parts_dir),
            "qwen_source_identity_sha256": source_identity["sha256"],
            "qwen_vision_front_sha256": vision_fingerprint["sha256"],
            "qwen_front_pair_sha256": front_pair_identity["sha256"],
        }

    return {
        "vision": vision_report,
        "language": language_report,
        "target_name": target_name,
        "prompt": TARGET_PROMPTS[target_name],
        "counts": {
            split: sum(row.split == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "qwen_execution": {
            "official_processor": True,
            "vision_patch_embed": True,
            "vision_position_embedding": True,
            "fixed_quality14_side_input": True,
            "vision_blocks": 0,
            "vision_merger": False,
            "language_embed_tokens": True,
            "language_blocks": 0,
            "attention": False,
            "lm_head": False,
        },
        "qwen_front_identity": {
            "source_identity_sha256": source_identity["sha256"],
            "vision_front_sha256": vision_fingerprint["sha256"],
            "language_embedding_sha256": language_fingerprint["sha256"],
            "front_pair_sha256": front_pair_identity["sha256"],
            "checkpoint_revision": source_identity["checkpoint_revision"],
            "model_config_sha256": source_identity["model_config_sha256"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache the frozen Qwen3-VL front only: 16-frame Vision patch+position "
            "tokens and one target-specific chat-template embedding"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Use target/dataset/manifest/Vision-cache/Language-cache paths from an "
            "audited release config. When set, only --model-path may override a "
            "config value; data-path overrides are rejected."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Auditable override for initialization.qwen_model_path",
    )
    parser.add_argument("--vision-output", type=Path, default=None)
    parser.add_argument("--language-output", type=Path, default=None)
    parser.add_argument("--target", choices=sorted(TARGET_PROMPTS), default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2, help="Videos per GPU batch (32 frames at batch=2)")
    parser.add_argument("--chunk-rows", type=int, default=16, help="Videos per resumable shard")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite-vision", action="store_true")
    args = parser.parse_args()
    if args.config is not None:
        forbidden = {
            "--dataset-root": args.dataset_root,
            "--vision-output": args.vision_output,
            "--language-output": args.language_output,
            "--target": args.target,
            "--manifest": args.manifest,
        }
        supplied = [name for name, value in forbidden.items() if value is not None]
        if supplied:
            parser.error(
                "--config owns all target/data/cache paths; remove explicit "
                + ", ".join(supplied)
            )
        settings = load_settings(args.config)
        required = {
            "data.dataset_root": settings.dataset_root,
            "data.manifest": settings.manifest_path,
            "data.vision_cache": settings.vision_cache_path,
            "data.language_cache": settings.language_cache_path,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"Config is missing required cache fields: {missing}")
        dataset_root = settings.dataset_root
        manifest = settings.manifest_path
        vision_output = settings.vision_cache_path
        language_output = settings.language_cache_path
        target_name = settings.target_name
        model_path = args.model_path or settings.qwen_model_path
        model_path_source = (
            "explicit --model-path override"
            if args.model_path is not None
            else "config initialization.qwen_model_path"
        )
    else:
        required = {
            "--dataset-root": args.dataset_root,
            "--model-path": args.model_path,
            "--vision-output": args.vision_output,
            "--language-output": args.language_output,
            "--target": args.target,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(
                "without --config the following arguments are required: "
                + ", ".join(missing)
            )
        dataset_root = args.dataset_root
        manifest = args.manifest
        vision_output = args.vision_output
        language_output = args.language_output
        target_name = args.target
        model_path = args.model_path
        model_path_source = "explicit --model-path"
    if model_path is None:
        parser.error(
            "Qwen model path is missing: set initialization.qwen_model_path in "
            "the config or pass the auditable --model-path override"
        )
    assert dataset_root is not None
    assert vision_output is not None
    assert language_output is not None
    assert target_name is not None
    report = build_cache(
        dataset_root=dataset_root,
        model_path=model_path,
        vision_output=vision_output,
        language_output=language_output,
        target_name=target_name,
        manifest=manifest,
        batch_size=args.batch_size,
        chunk_rows=args.chunk_rows,
        device_name=args.device,
        overwrite_vision=args.overwrite_vision,
    )
    report["model_path_source"] = model_path_source
    report["config"] = None if args.config is None else str(args.config.resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FRAME_FRACTIONS",
    "build_cache",
    "decode_sixteen_frames",
    "pool_qwen_front_tokens",
    "quality_tokens_from_images",
    "qwen_patch_with_position",
    "render_prompt",
]
