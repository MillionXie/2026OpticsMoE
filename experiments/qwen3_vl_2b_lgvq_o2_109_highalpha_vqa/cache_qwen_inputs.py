from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from .data import file_sha256, read_manifest
from .prepare_manifest import prepare_manifest


FRAME_FRACTIONS = (0.10, 0.37, 0.63, 0.90)
PROMPT = (
    "Please evaluate the quality of this video and rate it using one of the "
    "following five levels: Excellent, Good, Fair, Poor, or Bad."
)
EXPECTED_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
FEATURE_CONTRACT = "qwen3vl_full_visual_main_merger_196x2048_v1"
FRAME_COUNT = 4
PREMERGER_TOKENS = 784
MERGED_TOKENS = 196
VISION_HIDDEN_SIZE = 1024
VISION_OUTPUT_SIZE = 2048
VISION_DEPTH = 24
LANGUAGE_HIDDEN_SIZE = 2048
LANGUAGE_DEPTH = 28
PART_SCHEMA_VERSION = 1


def decode_four_frames(path: Path) -> list[Image.Image]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required to decode LGVQ MP4 files") from error
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {path}")
    frames = []
    for fraction in FRAME_FRACTIONS:
        position = min(count - 1, max(0, round((count - 1) * fraction)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, bgr = capture.read()
        if not ok:
            capture.release()
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
    capture.release()
    return frames


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_order_sha256(sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _model_identity_evidence(
    model_path: Path, processor: Any, model: torch.nn.Module
) -> list[str]:
    values = [str(model_path), model_path.name]
    for owner in (processor, getattr(processor, "tokenizer", None), model, model.config):
        if owner is None:
            continue
        for name in ("name_or_path", "_name_or_path"):
            value = getattr(owner, name, None)
            if value:
                values.append(str(value))
    return sorted(set(values))


def _validate_instruct_identity(evidence: Sequence[str]) -> None:
    normalized = "\n".join(evidence).lower().replace("_", "-")
    if "embedding" in normalized:
        raise RuntimeError(
            "The requested cache is Qwen3-VL-2B-Instruct only, but model identity "
            f"evidence contains 'Embedding': {list(evidence)}"
        )
    if "qwen3-vl-2b-instruct" not in normalized:
        raise RuntimeError(
            "Cannot prove that --model-path is Qwen/Qwen3-VL-2B-Instruct. Keep "
            "the official repository name in the local path (or its Hugging Face "
            f"cache ancestors). Identity evidence: {list(evidence)}"
        )


def _model_config_value(config: Any, name: str) -> Any:
    value = getattr(config, name, None)
    if value is None and isinstance(config, Mapping):
        value = config.get(name)
    return value


def _validate_model_architecture(
    model: torch.nn.Module, visual: torch.nn.Module
) -> dict[str, Any]:
    model_config = model.config
    vision_config = getattr(model_config, "vision_config", getattr(visual, "config", None))
    text_config = getattr(model_config, "text_config", None)
    if vision_config is None or text_config is None:
        raise RuntimeError("Qwen3-VL model is missing vision_config or text_config")
    architecture = {
        "model_type": str(_model_config_value(model_config, "model_type")),
        "vision_hidden_size": int(_model_config_value(vision_config, "hidden_size")),
        "vision_out_hidden_size": int(_model_config_value(vision_config, "out_hidden_size")),
        "vision_depth": int(_model_config_value(vision_config, "depth")),
        "spatial_merge_size": int(_model_config_value(vision_config, "spatial_merge_size")),
        "patch_size": int(_model_config_value(vision_config, "patch_size")),
        "temporal_patch_size": int(_model_config_value(vision_config, "temporal_patch_size")),
        "checkpoint_deepstack_visual_indexes": [
            int(value)
            for value in (_model_config_value(vision_config, "deepstack_visual_indexes") or [])
        ],
        "language_hidden_size": int(_model_config_value(text_config, "hidden_size")),
        "language_depth": int(_model_config_value(text_config, "num_hidden_layers")),
    }
    expected = {
        "model_type": "qwen3_vl",
        "vision_hidden_size": VISION_HIDDEN_SIZE,
        "vision_out_hidden_size": VISION_OUTPUT_SIZE,
        "vision_depth": VISION_DEPTH,
        "spatial_merge_size": 2,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "checkpoint_deepstack_visual_indexes": [5, 11, 17],
        "language_hidden_size": LANGUAGE_HIDDEN_SIZE,
        "language_depth": LANGUAGE_DEPTH,
    }
    if architecture != expected:
        raise RuntimeError(
            "Qwen3-VL-2B-Instruct architecture contract changed: "
            f"expected={expected}, actual={architecture}"
        )
    if not hasattr(visual, "blocks") or len(visual.blocks) != VISION_DEPTH:
        raise RuntimeError("Vision tower does not expose the expected 24 frozen blocks")
    if not hasattr(visual, "merger"):
        raise RuntimeError("Vision tower does not expose the learned main merger")
    return architecture


def _checkpoint_files(model_path: Path) -> list[Path]:
    names = (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    files = [model_path / name for name in names if (model_path / name).is_file()]
    weights = sorted(model_path.glob("*.safetensors"))
    if not weights:
        weights = sorted(model_path.glob("pytorch_model*.bin"))
    if not weights:
        raise RuntimeError(f"No model weights found directly under {model_path}")
    return sorted(set([*files, *weights]), key=lambda value: value.name)


def _model_provenance(
    model_path: Path,
    evidence: Sequence[str],
    architecture: Mapping[str, Any],
) -> dict[str, Any]:
    files = []
    for path in _checkpoint_files(model_path):
        files.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    fingerprint_payload = {
        "expected_model_id": EXPECTED_MODEL_ID,
        "architecture": dict(architecture),
        "files": files,
    }
    return {
        **fingerprint_payload,
        "resolved_model_path": str(model_path),
        "identity_evidence": list(evidence),
        "checkpoint_fingerprint_sha256": _json_digest(fingerprint_payload),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _part_path(parts_dir: Path, start: int, stop: int) -> Path:
    return parts_dir / f"vision_{start:06d}_{stop:06d}.pt"


def _part_identity(
    *,
    start: int,
    stop: int,
    sample_ids: Sequence[str],
    manifest_sha256: str,
    model_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": PART_SCHEMA_VERSION,
        "feature_contract": FEATURE_CONTRACT,
        "start": int(start),
        "stop": int(stop),
        "sample_ids": list(sample_ids),
        "manifest_sha256": manifest_sha256,
        "model_checkpoint_fingerprint_sha256": model_fingerprint,
    }


def _load_valid_part(
    path: Path, expected: Mapping[str, Any]
) -> tuple[torch.Tensor | None, str | None]:
    if not path.exists():
        return None, "missing"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actual_identity = {name: payload.get(name) for name in expected}
        if actual_identity != dict(expected):
            return None, "identity mismatch"
        tokens = payload.get("frame_tokens")
        expected_rows = int(expected["stop"]) - int(expected["start"])
        expected_shape = (
            expected_rows,
            FRAME_COUNT,
            MERGED_TOKENS,
            VISION_OUTPUT_SIZE,
        )
        if not torch.is_tensor(tokens) or tokens.dtype != torch.float16:
            return None, "frame_tokens is not a float16 tensor"
        if tuple(tokens.shape) != expected_shape:
            return None, f"shape {tuple(tokens.shape)} != {expected_shape}"
        if not bool(torch.isfinite(tokens).all()):
            return None, "non-finite frame tokens"
        return tokens.contiguous(), None
    except Exception as error:  # A truncated interrupted shard is safely regenerated.
        return None, f"unreadable: {type(error).__name__}: {error}"


def _main_merger_features(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    image_count: int,
) -> torch.Tensor:
    # transformers 4.57 returns ``(main_merger, deepstack_list)`` from the
    # Qwen3-VL vision module, whereas some remote-code revisions expose a
    # ModelOutput.  Support both without forwarding ``return_dict`` into every
    # vision block (the upstream 4.57 implementation passes arbitrary kwargs
    # to its blocks and would reject it there).
    outputs = visual(pixel_values, grid_thw=grid_thw)
    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise RuntimeError("Qwen Vision forward returned an empty tuple")
        merged = outputs[0]
        deepstack = outputs[1] if len(outputs) > 1 else None
    else:
        merged = getattr(outputs, "pooler_output", None)
        deepstack = getattr(outputs, "deepstack_features", None)
    if deepstack not in (None, [], ()):
        raise RuntimeError(
            "DeepStack features were produced despite the no-DeepStack cache contract"
        )
    if merged is None:
        raise RuntimeError("Qwen Vision forward did not expose main-merger pooler_output")
    expected_shape = (int(image_count) * MERGED_TOKENS, VISION_OUTPUT_SIZE)
    if tuple(merged.shape) != expected_shape:
        raise RuntimeError(
            f"Expected main-merger output {expected_shape}, got {tuple(merged.shape)}"
        )
    if not bool(torch.isfinite(merged).all()):
        raise RuntimeError("Qwen Vision main-merger output contains non-finite values")
    return merged.reshape(image_count, MERGED_TOKENS, VISION_OUTPUT_SIZE)


def _extract_vision_rows(
    *,
    rows: Sequence[Any],
    processor: Any,
    visual: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    batches = []
    visual_dtype = next(visual.patch_embed.parameters()).dtype
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = [
            frame
            for row in batch_rows
            for frame in decode_four_frames(Path(row.video_path))
        ]
        processed = processor(
            text=["x"] * len(images), images=images, return_tensors="pt"
        )
        pixel_values = processed["pixel_values"].to(
            device=device, dtype=visual_dtype
        )
        grid = processed["image_grid_thw"].to(device)
        lengths = grid.prod(-1).long()
        if grid.shape[0] != len(images) or not bool(
            torch.all(lengths == PREMERGER_TOKENS)
        ):
            raise RuntimeError(
                "Qwen processor token contract changed: expected one [1,28,28] "
                f"grid (product 784) per frame, got grid={grid.tolist()}"
            )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            merged = _main_merger_features(
                visual,
                pixel_values,
                grid,
                image_count=len(images),
            )
        batches.append(
            merged.reshape(
                len(batch_rows),
                FRAME_COUNT,
                MERGED_TOKENS,
                VISION_OUTPUT_SIZE,
            )
            .detach()
            .cpu()
            .half()
            .contiguous()
        )
    return torch.cat(batches, dim=0)


def build_cache(
    *,
    dataset_root: Path,
    model_path: Path,
    output: Path,
    manifest: Path | None,
    batch_size: int,
    device_name: str,
    chunk_rows: int = 32,
) -> dict[str, Any]:
    try:
        import transformers
    except ImportError as error:
        raise RuntimeError("transformers with Qwen3-VL support is required") from error
    if batch_size <= 0 or chunk_rows <= 0:
        raise ValueError("--batch-size and --chunk-rows must both be positive")

    dataset_root = dataset_root.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(".manifest.csv") if manifest is None else manifest.resolve()
    if not manifest.exists():
        prepare_manifest(dataset_root, manifest)
    rows = read_manifest(manifest)
    if not rows:
        raise RuntimeError(f"Manifest contains no samples: {manifest}")
    manifest_sha256 = file_sha256(manifest)
    sample_ids = [row.sample_id for row in rows]

    device = torch.device(
        device_name
        if not device_name.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    processor = transformers.AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=200704,
        max_pixels=200704,
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
        raise RuntimeError("Loaded model does not expose the Qwen3-VL Vision tower")

    evidence = _model_identity_evidence(model_path, processor, model)
    _validate_instruct_identity(evidence)
    architecture = _validate_model_architecture(model, visual)
    provenance = _model_provenance(model_path, evidence, architecture)
    model_fingerprint = provenance["checkpoint_fingerprint_sha256"]

    # The full 24-block tower and learned main merger remain active. Only the
    # three auxiliary DeepStack mergers are disabled for this cache boundary.
    checkpoint_deepstack_indexes = list(visual.deepstack_visual_indexes)
    visual.deepstack_visual_indexes = []

    parts_dir = Path(str(output) + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    resume_index = {
        "schema_version": PART_SCHEMA_VERSION,
        "feature_contract": FEATURE_CONTRACT,
        "row_count": len(rows),
        "sample_order_sha256": _sample_order_sha256(sample_ids),
        "manifest_sha256": manifest_sha256,
        "model_checkpoint_fingerprint_sha256": model_fingerprint,
        "chunk_rows": int(chunk_rows),
        "vision_shape": [
            len(rows),
            FRAME_COUNT,
            MERGED_TOKENS,
            VISION_OUTPUT_SIZE,
        ],
        "vision_dtype": "float16",
        "vision_forward": "full_frozen_24_block_tower_then_learned_main_merger",
        "deepstack_used": False,
    }
    resume_index_path = parts_dir / "resume_index.json"
    if resume_index_path.exists():
        existing = json.loads(resume_index_path.read_text(encoding="utf-8"))
        if existing != resume_index:
            raise RuntimeError(
                "Existing cache parts belong to a different manifest/model/contract. "
                f"Use a different --output rather than mixing shards: {resume_index_path}"
            )
    else:
        _atomic_json(resume_index_path, resume_index)

    part_specs = []
    for start in range(0, len(rows), chunk_rows):
        stop = min(len(rows), start + chunk_rows)
        expected = _part_identity(
            start=start,
            stop=stop,
            sample_ids=sample_ids[start:stop],
            manifest_sha256=manifest_sha256,
            model_fingerprint=model_fingerprint,
        )
        path = _part_path(parts_dir, start, stop)
        tokens, reason = _load_valid_part(path, expected)
        if tokens is None:
            if reason != "missing":
                print(
                    f"[qwen_cache] regenerating invalid shard {path.name}: {reason}",
                    flush=True,
                )
            tokens = _extract_vision_rows(
                rows=rows[start:stop],
                processor=processor,
                visual=visual,
                device=device,
                batch_size=batch_size,
            )
            _atomic_torch_save(path, {**expected, "frame_tokens": tokens})
            print(
                f"[qwen_cache] saved {stop}/{len(rows)} -> {path.name}", flush=True
            )
        else:
            print(
                f"[qwen_cache] resumed {stop}/{len(rows)} <- {path.name}", flush=True
            )
        part_specs.append((path, expected))

    text_inputs = processor(text=[PROMPT], padding=True, return_tensors="pt")
    language_inputs = {
        key: value.to(device)
        for key, value in text_inputs.items()
        if torch.is_tensor(value) and key in {"input_ids", "attention_mask"}
    }
    with torch.inference_mode():
        language_output = core(
            **language_inputs,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
    language = language_output.last_hidden_state.detach().cpu().half()
    if (
        language.ndim != 3
        or language.shape[0] != 1
        or language.shape[-1] != LANGUAGE_HIDDEN_SIZE
    ):
        raise RuntimeError(f"Unexpected Qwen language hidden shape {tuple(language.shape)}")
    if not bool(torch.isfinite(language).all()):
        raise RuntimeError("Qwen language cache contains non-finite values")
    language_mask = language_inputs["attention_mask"].detach().cpu().bool()

    # Release the 2B model before allocating the final approximately 8.4-GiB
    # contiguous tensor. If final serialization is interrupted, all validated
    # parts remain and the next invocation only repeats this assembly step.
    del language_output, language_inputs, core, visual, model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    features = torch.empty(
        len(rows),
        FRAME_COUNT,
        MERGED_TOKENS,
        VISION_OUTPUT_SIZE,
        dtype=torch.float16,
    )
    for path, expected in part_specs:
        tokens, reason = _load_valid_part(path, expected)
        if tokens is None:
            raise RuntimeError(
                f"Validated shard became invalid during assembly: {path}: {reason}"
            )
        features[int(expected["start"]) : int(expected["stop"])].copy_(tokens)
    # Each shard was checked for finiteness immediately before this copy. Do
    # not allocate a second 4.2-GiB boolean tensor to recheck the 8.4-GiB
    # assembled cache in one operation.

    payload = {
        "schema_version": 1,
        "feature_contract": FEATURE_CONTRACT,
        "model_provenance": provenance,
        "frame_sampling_fractions": FRAME_FRACTIONS,
        "center_crop_short_side_fraction": 0.65,
        "preprocessor_intermediate_size": [448, 448],
        "processor_min_max_pixels": 200704,
        "vision_tower_blocks_executed": VISION_DEPTH,
        "vision_main_merger_used": True,
        "deepstack_used": False,
        "checkpoint_deepstack_visual_indexes": checkpoint_deepstack_indexes,
        "runtime_deepstack_visual_indexes": [],
        "frame_tokens": features,
        "language_tokens": language,
        "language_mask": language_mask,
        "language_cache_broadcast": "singleton fixed prompt is broadcast to all videos",
        "sample_ids": sample_ids,
        "video_paths": [row.video_path for row in rows],
        "splits": [row.split for row in rows],
        "targets": torch.tensor([[row.spatial, row.temporal] for row in rows]),
        "target_names": ["spatial", "temporal"],
        "alignment_target_present": False,
        "qwen_prompt": PROMPT,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "qwen_model_id": EXPECTED_MODEL_ID,
        "qwen_model_path": str(model_path),
        "resume_parts_directory": str(parts_dir),
        "resume_index_sha256": file_sha256(resume_index_path),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return {
        "output": str(output),
        "feature_contract": FEATURE_CONTRACT,
        "model_id": EXPECTED_MODEL_ID,
        "model_checkpoint_fingerprint_sha256": model_fingerprint,
        "vision_shape": list(features.shape),
        "vision_dtype": str(features.dtype),
        "vision_main_merger_used": True,
        "deepstack_used": False,
        "language_shape": list(language.shape),
        "counts": {
            split: payload["splits"].count(split)
            for split in ("train", "validation", "test")
        },
        "resume_parts_directory": str(parts_dir),
        "alignment_cached": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Cache full frozen Qwen3-VL-2B-Instruct Vision tower + main merger "
            "features for LGVQ (DeepStack disabled)"
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=32,
        help="Videos per atomic resume shard; a failure loses at most one shard.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = build_cache(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output=args.output,
        manifest=args.manifest,
        batch_size=args.batch_size,
        device_name=args.device,
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


