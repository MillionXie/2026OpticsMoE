from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .data import file_sha256, read_manifest
from .prepare_manifest import prepare_manifest


FRAME_FRACTIONS = (0.10, 0.37, 0.63, 0.90)
PROMPT = (
    "Please evaluate the quality of this video and rate it using one of the "
    "following five levels: Excellent, Good, Fair, Poor, or Bad."
)


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


def qwen_patch_with_position(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(visual, "fast_pos_embed_interpolate"):
        raise RuntimeError(
            "Qwen visual.fast_pos_embed_interpolate is required. The unavailable "
            "get_vision_interpolation_indices_and_weights helper is not used."
        )
    hidden = visual.patch_embed(pixel_values)
    positional = visual.fast_pos_embed_interpolate(grid_thw)
    return hidden + positional.to(hidden.dtype)


def qwen_pool_premerger_tokens(
    hidden: torch.Tensor, *, image_count: int
) -> torch.Tensor:
    """Deterministically shrink 784 pre-merger tokens to 196 per image.

    Qwen3-VL's processor stores every spatial 2x2 merge group contiguously.
    We average those four 1024-wide patch+position tokens, but deliberately do
    not call the learned Qwen merger. This preserves the sister experiment's
    196-token student contract without introducing a hidden trainable cache
    stage.
    """

    expected = int(image_count) * 784
    if hidden.ndim != 2 or hidden.shape[0] != expected or hidden.shape[1] != 1024:
        raise RuntimeError(
            "Expected Qwen pre-merger hidden [image_count*784,1024], got "
            f"{tuple(hidden.shape)} for image_count={image_count}"
        )
    return hidden.reshape(image_count, 196, 4, 1024).mean(2)


def build_cache(
    *,
    dataset_root: Path,
    model_path: Path,
    output: Path,
    manifest: Path | None,
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    try:
        import transformers
    except ImportError as error:
        raise RuntimeError("transformers with Qwen3-VL support is required") from error
    dataset_root = dataset_root.expanduser().resolve()
    output = output.expanduser().resolve()
    manifest = output.with_suffix(".manifest.csv") if manifest is None else manifest.resolve()
    if not manifest.exists():
        prepare_manifest(dataset_root, manifest)
    rows = read_manifest(manifest)
    device = torch.device(
        device_name if not device_name.startswith("cuda") or torch.cuda.is_available() else "cpu"
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
    model = model_cls.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).requires_grad_(False).eval()
    core = getattr(model, "model", model)
    visual = core.visual
    features = torch.empty(len(rows), 4, 196, 1024, dtype=torch.float16)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = [frame for row in batch_rows for frame in decode_four_frames(Path(row.video_path))]
        # Qwen3VLProcessor in transformers 4.57.6 does not accept text=None.
        # Dummy text satisfies processor collation only; none of its IDs enter
        # the Vision cache or the formal fixed-prompt Language cache.
        processed = processor(
            text=["x"] * len(images), images=images, return_tensors="pt"
        )
        pixel_values = processed["pixel_values"].to(
            device=device, dtype=next(visual.patch_embed.parameters()).dtype
        )
        grid = processed["image_grid_thw"].to(device)
        lengths = grid.prod(-1).long()
        if not bool(torch.all(lengths == 784)):
            raise RuntimeError(
                "Qwen processor token contract changed: expected every pre-merger "
                f"grid product=784, got {lengths.tolist()}"
            )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            hidden = qwen_patch_with_position(visual, pixel_values, grid)
            hidden = qwen_pool_premerger_tokens(
                hidden, image_count=len(batch_rows) * 4
            )
        features[start : start + len(batch_rows)] = hidden.reshape(
            len(batch_rows), 4, 196, 1024
        ).detach().cpu().half()
        print(f"[qwen_cache] {start + len(batch_rows)}/{len(rows)}", flush=True)

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
    if language.ndim != 3 or language.shape[0] != 1 or language.shape[-1] != 2048:
        raise RuntimeError(f"Unexpected Qwen language hidden shape {tuple(language.shape)}")
    language_mask = language_inputs["attention_mask"].detach().cpu().bool()
    payload = {
        "schema_version": 1,
        "feature_contract": "qwen3vl_patch_position_784_mean2x2_to196x1024_v1",
        "frame_sampling_fractions": FRAME_FRACTIONS,
        "center_crop_short_side_fraction": 0.65,
        "preprocessor_intermediate_size": [448, 448],
        "processor_min_max_pixels": 200704,
        "premerger_pooling": "block-major contiguous 2x2 mean; no learned Qwen merger",
        "frame_tokens": features,
        "language_tokens": language,
        "language_mask": language_mask,
        "language_cache_broadcast": "singleton fixed prompt is broadcast to all videos",
        "sample_ids": [row.sample_id for row in rows],
        "video_paths": [row.video_path for row in rows],
        "splits": [row.split for row in rows],
        "targets": torch.tensor([[row.spatial, row.temporal] for row in rows]),
        "target_names": ["spatial", "temporal"],
        "alignment_target_present": False,
        "qwen_prompt": PROMPT,
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "qwen_model_path": str(model_path.resolve()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return {
        "output": str(output),
        "vision_shape": list(features.shape),
        "language_shape": list(language.shape),
        "counts": {split: payload["splits"].count(split) for split in ("train", "validation", "test")},
        "alignment_cached": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache raw Qwen inputs from LGVQ MP4")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = build_cache(
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        output=args.output,
        manifest=args.manifest,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
