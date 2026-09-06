from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from . import quality_token_common as core


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_ids(rows: list[dict[str, Any]]) -> str:
    value = "\n".join(row["sample_id"] for row in rows).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, required=True, choices=core.COUNTS)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.image_size % 32 or args.image_size < 224:
        raise ValueError("Qwen3-VL image-size must be >=224 and divisible by 32")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = torch.device("cuda:0")
    model_path = args.model.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    rows = core.read_manifest(manifest)
    if len(rows) != 2808:
        raise RuntimeError(f"Expected 2808 rows, got {len(rows)}")
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        min_pixels=args.image_size * args.image_size,
        max_pixels=args.image_size * args.image_size,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompt = core.render_prompt(processor)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    identity = {
        "schema_version": 1,
        "feature_contract": "frozen_qwen3vl_temporal_last_valid_hidden_2048_v1",
        "model": str(model_path),
        "frame_count": args.frames,
        "frame_fractions": core.FRAME_FRACTIONS[args.frames],
        "image_size_wh": [args.image_size, args.image_size],
        "processor_pixels_per_frame": args.image_size * args.image_size,
        "center_crop_short_side_fraction": 0.65,
        "prompt": core.PROMPT,
        "sample_count": len(rows),
        "sample_order_sha256": sha256_ids(rows),
        "qwen_trainable_parameters": 0,
    }
    output = args.output.expanduser().resolve()
    shard_root = output.with_suffix(".parts")
    shard_root.mkdir(parents=True, exist_ok=True)
    all_features = torch.empty(len(rows), 2048, dtype=torch.float16)
    started = time.perf_counter()
    fractions = core.FRAME_FRACTIONS[args.frames]
    for chunk_start in range(0, len(rows), args.chunk_rows):
        chunk_stop = min(len(rows), chunk_start + args.chunk_rows)
        shard = shard_root / f"rows_{chunk_start:05d}_{chunk_stop:05d}.pt"
        expected_ids = [row["sample_id"] for row in rows[chunk_start:chunk_stop]]
        if shard.is_file():
            candidate = torch.load(shard, map_location="cpu", weights_only=False)
            if (
                candidate.get("identity") == identity
                and candidate.get("sample_ids") == expected_ids
                and tuple(candidate.get("features", ()).shape) == (chunk_stop - chunk_start, 2048)
            ):
                all_features[chunk_start:chunk_stop].copy_(candidate["features"])
                print(f"[resume] {chunk_stop}/{len(rows)}", flush=True)
                continue
        chunk_features: list[torch.Tensor] = []
        for start in range(chunk_start, chunk_stop, args.batch_size):
            stop = min(chunk_stop, start + args.batch_size)
            batch_rows = rows[start:stop]
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.decode_workers, len(batch_rows))
            ) as executor:
                decoded = list(
                    executor.map(
                        lambda row: core.decode_random_seek(
                            Path(row["video_path"]), fractions, args.image_size
                        ),
                        batch_rows,
                    )
                )
            frames = [item[0] for item in decoded]
            metadata = [item[1] for item in decoded]
            inputs = processor(
                text=[prompt] * len(batch_rows),
                videos=frames,
                video_metadata=metadata,
                padding=True,
                return_tensors="pt",
                do_sample_frames=False,
            )
            inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(**inputs, return_dict=True, use_cache=False).last_hidden_state
            mask = inputs["attention_mask"].bool()
            indices = torch.arange(mask.shape[1], device=device).expand_as(mask)
            last = indices.masked_fill(~mask, -1).amax(1)
            pooled = hidden[torch.arange(hidden.shape[0], device=device), last]
            pooled = pooled.float().cpu().half().contiguous()
            if tuple(pooled.shape) != (len(batch_rows), 2048):
                raise RuntimeError(f"Unexpected pooled shape: {tuple(pooled.shape)}")
            if not bool(torch.isfinite(pooled).all()):
                raise RuntimeError("Non-finite feature")
            chunk_features.append(pooled)
            del inputs, hidden, pooled
        chunk_tensor = torch.cat(chunk_features)
        atomic_save(
            shard,
            {"identity": identity, "sample_ids": expected_ids, "features": chunk_tensor},
        )
        all_features[chunk_start:chunk_stop].copy_(chunk_tensor)
        print(f"[saved] {chunk_stop}/{len(rows)}", flush=True)
    payload = {
        "identity": identity,
        "sample_ids": [row["sample_id"] for row in rows],
        "splits": [row["split"] for row in rows],
        "targets": torch.tensor([row["temporal"] for row in rows], dtype=torch.float32),
        "features": all_features,
    }
    atomic_save(output, payload)
    report = {**identity, "output": str(output), "elapsed_seconds": time.perf_counter() - started, "feature_shape": list(all_features.shape), "feature_dtype": str(all_features.dtype)}
    core.write_json(output.with_suffix(".report.json"), report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
