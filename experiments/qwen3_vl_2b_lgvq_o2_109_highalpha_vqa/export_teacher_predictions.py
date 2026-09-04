from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.modeling import build_model
from experiments.qwen3_vl_2b_lgvq_o2_109_highalpha_vqa.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    settings = load_settings(args.config)
    payload = torch.load(args.cache, map_location="cpu", weights_only=False)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model, _ = build_model(settings)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.cuda().eval()

    features = payload["frame_tokens"]
    language = payload["language_tokens"]
    language_mask = payload["language_mask"]
    predictions = torch.empty(features.shape[0], 2, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, features.shape[0], args.batch_size):
            stop = min(features.shape[0], start + args.batch_size)
            count = stop - start
            text = language[0:1].expand(count, -1, -1) if language.shape[0] == 1 else language[start:stop]
            mask = language_mask[0:1].expand(count, -1) if language_mask.shape[0] == 1 else language_mask[start:stop]
            result = model(
                features[start:stop].cuda(non_blocking=True).float(),
                text.cuda(non_blocking=True).float(),
                mask.cuda(non_blocking=True).bool(),
            )
            predictions[start:stop] = result["prediction"].cpu()
            print(f"[teacher] {stop}/{features.shape[0]}", flush=True)

    result = {
        "schema_version": 1,
        "sample_ids": list(payload["sample_ids"]),
        "predictions": predictions,
        "target_names": ["spatial", "temporal"],
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "teacher_checkpoint_epoch": int(saved["epoch"]),
        "inference_dependency": False,
        "purpose": "training-only scalar soft targets; never loaded by the deployed model",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(result, temporary)
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
