"""Build deterministic SD-Turbo latent targets for the compact one-step student."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import read_manifest


def _unique_caption_indices(data_dir: Path) -> list[int]:
    rows = read_manifest(data_dir / "train.jsonl")
    seen: set[str] = set()
    result = []
    for index, row in enumerate(rows):
        if row.caption not in seen:
            seen.add(row.caption)
            result.append(index)
    return result


@torch.inference_mode()
def _generate_split(
    qwen: torch.Tensor,
    condition: torch.Tensor,
    seeds_per_prompt: int,
    seed: int,
    unet: torch.nn.Module,
    scheduler: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    prompt_indices = torch.arange(len(qwen)).repeat_interleave(seeds_per_prompt)
    sample_seeds = torch.arange(len(prompt_indices), dtype=torch.int64) + seed
    qwen_samples, noise_samples, target_samples = [], [], []
    for start in range(0, len(prompt_indices), batch_size):
        indices = prompt_indices[start : start + batch_size]
        seeds = sample_seeds[start : start + batch_size]
        noise = torch.cat([
            torch.randn(
                (1, 4, 64, 64),
                generator=torch.Generator(device=device).manual_seed(int(value)),
                device=device,
                dtype=unet.dtype,
            )
            for value in seeds
        ])
        latent = noise * scheduler.init_noise_sigma
        scheduler.set_timesteps(1, device=device)
        timestep = scheduler.timesteps[0]
        latent_input = scheduler.scale_model_input(latent, timestep)
        prediction = unet(
            latent_input,
            timestep,
            encoder_hidden_states=condition[indices].to(device=device, dtype=unet.dtype),
            return_dict=False,
        )[0]
        target = scheduler.step(prediction, timestep, latent, return_dict=False)[0]
        qwen_samples.append(qwen[indices].half().cpu())
        noise_samples.append(noise.half().cpu())
        target_samples.append(target.half().cpu())
        print(json.dumps({"cached": min(start + batch_size, len(prompt_indices)), "total": len(prompt_indices)}), flush=True)
    qwen_result = torch.cat(qwen_samples)
    noise_result = torch.cat(noise_samples)
    target_result = torch.cat(target_samples)
    return {
        "qwen": qwen_result,
        "noise": noise_result,
        "target": target_result,
        "prompt_indices": prompt_indices,
        "seeds": sample_seeds,
        "statistics": {
            "noise_mean": float(noise_result.float().mean()),
            "noise_std": float(noise_result.float().std()),
            "target_mean": float(target_result.float().mean()),
            "target_std": float(target_result.float().std()),
        },
    }


@torch.inference_mode()
def build_teacher_latent_cache(
    data_dir: Path,
    teacher_condition_cache: Path,
    turbo_checkpoint: Path,
    output_dir: Path,
    train_seeds_per_prompt: int,
    val_seeds_per_prompt: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    from diffusers import EulerDiscreteScheduler, UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        turbo_checkpoint,
        subfolder="unet",
        variant="fp16",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to(device).eval().requires_grad_(False)
    scheduler = EulerDiscreteScheduler.from_pretrained(
        turbo_checkpoint, subfolder="scheduler", local_files_only=True
    )
    qwen_train_cache = torch.load(
        data_dir / "feature_cache" / "train.pt", map_location="cpu", weights_only=False, mmap=True
    )
    qwen_val_cache = torch.load(
        data_dir / "feature_cache" / "val.pt", map_location="cpu", weights_only=False, mmap=True
    )
    teacher = torch.load(teacher_condition_cache, map_location="cpu", weights_only=False, mmap=True)
    unique = _unique_caption_indices(data_dir)
    train_qwen = torch.cat((qwen_train_cache["text"][unique], teacher["synthetic"]["qwen"]))
    train_condition = torch.cat((teacher["train"]["condition"][unique], teacher["synthetic"]["condition"]))
    val_qwen = qwen_val_cache["text"]
    val_condition = teacher["val"]["condition"]
    split_inputs = {
        "train": (train_qwen, train_condition, train_seeds_per_prompt, seed * 100_000),
        "val": (val_qwen, val_condition, val_seeds_per_prompt, seed * 200_000),
    }
    summaries = {}
    for split, (qwen, condition, seeds_per_prompt, split_seed) in split_inputs.items():
        payload = _generate_split(
            qwen.float(), condition.float(), seeds_per_prompt, split_seed,
            unet, scheduler, device, batch_size,
        )
        payload["meta"] = {
            "schema_version": 1,
            "split": split,
            "teacher": "SD-Turbo UNet, one call, native frozen CLIP condition",
            "prompts": len(qwen),
            "seeds_per_prompt": seeds_per_prompt,
            "samples": len(payload["qwen"]),
            "latent_shape": [4, 64, 64],
            "init_noise_sigma": float(scheduler.init_noise_sigma),
            "pca_used": False,
        }
        torch.save(payload, output_dir / f"{split}.pt")
        summaries[split] = {**payload["meta"], **payload["statistics"]}
        del payload
    report = {
        "schema_version": 1,
        "output_dir": str(output_dir),
        "splits": summaries,
        "deployment_teacher_components": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    del unet
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache one-step SD-Turbo latent distillation targets")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-condition-cache", type=Path, required=True)
    parser.add_argument("--turbo-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seeds-per-prompt", type=int, default=8)
    parser.add_argument("--val-seeds-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = build_teacher_latent_cache(
        args.data_dir.expanduser().resolve(),
        args.teacher_condition_cache.expanduser().resolve(),
        args.turbo_checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.train_seeds_per_prompt,
        args.val_seeds_per_prompt,
        args.batch_size,
        args.seed,
        torch.device(args.device),
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
