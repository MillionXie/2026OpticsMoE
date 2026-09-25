"""Distill the unified 13M student's Qwen head into the 150M latent editor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .qwen_mini_small import QwenMiniConfig, QwenMiniTextEncoder


def fit_bridge(*, small_checkpoint: Path, embedding_cache: Path, teacher_cache: Path,
               output: Path, device: torch.device, steps: int = 600) -> dict:
    small = torch.load(small_checkpoint, map_location="cpu", weights_only=False)
    text = QwenMiniTextEncoder(QwenMiniConfig(**small["qwen_mini_config"]))
    text.load_state_dict({key.removeprefix("text."): value
                          for key, value in small["model"].items() if key.startswith("text.")})
    text = text.to(device).eval().requires_grad_(False)
    source = torch.load(embedding_cache, map_location="cpu", weights_only=False)
    teacher = torch.load(teacher_cache, map_location="cpu", weights_only=False)
    if source["prompts"] != [row["prompt"] for row in teacher["rows"]]:
        raise ValueError("Prompt order mismatch between Qwen embedding and teacher caches")
    hidden = []
    with torch.inference_mode():
        for start in range(0, len(source["prompts"]), 16):
            embeddings = source["embeddings"][start:start+16].to(device).float()
            mask = source["attention_mask"][start:start+16].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden.append(text.hidden(embeddings, mask).float())
    hidden = torch.cat(hidden)
    target = teacher["text"].float().to(device)
    bridge = nn.Linear(hidden.shape[-1], target.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ in range(steps):
        predicted = bridge(hidden)
        loss = F.mse_loss(predicted, target) + 0.5 * (1-F.cosine_similarity(predicted,target).mean())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.inference_mode():
        predicted = bridge(hidden)
        mse = float(F.mse_loss(predicted, target))
        cosine = float(F.cosine_similarity(predicted, target).mean())
    counted = sum(p.numel() for p in text.parameters()) + sum(p.numel() for p in bridge.parameters())
    result = {"schema_version": 1, "small_checkpoint": str(small_checkpoint),
              "prompts": len(source["prompts"]), "mse_to_full_qwen": mse,
              "cosine_to_full_qwen": cosine, "counted_text_parameters": counted,
              "shared_token_embedding_excluded": 311_164_928}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": result, "text_config": small["qwen_mini_config"],
                "text": {k:v.half().cpu() for k,v in text.state_dict().items()},
                "bridge": {k:v.half().cpu() for k,v in bridge.state_dict().items()},
                "prompts": source["prompts"], "features": predicted.half().cpu()}, output)
    return result


def rewrite_caches(*, teacher_cache: Path, bridge_checkpoint: Path,
                   original_latents: Path, instruction_output: Path,
                   latent_output: Path) -> dict:
    teacher = torch.load(teacher_cache, map_location="cpu", weights_only=False)
    bridge = torch.load(bridge_checkpoint, map_location="cpu", weights_only=False)
    if bridge["prompts"] != [row["prompt"] for row in teacher["rows"]]:
        raise ValueError("Bridge and instruction prompts differ")
    features = bridge["features"]
    meta = dict(teacher["meta"])
    qwen = dict(meta["qwen_pruning"])
    qwen["counted_text_encoder_parameters"] = bridge["meta"]["counted_text_parameters"]
    qwen["model_kind"] = "two-block width-640 Qwen-mini plus 640-to-2048 bridge"
    meta["qwen_pruning"] = qwen
    instruction_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "rows": teacher["rows"], "text": features}, instruction_output)
    lookup = {prompt: features[i] for i,prompt in enumerate(bridge["prompts"])}
    latent_output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train","val","test"):
        source = torch.load(original_latents/f"{split}.pt",map_location="cpu",weights_only=False,mmap=True)
        source["qwen_text"] = torch.stack([lookup[prompt] for prompt in source["prompts"]])
        torch.save(source, latent_output/f"{split}.pt")
        counts[split] = len(source["prompts"])
    return {"instruction_output": str(instruction_output),
            "latent_output": str(latent_output), "splits": counts,
            "counted_text_parameters": bridge["meta"]["counted_text_parameters"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    for name in ("small-checkpoint","embedding-cache","teacher-cache","output"):
        fit.add_argument(f"--{name}",type=Path,required=True)
    fit.add_argument("--steps",type=int,default=600)
    fit.add_argument("--device",default="cuda")
    rewrite = sub.add_parser("rewrite")
    for name in ("teacher-cache","bridge-checkpoint","original-latents",
                 "instruction-output","latent-output"):
        rewrite.add_argument(f"--{name}",type=Path,required=True)
    args = parser.parse_args()
    if args.command == "fit":
        result = fit_bridge(small_checkpoint=args.small_checkpoint.resolve(),
                            embedding_cache=args.embedding_cache.resolve(),
                            teacher_cache=args.teacher_cache.resolve(),
                            output=args.output.resolve(),device=torch.device(args.device),steps=args.steps)
    else:
        result = rewrite_caches(teacher_cache=args.teacher_cache.resolve(),
                                bridge_checkpoint=args.bridge_checkpoint.resolve(),
                                original_latents=args.original_latents.resolve(),
                                instruction_output=args.instruction_output.resolve(),
                                latent_output=args.latent_output.resolve())
    print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
