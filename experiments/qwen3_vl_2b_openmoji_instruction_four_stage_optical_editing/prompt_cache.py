from __future__ import annotations

import json
import time
from typing import Any

import torch

from .datasets import read_manifest
from .scenes import prompt_key
from .settings import Settings


SYSTEM_PROMPT = (
    "You edit small scenes made from familiar semantic icons. "
    "Represent the requested object category, operation, reference object, and spatial relation precisely."
)


def _chat_text(tokenizer: Any, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def build_prompt_cache(settings: Settings, device: torch.device) -> dict[str, Any]:
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

    records = [*read_manifest(settings.train_manifest), *read_manifest(settings.test_manifest)]
    instructions = sorted({str(record["instruction"]) for record in records})
    tokenizer = AutoTokenizer.from_pretrained(
        settings.qwen_checkpoint, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings.qwen_checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.requires_grad_(False).eval()
    language_model = model.model.language_model
    prompts: dict[str, torch.Tensor] = {}
    truncated = 0
    started = time.perf_counter()
    for start in range(0, len(instructions), settings.prompt_cache_batch_size):
        chunk = instructions[start : start + settings.prompt_cache_batch_size]
        values = tokenizer([_chat_text(tokenizer, text) for text in chunk], padding=True, return_tensors="pt")
        attention = values["attention_mask"].to(device)
        output = language_model(
            input_ids=values["input_ids"].to(device),
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        for index, instruction in enumerate(chunk):
            hidden = output[index][attention[index].bool()]
            if len(hidden) > settings.prompt_max_tokens:
                hidden = hidden[-settings.prompt_max_tokens :]
                truncated += 1
            prompts[prompt_key(instruction)] = hidden.detach().cpu().to(torch.bfloat16).contiguous()
    del language_model, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta = {
        "type": "frozen_qwen3_vl_openmoji_prompt_hidden",
        "model_id": settings.qwen_model_id,
        "checkpoint": str(settings.qwen_checkpoint),
        "system_prompt": SYSTEM_PROMPT,
        "hidden_size": 2048,
        "unique_prompts": len(prompts),
        "max_tokens": settings.prompt_max_tokens,
        "truncated_prompts": truncated,
        "dtype": "bfloat16",
        "elapsed_seconds": time.perf_counter() - started,
    }
    settings.prompt_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "prompts": prompts}, settings.prompt_cache_path)
    settings.prompt_cache_path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


__all__ = ["SYSTEM_PROMPT", "build_prompt_cache"]
