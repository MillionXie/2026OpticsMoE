from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .datasets import read_manifest
from .scenes import prompt_key
from .settings import Settings


SYSTEM_PROMPT = (
    "You edit simple images containing colored geometric shapes. "
    "Represent the user's requested operation, target object, attributes, and spatial relation precisely."
)


def _chat_text(tokenizer: Any, instruction: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _unique_instructions(settings: Settings) -> list[str]:
    records = [
        *read_manifest(settings.train_manifest),
        *read_manifest(settings.test_manifest),
    ]
    return sorted({str(record["instruction"]) for record in records})


@torch.inference_mode()
def build_prompt_cache(settings: Settings, device: torch.device) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("transformers with Qwen3-VL support is required") from error

    instructions = _unique_instructions(settings)
    tokenizer = AutoTokenizer.from_pretrained(
        settings.qwen_checkpoint, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    started = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings.qwen_checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.requires_grad_(False).eval()
    language_model = model.model.language_model
    prompts: dict[str, torch.Tensor] = {}
    truncation_count = 0

    for start in range(0, len(instructions), settings.prompt_cache_batch_size):
        chunk = instructions[start : start + settings.prompt_cache_batch_size]
        text = [_chat_text(tokenizer, instruction) for instruction in chunk]
        inputs = tokenizer(text, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs = language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        for index, instruction in enumerate(chunk):
            valid = hidden[index][attention_mask[index].bool()]
            if len(valid) > settings.prompt_max_tokens:
                valid = valid[-settings.prompt_max_tokens :]
                truncation_count += 1
            prompts[prompt_key(instruction)] = valid.detach().to("cpu", torch.bfloat16).contiguous()

    del language_model, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    settings.prompt_cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "type": "frozen_qwen3_vl_contextual_prompt_hidden",
            "model_id": settings.qwen_model_id,
            "checkpoint": str(settings.qwen_checkpoint),
            "system_prompt": SYSTEM_PROMPT,
            "hidden_size": 2048,
            "unique_prompts": len(prompts),
            "max_tokens": settings.prompt_max_tokens,
            "truncated_prompts": truncation_count,
            "dtype": "bfloat16",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "prompts": prompts,
    }
    torch.save(payload, settings.prompt_cache_path)
    metadata_path = settings.prompt_cache_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(payload["meta"], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload["meta"]


__all__ = ["SYSTEM_PROMPT", "build_prompt_cache"]
