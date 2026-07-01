#!/usr/bin/env python3
"""
CIFAR-10 experiments for Qwen-VL/Qwen3-VL models.

This file is intentionally self-contained so it can be dropped into a repo and
used as the first reproducible entry point for:
  1. exporting CIFAR-10 to the official Qwen-VL fine-tuning JSONL format;
  2. zero-shot / post-finetune generative classification evaluation;
  3. a lightweight LoRA SFT baseline.

Suggested installs, depending on your machine:
  pip install "torch" "torchvision" "transformers>=4.57.0" accelerate peft pillow tqdm

Examples:
  # 1) Prepare Qwen-VL conversation JSONL files.
  python qwen_vl_experiments/cifar10_qwen_vl.py prepare \
      --data-root ./data --output-dir ./runs/qwen_vl_cifar10 --image-size 224

  # 2) Measure zero-shot classification speed/accuracy.
  python qwen_vl_experiments/cifar10_qwen_vl.py eval \
      --model-id Qwen/Qwen3-VL-2B-Instruct \
      --data-root ./data --output-dir ./runs/qwen_vl_cifar10_eval \
      --eval-limit 200 --image-size 224 --max-new-tokens 8

  # 3) LoRA fine-tune on a small subset.
  python qwen_vl_experiments/cifar10_qwen_vl.py finetune_lora \
      --model-id Qwen/Qwen3-VL-2B-Instruct \
      --data-root ./data --output-dir ./runs/qwen_vl_cifar10_lora \
      --train-limit 5000 --epochs 1 --batch-size 1 --grad-accum 16 \
      --learning-rate 2e-5 --image-size 224
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from tqdm import tqdm


CIFAR10_LABELS: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

QUESTION = (
    "Classify this CIFAR-10 image. "
    "Answer with exactly one label from this list: "
    "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck."
)


@dataclass
class EvalMetrics:
    total: int
    correct: int
    accuracy: float
    macro_f1: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    images_per_second: float
    generated_tokens_per_second: float
    total_wall_time_sec: float
    peak_cuda_memory_mb: Optional[float]
    model_id: str
    image_size: int
    max_new_tokens: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resize_for_vlm(image: Image.Image, image_size: int) -> Image.Image:
    """Upscale CIFAR-10's 32x32 images so VLMs receive enough visual detail."""
    if image_size and image.size != (image_size, image_size):
        return image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return image


def load_cifar10(data_root: str, split: str) -> CIFAR10:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split!r}")
    return CIFAR10(root=data_root, train=(split == "train"), download=True)


def get_limited_indices(length: int, limit: Optional[int], seed: int) -> List[int]:
    indices = list(range(length))
    if limit is None or limit >= length:
        return indices
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:limit]


def build_messages(question: str, answer: Optional[str] = None) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )
    return messages


def apply_chat_template(processor: Any, question: str, answer: Optional[str]) -> str:
    """Create prompt/full text using the model's native chat template."""
    messages = build_messages(question=question, answer=answer)
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=(answer is None),
    )


def export_qwenvl_split(
    dataset: CIFAR10,
    indices: Sequence[int],
    split: str,
    output_dir: Path,
    image_size: int,
) -> Path:
    image_dir = output_dir / "images" / split
    ensure_dir(image_dir)
    jsonl_path = output_dir / f"cifar10_{split}_qwenvl.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for out_idx, idx in enumerate(tqdm(indices, desc=f"export {split}")):
            image, target = dataset[idx]
            label = CIFAR10_LABELS[int(target)]
            image = resize_for_vlm(image.convert("RGB"), image_size)
            image_path = image_dir / f"{split}_{out_idx:06d}_{label}.png"
            image.save(image_path)

            row = {
                "image": str(image_path.resolve()),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{QUESTION}"},
                    {"from": "gpt", "value": label},
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return jsonl_path


class Cifar10QwenVLDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        image_size: int,
        limit: Optional[int],
        seed: int,
    ) -> None:
        self.dataset = load_cifar10(data_root, split)
        self.indices = get_limited_indices(len(self.dataset), limit, seed)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        image, target = self.dataset[self.indices[item]]
        label = CIFAR10_LABELS[int(target)]
        return {
            "image": resize_for_vlm(image.convert("RGB"), self.image_size),
            "label_id": int(target),
            "label": label,
            "question": QUESTION,
            "answer": label,
        }


def parse_torch_dtype(name: str) -> Any:
    name = name.lower()
    if name in {"auto", "none"}:
        return "auto"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def maybe_autoclass(model_id: str) -> Any:
    """Resolve Qwen VL model classes while keeping the script version-tolerant."""
    import transformers

    candidates: List[str] = []
    model_key = model_id.lower()
    if "qwen3-vl" in model_key:
        candidates.append("Qwen3VLForConditionalGeneration")
    if "qwen2.5-vl" in model_key or "qwen2_5-vl" in model_key:
        candidates.append("Qwen2_5_VLForConditionalGeneration")

    candidates.extend(
        [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]
    )

    for class_name in candidates:
        model_cls = getattr(transformers, class_name, None)
        if model_cls is not None:
            return model_cls

    raise ImportError(
        "Cannot find a suitable Qwen-VL model class. "
        "Upgrade transformers, e.g. pip install 'transformers>=4.57.0'."
    )


def load_processor_and_model(
    model_id: str,
    dtype_name: str,
    device_map: str,
    attn_implementation: str,
    trust_remote_code: bool,
    load_in_4bit: bool,
) -> Tuple[Any, torch.nn.Module]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    dtype = parse_torch_dtype(dtype_name)
    model_cls = maybe_autoclass(model_id)

    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    kwargs["dtype"] = dtype

    if device_map.lower() != "none":
        kwargs["device_map"] = device_map

    if attn_implementation.lower() != "none":
        kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs.pop("dtype", None)

    try:
        model = model_cls.from_pretrained(model_id, **kwargs)
    except TypeError:
        # Older transformers versions used torch_dtype instead of dtype.
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_cls.from_pretrained(model_id, **kwargs)

    return processor, model


def model_input_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def normalize_prediction(text: str) -> str:
    answer = text.lower().strip()
    answer = answer.replace("aeroplane", "airplane")
    answer = answer.replace("air plane", "airplane")
    answer = answer.replace("car", "automobile")

    # Prefer exact standalone label matches.
    for label in CIFAR10_LABELS:
        if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", answer):
            return label

    # Fallback: first token sometimes is enough after instruction tuning.
    token = re.sub(r"[^a-z]", " ", answer).split()
    return token[0] if token else "unknown"


def macro_f1_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    scores: List[float] = []
    for label in CIFAR10_LABELS:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            scores.append(f1)
    return sum(scores) / len(scores)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - rank) + sorted_values[hi] * (rank - lo)


def generate_one(
    processor: Any,
    model: torch.nn.Module,
    image: Image.Image,
    question: str,
    max_new_tokens: int,
) -> Tuple[str, float, int]:
    prompt = apply_chat_template(processor, question=question, answer=None)
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = to_device(inputs, model_input_device(model))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start

    prompt_len = int(inputs["input_ids"].shape[-1])
    output_ids = generated[0][prompt_len:]
    text = processor.decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return text.strip(), latency, int(output_ids.numel())


def run_eval(args: argparse.Namespace) -> EvalMetrics:
    set_seed(args.seed)
    processor, model = load_processor_and_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()

    dataset = Cifar10QwenVLDataset(
        data_root=args.data_root,
        split="test",
        image_size=args.image_size,
        limit=args.eval_limit,
        seed=args.seed,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    y_true: List[str] = []
    y_pred: List[str] = []
    latencies: List[float] = []
    generated_tokens = 0
    predictions_path = Path(args.output_dir) / "predictions.jsonl"
    ensure_dir(Path(args.output_dir))

    wall_start = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8") as f:
        for sample in tqdm(dataset, desc="eval"):
            raw_text, latency, new_tokens = generate_one(
                processor=processor,
                model=model,
                image=sample["image"],
                question=sample["question"],
                max_new_tokens=args.max_new_tokens,
            )
            pred = normalize_prediction(raw_text)

            y_true.append(sample["label"])
            y_pred.append(pred)
            latencies.append(latency)
            generated_tokens += new_tokens

            f.write(
                json.dumps(
                    {
                        "label": sample["label"],
                        "prediction": pred,
                        "raw_text": raw_text,
                        "latency_ms": latency * 1000,
                        "generated_tokens": new_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_wall = time.perf_counter() - wall_start
    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    peak_mem = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else None
    )

    metrics = EvalMetrics(
        total=total,
        correct=correct,
        accuracy=correct / total if total else 0.0,
        macro_f1=macro_f1_score(y_true, y_pred),
        avg_latency_ms=statistics.mean(latencies) * 1000 if latencies else 0.0,
        p50_latency_ms=percentile(latencies, 0.50) * 1000,
        p95_latency_ms=percentile(latencies, 0.95) * 1000,
        images_per_second=total / total_wall if total_wall > 0 else 0.0,
        generated_tokens_per_second=(
            generated_tokens / sum(latencies) if sum(latencies) > 0 else 0.0
        ),
        total_wall_time_sec=total_wall,
        peak_cuda_memory_mb=peak_mem,
        model_id=args.model_id,
        image_size=args.image_size,
        max_new_tokens=args.max_new_tokens,
    )

    metrics_path = Path(args.output_dir) / "metrics.json"
    metrics_path.write_text(
        json.dumps(asdict(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
    return metrics


class QwenVLCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor
        tokenizer = getattr(processor, "tokenizer", None)
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [sample["image"] for sample in samples]
        full_texts = [
            apply_chat_template(self.processor, sample["question"], sample["answer"])
            for sample in samples
        ]
        prompt_texts = [
            apply_chat_template(self.processor, sample["question"], None)
            for sample in samples
        ]

        inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )

        labels = inputs["input_ids"].clone()
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        # Mask prompt tokens so loss is only computed on the answer label.
        for row, (prompt_text, image) in enumerate(zip(prompt_texts, images)):
            prompt_inputs = self.processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                padding=False,
            )
            prompt_len = int(prompt_inputs["input_ids"].shape[-1])
            labels[row, :prompt_len] = -100

        inputs["labels"] = labels
        return inputs


def run_finetune_lora(args: argparse.Namespace) -> None:
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    set_seed(args.seed)
    processor, model = load_processor_and_model(
        model_id=args.model_id,
        dtype_name=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = Cifar10QwenVLDataset(
        data_root=args.data_root,
        split="train",
        image_size=args.image_size,
        limit=args.train_limit,
        seed=args.seed,
    )
    collator = QwenVLCollator(processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        report_to=[] if args.report_to == "none" else [args.report_to],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    meta = {
        "model_id": args.model_id,
        "train_limit": args.train_limit,
        "image_size": args.image_size,
        "labels": list(CIFAR10_LABELS),
        "question": QUESTION,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": lora_targets,
        },
    }
    Path(args.output_dir, "run_config.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_prepare(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    train = load_cifar10(args.data_root, "train")
    test = load_cifar10(args.data_root, "test")
    train_indices = get_limited_indices(len(train), args.train_limit, args.seed)
    test_indices = get_limited_indices(len(test), args.eval_limit, args.seed)

    train_jsonl = export_qwenvl_split(train, train_indices, "train", output_dir, args.image_size)
    test_jsonl = export_qwenvl_split(test, test_indices, "test", output_dir, args.image_size)

    print(
        json.dumps(
            {
                "train_jsonl": str(train_jsonl),
                "test_jsonl": str(test_jsonl),
                "train_count": len(train_indices),
                "test_count": len(test_indices),
                "format": {
                    "image": "/absolute/path/to/image.png",
                    "conversations": [
                        {"from": "human", "value": "<image>\\n..."},
                        {"from": "gpt", "value": "airplane"},
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="./data", help="CIFAR-10 download/cache root.")
    parser.add_argument("--output-dir", default="./runs/qwen_vl_cifar10")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto", help="Use 'none' to disable device_map.")
    parser.add_argument("--attn-implementation", default="none", help="e.g. flash_attention_2 or none.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen-VL CIFAR-10 fine-tuning/eval starter.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Export CIFAR-10 to Qwen-VL JSONL.")
    add_common_args(prepare)
    prepare.add_argument("--train-limit", type=int, default=None)
    prepare.add_argument("--eval-limit", type=int, default=None)
    prepare.set_defaults(func=run_prepare)

    evaluate = subparsers.add_parser("eval", help="Generative classification evaluation.")
    add_common_args(evaluate)
    add_model_args(evaluate)
    evaluate.add_argument("--eval-limit", type=int, default=200)
    evaluate.add_argument("--max-new-tokens", type=int, default=8)
    evaluate.set_defaults(func=run_eval)

    ft = subparsers.add_parser("finetune_lora", help="LoRA SFT baseline.")
    add_common_args(ft)
    add_model_args(ft)
    ft.add_argument("--train-limit", type=int, default=5000)
    ft.add_argument("--epochs", type=float, default=1.0)
    ft.add_argument("--batch-size", type=int, default=1)
    ft.add_argument("--grad-accum", type=int, default=16)
    ft.add_argument("--learning-rate", type=float, default=2e-5)
    ft.add_argument("--weight-decay", type=float, default=0.0)
    ft.add_argument("--warmup-ratio", type=float, default=0.03)
    ft.add_argument("--lr-scheduler-type", default="cosine")
    ft.add_argument("--logging-steps", type=int, default=10)
    ft.add_argument("--save-steps", type=int, default=500)
    ft.add_argument("--save-total-limit", type=int, default=2)
    ft.add_argument("--num-workers", type=int, default=2)
    ft.add_argument("--optim", default="adamw_torch")
    ft.add_argument("--max-grad-norm", type=float, default=1.0)
    ft.add_argument("--bf16", action="store_true")
    ft.add_argument("--fp16", action="store_true")
    ft.add_argument("--gradient-checkpointing", action="store_true")
    ft.add_argument("--report-to", default="none")
    ft.add_argument("--lora-r", type=int, default=8)
    ft.add_argument("--lora-alpha", type=int, default=16)
    ft.add_argument("--lora-dropout", type=float, default=0.05)
    ft.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module suffixes for PEFT LoRA.",
    )
    ft.set_defaults(func=run_finetune_lora)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
