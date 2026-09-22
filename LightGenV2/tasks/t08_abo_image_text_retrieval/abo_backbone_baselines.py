"""Frozen-backbone ABO retrieval baselines for CLIP, DeepSeek-VL2 and YOLO11s.

This module keeps the published Qwen protocols unchanged:

* easy100 image-to-text: 2,400 test images rank 100 official titles;
* easy100 text-to-image: 100 official titles rank 2,400 test images; and
* ABO-200 image-to-image: 800 queries rank 1,600 enrolled gallery images.

CLIP and DeepSeek are evaluated without fitted parameters. YOLO11s has no text
encoder, so cross-modal retrieval is explicitly a composite baseline: frozen
YOLO11s image features, a frozen CLIP text tower, and one fitted bias-free
linear alignment matrix. The YOLO image-to-image result uses its raw frozen
descriptor and no fitted head.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F


I2T_QUERY = "Retrieve the product title that best describes this product image."
T2I_QUERY = "Retrieve product images that match the following product description."
DOCUMENT = "Represent the user's input."
QWEN_REFERENCES = {
    "image_to_text": {
        "source_commit": "b9a4be689f198f55ee0caaf4e5bcddd4dfc80200",
        "r_at_1": 0.7358333333333333,
    },
    "image_to_image": {
        "source_commit": "5f731fae979ef81a09770c1dd721b5cecd8b3392",
        "r_at_1": 0.85125,
    },
    "text_to_image": {
        "source_commits": [
            "1e218991",
            "6d2930e3a700caee7bd5752da180b7e4136c8858",
        ],
        "hit_at_1": 0.82,
    },
}


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[3]
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _limit_per_key(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any], limit: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        return rows
    counts: Counter[Any] = Counter()
    selected = []
    for row in rows:
        value = key(row)
        if counts[value] < limit:
            selected.append(row)
            counts[value] += 1
    return selected


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _trainable_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _normalized(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    if value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"Expected finite 2-D features, got {tuple(value.shape)}")
    if bool((value.norm(dim=1) < 1e-8).any()):
        raise ValueError("Feature matrix contains a zero descriptor")
    return F.normalize(value, dim=1)


@dataclass
class EncoderBundle:
    name: str
    image_width: int
    text_width: int | None
    encode_images: Callable[[list[Image.Image], str], torch.Tensor]
    encode_texts: Callable[[list[str], str], torch.Tensor] | None
    metadata: dict[str, Any]
    close: Callable[[], None]


def _load_clip(model_path: Path, device: torch.device) -> EncoderBundle:
    import clip

    model, preprocess = clip.load(str(model_path), device=device, jit=False)
    model = model.eval().requires_grad_(False)
    if _trainable_count(model) != 0:
        raise RuntimeError("CLIP backbone is not fully frozen")

    @torch.inference_mode()
    def encode_images(images: list[Image.Image], _instruction: str) -> torch.Tensor:
        batch = torch.stack([preprocess(image) for image in images]).to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            return model.encode_image(batch).float().cpu()

    @torch.inference_mode()
    def encode_texts(texts: list[str], _instruction: str) -> torch.Tensor:
        tokens = clip.tokenize(texts, truncate=True).to(device)
        return model.encode_text(tokens).float().cpu()

    width = int(model.text_projection.shape[1])
    metadata = {
        "implementation": "openai/CLIP",
        "implementation_path": str(Path(clip.__file__).resolve()),
        "checkpoint": str(model_path.resolve()),
        "backbone_total_parameters": _parameter_count(model),
        "backbone_trainable_parameters": 0,
        "descriptor": "native normalized CLIP image/text embedding",
        "preprocessing": "official OpenAI CLIP ViT-B/32 preprocessing",
    }

    def close() -> None:
        nonlocal model
        del model
        torch.cuda.empty_cache()

    return EncoderBundle("clip_vit_b32", width, width, encode_images, encode_texts, metadata, close)


def _letterbox(image: Image.Image, size: int) -> Image.Image:
    contained = ImageOps.contain(image, (size, size), method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return canvas


def _unique_parameter_count(modules: Iterable[nn.Module]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                total += parameter.numel()
    return total


def _load_yolo_clip(
    yolo_path: Path, clip_path: Path, device: torch.device, image_size: int
) -> EncoderBundle:
    import clip
    import ultralytics
    from ultralytics import YOLO

    wrapper = YOLO(str(yolo_path))
    yolo = wrapper.model.to(device).eval().requires_grad_(False)
    layers = getattr(yolo, "model", None)
    if layers is None or len(layers) <= 10 or layers[10].__class__.__name__ != "C2PSA":
        raise RuntimeError("Expected YOLO11s C2PSA at layer 10")
    feature_layer = layers[10]
    capture: dict[str, torch.Tensor] = {}

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        capture["feature"] = output

    handle = feature_layer.register_forward_hook(hook)
    clip_model, _ = clip.load(str(clip_path), device=device, jit=False)
    clip_model = clip_model.eval().requires_grad_(False)
    if _trainable_count(yolo) or _trainable_count(clip_model):
        raise RuntimeError("YOLO/CLIP composite backbone is not fully frozen")

    @torch.inference_mode()
    def encode_images(images: list[Image.Image], _instruction: str) -> torch.Tensor:
        arrays = [np.asarray(_letterbox(image, image_size), dtype=np.uint8).copy() for image in images]
        batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
        batch = batch.to(device=device, dtype=torch.float32).div_(255.0)
        capture.clear()
        with torch.autocast("cuda", dtype=torch.float16):
            yolo(batch)
        feature_map = capture.get("feature")
        if feature_map is None:
            raise RuntimeError("YOLO11s feature hook did not run")
        return feature_map.float().mean(dim=(-2, -1)).cpu()

    @torch.inference_mode()
    def encode_texts(texts: list[str], _instruction: str) -> torch.Tensor:
        return clip_model.encode_text(clip.tokenize(texts, truncate=True).to(device)).float().cpu()

    text_modules = [
        clip_model.transformer,
        clip_model.token_embedding,
        clip_model.ln_final,
    ]
    text_parameters = _unique_parameter_count(text_modules)
    text_parameters += int(clip_model.positional_embedding.numel())
    text_parameters += int(clip_model.text_projection.numel())
    metadata = {
        "implementation": "ultralytics/ultralytics + openai/CLIP text tower",
        "ultralytics_version": ultralytics.__version__,
        "yolo_checkpoint": str(yolo_path.resolve()),
        "clip_checkpoint": str(clip_path.resolve()),
        "yolo_total_parameters": _parameter_count(yolo),
        "clip_text_parameters": text_parameters,
        "backbone_total_parameters": _parameter_count(yolo) + text_parameters,
        "backbone_trainable_parameters": 0,
        "image_descriptor": "global average of YOLO11s layer-10 C2PSA output",
        "text_descriptor": "native frozen CLIP text embedding",
        "preprocessing": f"EXIF RGB, aspect-preserving {image_size} square letterbox, fill 114",
    }

    def close() -> None:
        nonlocal yolo, clip_model
        handle.remove()
        del yolo
        del clip_model
        torch.cuda.empty_cache()

    return EncoderBundle("yolo11s_clip_text", 512, 512, encode_images, encode_texts, metadata, close)


def _load_deepseek(model_path: Path, device: torch.device) -> EncoderBundle:
    from deepseek_vl2.models import DeepseekVLV2Processor
    from transformers import AutoModelForCausalLM

    processor = DeepseekVLV2Processor.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device).eval().requires_grad_(False)
    if _trainable_count(model) != 0:
        raise RuntimeError("DeepSeek-VL2 backbone is not fully frozen")
    width = int(model.config.language_config.hidden_size)

    @torch.inference_mode()
    def encode_one(*, image: Image.Image | None, text: str | None, instruction: str) -> torch.Tensor:
        if (image is None) == (text is None):
            raise ValueError("Exactly one of image/text must be supplied")
        content = "<image>" if image is not None else str(text)
        conversation = [
            {"role": "<|User|>", "content": content, "images": ["image.png"] if image is not None else []},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepared = processor(
            conversations=conversation,
            images=[image] if image is not None else [],
            force_batchify=True,
            system_prompt=instruction,
        ).to(device, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs_embeds = model.prepare_inputs_embeds(**prepared)
            outputs = model.language.model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepared.attention_mask,
                use_cache=False,
                return_dict=True,
            )
        mask = prepared.attention_mask.bool()
        positions = torch.arange(mask.shape[1], device=device).expand_as(mask)
        last = positions.masked_fill(~mask, -1).amax(1)
        return outputs.last_hidden_state[torch.arange(len(mask), device=device), last].float().cpu()[0]

    def encode_images(images: list[Image.Image], instruction: str) -> torch.Tensor:
        return torch.stack([encode_one(image=image, text=None, instruction=instruction) for image in images])

    def encode_texts(texts: list[str], instruction: str) -> torch.Tensor:
        return torch.stack([encode_one(image=None, text=text, instruction=instruction) for text in texts])

    metadata = {
        "implementation": "deepseek-ai/DeepSeek-VL2",
        "checkpoint": str(model_path.resolve()),
        "backbone_total_parameters": _parameter_count(model),
        "backbone_trainable_parameters": 0,
        "descriptor": "final decoder layer at the last valid prompt token",
        "preprocessing": "official DeepSeek-VL2 processor defaults",
        "hidden_width": width,
    }

    def close() -> None:
        nonlocal model
        del model
        torch.cuda.empty_cache()

    return EncoderBundle("deepseek_vl2_tiny", width, width, encode_images, encode_texts, metadata, close)


def load_encoder(args: argparse.Namespace, device: torch.device) -> EncoderBundle:
    if args.model_kind == "clip":
        return _load_clip(args.model, device)
    if args.model_kind == "deepseek":
        return _load_deepseek(args.model, device)
    if args.model_kind == "yolo11s":
        if args.clip_model is None:
            raise ValueError("YOLO cross-modal baseline requires --clip-model")
        return _load_yolo_clip(args.model, args.clip_model, device, args.yolo_image_size)
    raise AssertionError(args.model_kind)


def _load_images(rows: list[dict[str, Any]], data_root: Path) -> list[Image.Image]:
    images = []
    for row in rows:
        path = (data_root / row["image_path"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            images.append(ImageOps.exif_transpose(source).convert("RGB"))
    return images


def _extract_images(
    *,
    rows: list[dict[str, Any]],
    data_root: Path,
    output: Path,
    bundle: EncoderBundle,
    instruction: str,
    batch_size: int,
) -> torch.Tensor:
    identifiers = [str(row["sample_id"]) for row in rows]
    identity = {
        "model": bundle.name,
        "model_metadata": bundle.metadata,
        "instruction": instruction,
        "sample_ids": identifiers,
    }
    if output.is_file():
        cached = torch.load(output, map_location="cpu", weights_only=False)
        if cached.get("identity") == identity:
            print(f"[resume] {output} {len(rows)} images", flush=True)
            return cached["features"].float()
    parts = output.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    features: list[torch.Tensor] = []
    for start in range(0, len(rows), batch_size):
        stop = min(len(rows), start + batch_size)
        shard = parts / f"rows_{start:05d}_{stop:05d}.pt"
        shard_ids = identifiers[start:stop]
        if shard.is_file():
            cached = torch.load(shard, map_location="cpu", weights_only=False)
            if cached.get("model") == bundle.name and cached.get("instruction") == instruction and cached.get("sample_ids") == shard_ids:
                features.append(cached["features"].float())
                print(f"[resume] {stop}/{len(rows)}", flush=True)
                continue
        images = _load_images(rows[start:stop], data_root)
        value = bundle.encode_images(images, instruction).float().cpu()
        if tuple(value.shape) != (stop - start, bundle.image_width):
            raise RuntimeError(f"Unexpected image feature shape: {tuple(value.shape)}")
        _atomic_save(shard, {"model": bundle.name, "instruction": instruction, "sample_ids": shard_ids, "features": value.half()})
        features.append(value)
        print(f"[extract] {stop}/{len(rows)}", flush=True)
    result = torch.cat(features)
    _atomic_save(output, {"identity": identity, "features": result.half()})
    return result


def _candidate_metrics(scores: torch.Tensor, true_labels: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    order = scores.argsort(dim=1, descending=True, stable=True)
    ranks = order.eq(true_labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    report = {
        "query_count": len(scores),
        "candidate_count": scores.shape[1],
        "r_at_1": float((ranks <= 1).float().mean()),
        "r_at_5": float((ranks <= 5).float().mean()),
        "r_at_10": float((ranks <= 10).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "mean_rank": float(ranks.float().mean()),
    }
    return report, order


def _gallery_metrics(
    scores: torch.Tensor, query_labels: torch.Tensor, gallery_labels: torch.Tensor
) -> tuple[dict[str, float], torch.Tensor]:
    order = scores.argsort(dim=1, descending=True, stable=True)
    relevant = gallery_labels[order].eq(query_labels[:, None])
    positives = relevant.sum(1)
    if not bool((positives > 0).all()):
        raise ValueError("At least one query has no relevant gallery item")
    rank = torch.arange(1, scores.shape[1] + 1, dtype=torch.float64)[None]
    precision = relevant.cumsum(1) / rank
    first = torch.where(relevant, rank, torch.inf).amin(1)
    report = {
        "query_count": len(scores),
        "gallery_count": scores.shape[1],
        "relevant_candidates_per_query": float(positives.float().mean()),
        "mrr": float((1.0 / first).mean()),
        "map": float(((precision * relevant).sum(1) / positives).mean()),
    }
    for k in (1, 5, 10):
        report[f"hit_at_{k}"] = float(relevant[:, :k].any(1).double().mean())
        report[f"recall_at_{k}"] = float((relevant[:, :k].sum(1) / positives).double().mean())
    return report, order


def _fit_yolo_alignment(
    image_features: torch.Tensor,
    labels: torch.Tensor,
    text_features: torch.Tensor,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[nn.Linear, list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    image_features = _normalized(image_features)
    text_features = _normalized(text_features).to(device)
    layer = nn.Linear(image_features.shape[1], text_features.shape[1], bias=False).to(device)
    nn.init.orthogonal_(layer.weight)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(image_features), generator=generator)
        losses = []
        correct = total = 0
        layer.train()
        for start in range(0, len(permutation), batch_size):
            index = permutation[start:start + batch_size]
            images = image_features[index].to(device)
            target = labels[index].to(device)
            logits = _normalized(layer(images)) @ text_features.T / 0.07
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            correct += int(logits.argmax(1).eq(target).sum())
            total += len(index)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "train_accuracy": correct / total}
        history.append(row)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"[adapter] epoch={epoch + 1} loss={row['loss']:.6f} accuracy={row['train_accuracy']:.4f}", flush=True)
    return layer.eval().requires_grad_(False), history


def _easy100(args: argparse.Namespace, bundle: EncoderBundle, device: torch.device) -> dict[str, Any]:
    root = args.data.resolve()
    titles = _read_csv(root / "titles.csv")
    train = _read_csv(root / "train.csv")
    test = _read_csv(root / "test.csv")
    if len(titles) != 100 or len(train) != 4800 or len(test) != 2400:
        raise RuntimeError("Expected ABO easy100 contract: 100 titles, 4800 train, 2400 test")
    for rows in (train, test):
        for row in rows:
            row["label"] = int(row["label"])
    train = _limit_per_key(train, lambda row: row["label"], args.limit_per_label)
    test = _limit_per_key(test, lambda row: row["label"], args.limit_per_label)
    title_texts = [row["title"] for row in titles]
    if bundle.encode_texts is None:
        raise RuntimeError("Selected model lacks a text encoder")
    title_instruction = T2I_QUERY if args.model_kind == "deepseek" else DOCUMENT
    title_features = bundle.encode_texts(title_texts, title_instruction)
    test_features = _extract_images(
        rows=test,
        data_root=root,
        output=args.output / "easy100_test_features.pt",
        bundle=bundle,
        instruction=I2T_QUERY if args.model_kind == "deepseek" else DOCUMENT,
        batch_size=args.batch_size,
    )
    adapter_history: list[dict[str, float]] | None = None
    adapter_parameters = 0
    if args.model_kind == "yolo11s":
        train_features = _extract_images(
            rows=train,
            data_root=root,
            output=args.output / "easy100_train_features.pt",
            bundle=bundle,
            instruction=DOCUMENT,
            batch_size=args.batch_size,
        )
        adapter, adapter_history = _fit_yolo_alignment(
            train_features,
            torch.tensor([row["label"] for row in train]),
            title_features,
            device=device,
            epochs=args.adapter_epochs,
            batch_size=args.adapter_batch_size,
            learning_rate=args.adapter_lr,
            seed=args.seed,
        )
        adapter_parameters = _parameter_count(adapter)
        _atomic_save(args.output / "yolo_clip_alignment.pt", {
            "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
            "image_width": bundle.image_width,
            "text_width": bundle.text_width,
            "training_rows": len(train),
            "epochs": args.adapter_epochs,
            "seed": args.seed,
        })
        with torch.inference_mode():
            test_features = adapter(_normalized(test_features).to(device)).cpu()
    image_vectors = _normalized(test_features)
    text_vectors = _normalized(title_features)
    test_labels = torch.tensor([row["label"] for row in test])
    title_labels = torch.tensor([int(row["label"]) for row in titles])
    i2t, i2t_order = _candidate_metrics(image_vectors @ text_vectors.T, test_labels)
    t2i, t2i_order = _gallery_metrics(text_vectors @ image_vectors.T, title_labels, test_labels)
    _write_json(args.output / "easy100_predictions.json", {
        "image_to_text_top1": [titles[index]["product_id"] for index in i2t_order[:, 0].tolist()],
        "text_to_image_top10": [[test[index]["sample_id"] for index in row] for row in t2i_order[:, :10].tolist()],
    })
    return {
        "protocol": "ABO easy100 fixed official-title closed catalogue",
        "train_images": len(train),
        "test_images": len(test),
        "titles": len(titles),
        "image_to_text": i2t,
        "text_to_image": t2i,
        "fitted_adapter": args.model_kind == "yolo11s",
        "adapter_trainable_parameters": adapter_parameters,
        "adapter_history": adapter_history,
        "data_sha256": {name: _sha256(root / name) for name in ("titles.csv", "train.csv", "test.csv")},
    }


def _image_to_image(args: argparse.Namespace, bundle: EncoderBundle) -> dict[str, Any]:
    root = args.data.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "abo200_enrolled_sku_hash8train4query_v1":
        raise ValueError("Unexpected ABO-200 image retrieval protocol")
    gallery = [dict(row) for row in manifest["rows"] if row["split"] == "train"]
    queries = [dict(row) for row in manifest["rows"] if row["split"] == "query"]
    if len(gallery) != 1600 or len(queries) != 800:
        raise RuntimeError("Expected 1600 enrolled gallery and 800 query rows")
    gallery = _limit_per_key(gallery, lambda row: row["product_id"], args.limit_per_label)
    queries = _limit_per_key(queries, lambda row: row["product_id"], args.limit_per_label)
    rows = gallery + queries
    features = _extract_images(
        rows=rows,
        data_root=root,
        output=args.output / "abo200_image_features.pt",
        bundle=bundle,
        instruction=DOCUMENT,
        batch_size=args.batch_size,
    )
    vectors = _normalized(features)
    gallery_vectors = vectors[: len(gallery)]
    query_vectors = vectors[len(gallery):]
    product_ids = sorted({row["product_id"] for row in gallery})
    label = {product_id: index for index, product_id in enumerate(product_ids)}
    gallery_labels = torch.tensor([label[row["product_id"]] for row in gallery])
    query_labels = torch.tensor([label[row["product_id"]] for row in queries])
    metrics, order = _gallery_metrics(query_vectors @ gallery_vectors.T, query_labels, gallery_labels)
    _write_json(args.output / "abo200_predictions.json", [
        {
            "sample_id": row["sample_id"],
            "product_id": row["product_id"],
            "top1_sample_id": gallery[int(order[index, 0])]["sample_id"],
            "top1_product_id": gallery[int(order[index, 0])]["product_id"],
        }
        for index, row in enumerate(queries)
    ])
    return {
        "protocol": manifest["protocol"],
        "manifest_sha256": _sha256(args.manifest),
        "gallery_images": len(gallery),
        "query_images": len(queries),
        "fitted_adapter": False,
        "metrics": metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size < 1 or args.limit_per_label < 0:
        raise ValueError("Invalid batch/limit")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    started = time.time()
    bundle = load_encoder(args, device)
    status = {
        "status": "running",
        "source_commit": _git_head(),
        "command": sys.argv,
        "pid": os.getpid(),
        "model_kind": args.model_kind,
        "model": str(args.model.resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(0),
        "backbone_frozen": True,
        "model_metadata": bundle.metadata,
        "qwen_references": QWEN_REFERENCES,
    }
    _write_json(args.output / "status.json", status)
    try:
        results: dict[str, Any] = {}
        if args.task in ("easy100", "all"):
            results["easy100"] = _easy100(args, bundle, device)
        if args.task in ("image_to_image", "all"):
            if args.manifest is None:
                raise ValueError("image_to_image requires --manifest")
            results["image_to_image"] = _image_to_image(args, bundle)
        report = {
            **status,
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "limit_per_label": args.limit_per_label,
            "results": results,
        }
        _write_json(args.output / "report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report
    except BaseException as error:
        status.update(status="failed_or_interrupted", error=repr(error), elapsed_seconds=time.time() - started)
        _write_json(args.output / "status.json", status)
        raise
    finally:
        bundle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("clip", "deepseek", "yolo11s"), required=True)
    parser.add_argument("--task", choices=("easy100", "image_to_image", "all"), default="all")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--clip-model", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-per-label", type=int, default=0)
    parser.add_argument("--yolo-image-size", type=int, default=640)
    parser.add_argument("--adapter-epochs", type=int, default=50)
    parser.add_argument("--adapter-batch-size", type=int, default=256)
    parser.add_argument("--adapter-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
