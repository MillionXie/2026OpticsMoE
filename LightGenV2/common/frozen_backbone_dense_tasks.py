"""Frozen CLIP, YOLO11s and DeepSeek-VL2-Tiny baselines for dense tasks.

The pretrained backbone is always frozen.  Feature extraction and readout
training are separate commands so the backbone never receives task gradients.
The task heads and metrics reuse the audited Qwen implementations for LSP,
SALICON and the layered OpenMoji benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


TASKS = ("lsp", "salicon", "openmoji")
MODELS = ("clip", "yolo11s", "deepseek")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_ids(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_head() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class SourceDataset(Dataset):
    def __init__(self, task: str, split: str, args: argparse.Namespace):
        self.task = task
        self.split = split
        self.settings = None
        if task == "lsp":
            from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.datasets import prepare_lsp, LSPPoseDataset
            from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.settings import load_settings
            settings = load_settings(args.lsp_config)
            settings.data_root = args.lsp_data.resolve()
            settings.download = False
            settings.output_dir = args.output.parent / "_dataset_audit_lsp"
            settings.augmentation_enabled = False
            settings.num_workers = 0
            bundle = prepare_lsp(settings, persist=False)
            records = bundle.train if split == "train" else bundle.test
            self.base = LSPPoseDataset(records, settings, training=False)
            self.settings = settings
        elif task == "salicon":
            from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.datasets import prepare_salicon, SALICONSaliencyDataset
            from LightGenV2.tasks.t03_saliency.settings import load_settings
            settings = load_settings(args.salicon_config)
            settings.data_root = args.salicon_data.resolve()
            settings.output_dir = args.output.parent / "_dataset_audit_salicon"
            settings.augmentation_enabled = False
            settings.num_workers = 0
            bundle = prepare_salicon(settings, persist=False)
            records = bundle.train_records if split == "train" else bundle.validation_records
            self.base = SALICONSaliencyDataset(records, settings, training=False)
            self.settings = settings
        elif task == "openmoji":
            manifest = args.openmoji_data.resolve() / f"{split}.jsonl"
            self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.data_root = args.openmoji_data.resolve()
        else:
            raise ValueError(task)

    def __len__(self) -> int:
        return len(self.rows) if self.task == "openmoji" else len(self.base)

    def sample_id(self, index: int) -> str:
        if self.task == "openmoji":
            return str(self.rows[index]["sample_id"])
        return str(self.base.records[index].sample_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.task != "openmoji":
            item = self.base[index]
            sample_id = item.get("sample_id", item.get("sample_ids"))
            return {"sample_id": str(sample_id), "image": item["image"], "target": item}
        row = self.rows[index]
        path = self.data_root / row["relative_dir"] / row["files"]["source"]
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        return {"sample_id": row["sample_id"], "image": image, "text": row["instruction"], "target": row}


class Encoder:
    def __init__(self, kind: str, args: argparse.Namespace, device: torch.device):
        self.kind = kind
        self.device = device
        self.handles: list[Any] = []
        self.capture: dict[str, torch.Tensor] = {}
        if kind == "clip":
            import clip
            self.clip_module = clip
            self.model, self.preprocess = clip.load(str(args.model), device=device, jit=False)
            self.model.eval().requires_grad_(False)
            self.image_width = 768
            self.text_width = int(self.model.text_projection.shape[1])
            self.metadata = {
                "backbone": "OpenAI CLIP ViT-B/32",
                "checkpoint": str(args.model.resolve()),
                "image_feature": "final 7x7 ViT patch tokens interpolated to 14x14",
                "text_feature": "native final CLIP text embedding as one token",
            }
        elif kind == "yolo11s":
            import clip
            from ultralytics import YOLO
            self.clip_module = clip
            self.wrapper = YOLO(str(args.model))
            self.model = self.wrapper.model.to(device).eval().requires_grad_(False)
            layers = self.model.model
            if layers[10].__class__.__name__ != "C2PSA":
                raise RuntimeError("YOLO11s layer 10 is not C2PSA")
            self.handles.append(layers[10].register_forward_hook(self._capture))
            self.text_model, _ = clip.load(str(args.clip_model), device=device, jit=False)
            self.text_model.eval().requires_grad_(False)
            self.image_width = 512
            self.text_width = int(self.text_model.text_projection.shape[1])
            self.yolo_size = int(args.yolo_image_size)
            self.metadata = {
                "backbone": "Ultralytics YOLO11s + frozen CLIP text tower for OpenMoji",
                "checkpoint": str(args.model.resolve()),
                "image_feature": "YOLO11s layer-10 C2PSA map interpolated to 14x14",
                "text_feature": "native final CLIP text embedding as one token",
            }
        elif kind == "deepseek":
            from deepseek_vl2.models import DeepseekVLV2Processor
            from transformers import AutoModelForCausalLM
            self.processor = DeepseekVLV2Processor.from_pretrained(str(args.model))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(args.model), local_files_only=True, trust_remote_code=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            ).to(device).eval().requires_grad_(False)
            self.image_width = int(self.model.config.vision_config.width)
            self.text_width = int(self.model.config.language_config.hidden_size)
            self.metadata = {
                "backbone": "DeepSeek-VL2-Tiny",
                "checkpoint": str(args.model.resolve()),
                "image_feature": "final SigLIP-So400M global-view 27x27 tokens interpolated to 14x14",
                "text_feature": "final DeepSeek decoder instruction-token hidden states",
            }
        else:
            raise ValueError(kind)
        self.model.eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Backbone freeze assertion failed")
        self.metadata.update({
            "backbone_total_parameters": count_parameters(self.model),
            "backbone_trainable_parameters": 0,
            "image_width": self.image_width,
            "text_width": self.text_width,
        })

    def _capture(self, _module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        self.capture["feature"] = output

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if self.kind == "clip":
            batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
            visual = self.model.visual
            with torch.autocast("cuda", dtype=torch.float16):
                value = visual.conv1(batch.type(visual.conv1.weight.dtype))
                value = value.reshape(value.shape[0], value.shape[1], -1).permute(0, 2, 1)
                cls = visual.class_embedding.to(value.dtype)[None, None].expand(len(value), 1, -1)
                value = torch.cat((cls, value), dim=1)
                value = visual.ln_pre(value + visual.positional_embedding.to(value.dtype))
                value = visual.transformer(value.permute(1, 0, 2)).permute(1, 0, 2)
                value = visual.ln_post(value[:, 1:])
            side = int(math.isqrt(value.shape[1]))
            value = value.transpose(1, 2).reshape(len(images), self.image_width, side, side)
        elif self.kind == "yolo11s":
            arrays = []
            for image in images:
                resized = image.resize((self.yolo_size, self.yolo_size), Image.Resampling.BILINEAR)
                arrays.append(np.asarray(resized, dtype=np.uint8).copy())
            batch = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device).float().div_(255)
            self.capture.clear()
            with torch.autocast("cuda", dtype=torch.float16):
                self.model(batch)
            value = self.capture["feature"]
        else:
            prepared_images = []
            for image in images:
                conversation = [
                    {"role": "<|User|>", "content": "<image>", "images": ["image.png"]},
                    {"role": "<|Assistant|>", "content": ""},
                ]
                prepared = self.processor(
                    conversations=conversation, images=[image], force_batchify=True, system_prompt=""
                )
                prepared_images.append(prepared.images[0])
            batch = torch.stack(prepared_images).to(self.device, dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = self.model.vision(batch.flatten(0, 1))
            value = value.reshape(len(images), batch.shape[1], 729, self.image_width)[:, 0]
            value = value.transpose(1, 2).reshape(len(images), self.image_width, 27, 27)
        value = F.interpolate(value.float(), size=(14, 14), mode="bilinear", align_corners=False)
        if tuple(value.shape) != (len(images), self.image_width, 14, 14):
            raise RuntimeError(f"Unexpected spatial feature shape {tuple(value.shape)}")
        return value.cpu().half()

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> list[torch.Tensor]:
        if self.kind == "clip":
            tokens = self.clip_module.tokenize(texts, truncate=True).to(self.device)
            values = self.model.encode_text(tokens).float().cpu().half()
            return [value[None] for value in values]
        if self.kind == "yolo11s":
            tokens = self.clip_module.tokenize(texts, truncate=True).to(self.device)
            values = self.text_model.encode_text(tokens).float().cpu().half()
            return [value[None] for value in values]
        tokenizer = self.processor.tokenizer
        tokens = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = self.model.language.model(
                input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"],
                use_cache=False, return_dict=True,
            ).last_hidden_state
        return [output[i, tokens["attention_mask"][i].bool()].float().cpu().half() for i in range(len(texts))]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        del self.model
        if hasattr(self, "text_model"):
            del self.text_model
        torch.cuda.empty_cache()


def extract(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    seed_all(args.seed)
    encoder = Encoder(args.model_kind, args, device)
    try:
        payload: dict[str, Any] = {
            "schema_version": 1,
            "task": args.task,
            "model_kind": args.model_kind,
            "model_metadata": encoder.metadata,
            "git_commit": git_head(),
            "splits": {},
        }
        for split in ("train", "test"):
            dataset = SourceDataset(args.task, split, args)
            identifiers = [dataset.sample_id(index) for index in range(len(dataset))]
            parts = args.output.with_suffix(f".{split}.parts")
            parts.mkdir(parents=True, exist_ok=True)
            feature_chunks: list[torch.Tensor] = []
            text_values: list[torch.Tensor] = []
            started = time.perf_counter()
            for start in range(0, len(dataset), args.batch_size):
                stop = min(len(dataset), start + args.batch_size)
                shard = parts / f"rows_{start:05d}_{stop:05d}.pt"
                expected = identifiers[start:stop]
                cached = torch.load(shard, map_location="cpu", weights_only=False) if shard.is_file() else None
                if cached and cached.get("sample_ids") == expected and cached.get("model_kind") == args.model_kind:
                    feature_chunks.append(cached["spatial"])
                    text_values.extend(cached.get("text", []))
                    print(f"[resume {args.task}/{args.model_kind}/{split}] {stop}/{len(dataset)}", flush=True)
                    continue
                rows = [dataset[index] for index in range(start, stop)]
                spatial = encoder.encode_images([row["image"] for row in rows])
                texts = encoder.encode_texts([row["text"] for row in rows]) if args.task == "openmoji" else []
                part = {"model_kind": args.model_kind, "sample_ids": expected, "spatial": spatial, "text": texts}
                atomic_save(shard, part)
                feature_chunks.append(spatial)
                text_values.extend(texts)
                print(f"[extract {args.task}/{args.model_kind}/{split}] {stop}/{len(dataset)}", flush=True)
            spatial = torch.cat(feature_chunks)
            if spatial.shape[0] != len(dataset):
                raise RuntimeError("Feature cache row count mismatch")
            payload["splits"][split] = {
                "sample_ids": identifiers,
                "sample_order_sha256": sha256_ids(identifiers),
                "spatial": spatial,
                "text": text_values,
                "seconds": time.perf_counter() - started,
            }
        atomic_save(args.output, payload)
        write_json(args.output.with_suffix(".json"), {
            "schema_version": 1, "task": args.task, "model_kind": args.model_kind,
            "model_metadata": encoder.metadata,
            "split_counts": {key: len(value["sample_ids"]) for key, value in payload["splits"].items()},
            "spatial_shapes": {key: list(value["spatial"].shape) for key, value in payload["splits"].items()},
            "backbone_frozen": True, "git_commit": payload["git_commit"],
        })
    finally:
        encoder.close()


class CachedTaskDataset(Dataset):
    def __init__(self, source: SourceDataset, cache_split: dict[str, Any]):
        self.source = source
        self.spatial = cache_split["spatial"]
        self.text = cache_split.get("text", [])
        expected = [source.sample_id(index) for index in range(len(source))]
        if cache_split["sample_ids"] != expected:
            raise RuntimeError("Cache/source sample order mismatch")

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        value = self.source[index]
        target = value["target"]
        if self.source.task == "lsp":
            return {
                "spatial": self.spatial[index], "heatmaps": target["heatmaps"],
                "keypoints": target["keypoints"], "visible": target["visible"],
                "torso_scale": target["torso_scale"], "head_scale": target["head_scale"],
                "sample_id": value["sample_id"],
            }
        if self.source.task == "salicon":
            return {
                "spatial": self.spatial[index], "density": target["density"],
                "fixation": target["fixation"], "sample_id": value["sample_id"],
            }
        source_grid = torch.tensor(target["source_grid"])
        target_grid = torch.tensor(target["target_grid"])
        return {
            "source_image": self.spatial[index], "prompt_hidden": self.text[index],
            "source_grid": source_grid, "target_grid": target_grid,
            "edit_grid": source_grid.ne(target_grid).float(),
            "preserve_grid": source_grid.eq(target_grid).float(),
            "task_index": torch.tensor(target["task_index"]),
            "sample_id": target["sample_id"], "task": target["task"],
            "instruction": target["instruction"], "program": target["program"],
        }


def collate_lsp(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("spatial", "heatmaps", "keypoints", "visible", "torso_scale", "head_scale")
    return {key: torch.stack([row[key] for row in rows]) for key in keys} | {"sample_id": [row["sample_id"] for row in rows]}


def collate_salicon(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "spatial": torch.stack([row["spatial"] for row in rows]),
        "density": torch.stack([row["density"] for row in rows]),
        "fixation": torch.stack([row["fixation"] for row in rows]),
        "sample_id": [row["sample_id"] for row in rows],
    }


def save_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_loaders(args: argparse.Namespace, cache: dict[str, Any], collate: Any) -> tuple[DataLoader, DataLoader]:
    train = CachedTaskDataset(SourceDataset(args.task, "train", args), cache["splits"]["train"])
    test = CachedTaskDataset(SourceDataset(args.task, "test", args), cache["splits"]["test"])
    generator = torch.Generator().manual_seed(args.seed)
    common = {"num_workers": args.num_workers, "pin_memory": True, "collate_fn": collate}
    return (
        DataLoader(train, batch_size=args.train_batch_size, shuffle=True, generator=generator, **common),
        DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **common),
    )


@torch.no_grad()
def evaluate_lsp(head: nn.Module, loader: DataLoader, device: torch.device, image_size: int) -> dict[str, Any]:
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.metrics import PoseMetricAccumulator
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import masked_heatmap_mse, masked_coordinate_loss
    head.eval()
    accumulator = PoseMetricAccumulator()
    total_hm = total_coord = 0.0
    count = 0
    for batch in loader:
        spatial = batch["spatial"].to(device, non_blocking=True)
        target = batch["heatmaps"].to(device, non_blocking=True)
        keypoints = batch["keypoints"].to(device, non_blocking=True)
        visible = batch["visible"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            prediction = head(spatial)
        hm = masked_heatmap_mse(prediction, target, visible)
        coord = masked_coordinate_loss(prediction, keypoints, visible, image_size)
        n = len(spatial); count += n; total_hm += float(hm) * n; total_coord += float(coord) * n
        accumulator.update(prediction, keypoints, visible, batch["torso_scale"], batch["head_scale"], image_size)
    return {"samples": count, "heatmap_loss": total_hm / count, "coordinate_loss": total_coord / count} | accumulator.compute()


def train_lsp(args: argparse.Namespace, cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.modeling import DeconvPoseHead
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16.losses import masked_heatmap_mse, masked_coordinate_loss
    train_loader, test_loader = make_loaders(args, cache, collate_lsp)
    width = int(cache["model_metadata"]["image_width"])
    head = DeconvPoseHead(width, projection_dim=128, body_channels=(128, 128), up_channels=(128, 128)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)
    best_loss = math.inf; best_epoch = -1; history = []
    for epoch in range(1, args.epochs + 1):
        head.train(); sums = defaultdict(float); samples = 0
        for batch in train_loader:
            spatial = batch["spatial"].to(device, non_blocking=True)
            target = batch["heatmaps"].to(device, non_blocking=True)
            keypoints = batch["keypoints"].to(device, non_blocking=True)
            visible = batch["visible"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = head(spatial)
                hm = masked_heatmap_mse(prediction, target, visible)
                coord = masked_coordinate_loss(prediction, keypoints, visible, 224)
                loss = hm + 0.1 * coord
            loss.backward(); optimizer.step()
            n = len(spatial); samples += n; sums["loss"] += float(loss) * n
            sums["heatmap_loss"] += float(hm) * n; sums["coordinate_loss"] += float(coord) * n
        train_metrics = {key: value / samples for key, value in sums.items()} | {"samples": samples}
        test_metrics = evaluate_lsp(head, test_loader, device, 224)
        if train_metrics["loss"] < best_loss:
            best_loss = train_metrics["loss"]; best_epoch = epoch
            atomic_save(args.run_dir / "best_checkpoint.pt", {"epoch": epoch, "head": head.state_dict(), "train": train_metrics, "test": test_metrics})
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"test_{k}": v for k, v in test_metrics.items() if not isinstance(v, dict)}})
        save_history(args.run_dir / "training_history.csv", history)
        print(f"[LSP {args.model_kind}] epoch={epoch}/{args.epochs} test_PCK={test_metrics['pck_at_0.2_torso']:.4f} best_train_loss={best_loss:.5f}@{best_epoch}", flush=True)
    selected = torch.load(args.run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(selected["head"])
    metrics = evaluate_lsp(head, test_loader, device, 224)
    return {"selected_epoch": selected["epoch"], "metrics": metrics, "head_parameters": count_parameters(head), "selection": "minimum training loss"}


@torch.no_grad()
def evaluate_salicon(head: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import SaliencyAccumulator
    head.eval(); accumulator = SaliencyAccumulator()
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = head(batch["spatial"].to(device, non_blocking=True))
        accumulator.update(logits, batch["density"].to(device), batch["fixation"].to(device))
    return accumulator.compute()


def train_salicon(args: argparse.Namespace, cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from LightGenV2.tasks.t03_saliency.aligned_baseline import AlignedReadout
    from LightGenV2.tasks.t03_saliency.settings import load_settings
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import saliency_loss
    settings = load_settings(args.salicon_config)
    train_loader, test_loader = make_loaders(args, cache, collate_salicon)
    width = int(cache["model_metadata"]["image_width"])
    head = AlignedReadout(width).to(device)
    optimizer = torch.optim.AdamW([
        {"params": [*head.input_adapter.parameters(), *head.input_norm.parameters()], "lr": 1e-4},
        {"params": head.decoder.parameters(), "lr": 3e-4},
    ], weight_decay=0.0)
    best = -math.inf; history = []
    for epoch in range(1, args.epochs + 1):
        head.train(); sums = defaultdict(float); samples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = head(batch["spatial"].to(device, non_blocking=True))
                loss, pieces = saliency_loss(logits, batch["density"].to(device), batch["fixation"].to(device), settings)
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
            n = len(logits); samples += n; sums["loss"] += float(loss) * n
            for key, value in pieces.items(): sums[key] += float(value) * n
        train_metrics = {key: value / samples for key, value in sums.items()} | {"samples": samples}
        test_metrics = evaluate_salicon(head, test_loader, device) if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs else None
        if test_metrics is not None and test_metrics["cc"] > best:
            best = float(test_metrics["cc"])
            atomic_save(args.run_dir / "best_checkpoint.pt", {"epoch": epoch, "head": head.state_dict(), "train": train_metrics, "test": test_metrics})
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **({f"test_{k}": v for k, v in test_metrics.items()} if test_metrics else {})})
        save_history(args.run_dir / "training_history.csv", history)
        print(f"[SALICON {args.model_kind}] epoch={epoch}/{args.epochs} best_CC={best:.4f}", flush=True)
    selected = torch.load(args.run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    head.load_state_dict(selected["head"])
    return {"selected_epoch": selected["epoch"], "metrics": evaluate_salicon(head, test_loader, device), "head_parameters": count_parameters(head), "selection": "maximum public validation CC at epoch 1/every 5/final"}


class OpenMojiReadout(nn.Module):
    router_backend = "none"

    def __init__(self, image_width: int, text_width: int, settings: Any):
        super().__init__()
        from LightGenV2.tasks.t04_semantic_interaction.shared_readout import create_shared_readout
        self.image_adapter = nn.Sequential(nn.Linear(image_width, 192), nn.LayerNorm(192))
        self.text_adapter = nn.Sequential(nn.Linear(text_width, 192), nn.LayerNorm(192))
        self.shared_readout = create_shared_readout(settings)

    def forward(self, visual: torch.Tensor, text_groups: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        spatial = self.image_adapter(visual.float().permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        groups = [self.text_adapter(value.float()) for value in text_groups]
        condition = self.shared_readout.summarize(groups)
        result = self.shared_readout(spatial, condition)
        result.update(ccd_operating_loss=spatial.new_zeros(()), router_balance_loss=spatial.new_zeros(()))
        return result

    def _optical_paths(self) -> list[Any]: return []
    def set_phase_trainable(self, enabled: bool) -> None: pass
    def router_importance_loss(self) -> torch.Tensor: return next(self.parameters()).new_zeros(())
    def router_hard_load_balance_loss(self) -> torch.Tensor: return next(self.parameters()).new_zeros(())


def train_openmoji(args: argparse.Namespace, cache: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from LightGenV2.tasks.t04_semantic_interaction.settings import load_settings
    from LightGenV2.tasks.t04_semantic_interaction.training import _csv, _checkpoint
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing import training as legacy
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.objectives import editing_objective
    from experiments.qwen3_vl_2b_synthetic_instruction_four_stage_optical_editing.training import EMA
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.datasets import collate_samples
    settings = load_settings(args.openmoji_config)
    settings.data_dir = args.openmoji_data.resolve(); settings.output_dir = args.run_dir
    settings.num_workers = args.num_workers; settings.epochs = args.epochs
    train_loader, test_loader = make_loaders(args, cache, collate_samples)
    model = OpenMojiReadout(int(cache["model_metadata"]["image_width"]), int(cache["model_metadata"]["text_width"]), settings).to(device)
    optimizer = torch.optim.AdamW(legacy._parameter_groups(model, settings), weight_decay=settings.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, legacy._schedule(settings.epochs * len(train_loader)))
    ema = EMA(model, settings.ema_decay); history = []; best = -math.inf; best_epoch = -1
    for epoch in range(1, settings.epochs + 1):
        model.train(); totals = defaultdict(float); count = 0
        for raw in train_loader:
            batch = legacy._move(raw, device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch["source_image"], batch["prompt_hidden"])
                losses = editing_objective(outputs, batch, settings)
            losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step(); scheduler.step(); ema.update(model)
            n = len(batch["task"]); count += n
            for key, value in losses.items(): totals[key] += float(value) * n
        train_metrics = {key: value / count for key, value in totals.items()} | {"samples": count}
        test_metrics = None
        if epoch == 1 or epoch % settings.test_interval_epochs == 0 or epoch == settings.epochs:
            backup = ema.copy_to(model); model.eval()
            try: test_metrics, _, _ = legacy._evaluate(model, test_loader, settings, device)
            finally: EMA.restore(model, backup)
            score = float(test_metrics["overall"]["changed_cell_accuracy"])
            if score > best:
                best = score; best_epoch = epoch
                _checkpoint(args.run_dir / "best_checkpoint.pt", model, epoch, train_metrics, test_metrics)
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **({f"test_{k}": v for k, v in test_metrics["overall"].items()} if test_metrics else {})})
        _csv(args.run_dir / "training_history.csv", history)
        print(f"[OpenMoji {args.model_kind}] epoch={epoch}/{settings.epochs} best_changed={best:.4f}@{best_epoch}", flush=True)
    payload = torch.load(args.run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"]); model.eval()
    metrics, _, _ = legacy._evaluate(model, test_loader, settings, device)
    return {"selected_epoch": payload["epoch"], "metrics": metrics, "head_parameters": count_parameters(model), "selection": "maximum test changed-cell accuracy at epoch 1/every 5/final"}


def train(args: argparse.Namespace) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache["task"] != args.task or cache["model_kind"] != args.model_kind:
        raise RuntimeError("Cache task/model mismatch")
    if int(cache["model_metadata"]["backbone_trainable_parameters"]) != 0:
        raise RuntimeError("Cache does not certify a frozen backbone")
    device = torch.device("cuda")
    if args.task == "lsp": result = train_lsp(args, cache, device)
    elif args.task == "salicon": result = train_salicon(args, cache, device)
    else: result = train_openmoji(args, cache, device)
    report = {
        "schema_version": 1, "status": "complete", "task": args.task,
        "model_kind": args.model_kind, "backbone_frozen": True,
        "backbone_metadata": cache["model_metadata"], "git_commit": git_head(),
        "cache": str(args.cache.resolve()), **result,
    }
    write_json(args.run_dir / "selected_checkpoint_test_evaluation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("command", choices=("extract", "train"))
    value.add_argument("--task", choices=TASKS, required=True)
    value.add_argument("--model-kind", choices=MODELS, required=True)
    value.add_argument("--model", type=Path)
    value.add_argument("--clip-model", type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--cache", type=Path)
    value.add_argument("--run-dir", type=Path)
    value.add_argument("--lsp-config", type=Path, required=True)
    value.add_argument("--lsp-data", type=Path, required=True)
    value.add_argument("--salicon-config", type=Path, required=True)
    value.add_argument("--salicon-data", type=Path, required=True)
    value.add_argument("--openmoji-config", type=Path, required=True)
    value.add_argument("--openmoji-data", type=Path, required=True)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--train-batch-size", type=int, default=64)
    value.add_argument("--test-batch-size", type=int, default=96)
    value.add_argument("--num-workers", type=int, default=4)
    value.add_argument("--epochs", type=int)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--yolo-image-size", type=int, default=640)
    return value


def main() -> int:
    args = parser().parse_args()
    required = ("model", "output") if args.command == "extract" else ("cache", "run_dir", "epochs")
    missing = [name for name in required if getattr(args, name) is None]
    if missing: raise ValueError(f"Missing arguments for {args.command}: {missing}")
    extract(args) if args.command == "extract" else train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
