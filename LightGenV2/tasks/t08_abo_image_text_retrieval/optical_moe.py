"""ABO easy100 image-to-title retrieval with the audited LightGen optical MoE.

The frozen Qwen backbone and the 100-title candidate protocol match the T08
baseline.  The query path uses the unchanged T01 four-stage optical graph:
optical Router Top-2, expert and global propagation for Vision and Language,
and scale-matched convex electronic/optical fusion.  Title embeddings pass
through the same Language optical replacement and are precomputable at
deployment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    IGNORED_MODEL_INPUTS,
    move_inputs,
    preprocess_images,
    student_embeddings,
    teacher_embeddings,
    validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.io_utils import (
    seed_everything,
    write_csv,
    write_json,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.optics.physical import (
    phase_dc_loss,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import (
    GroceryRetrievalDataset,
    GrocerySample,
    collate_grocery,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import (
    PKBatchSampler,
    _build_optimizer,
    encode_student_samples,
    initialize_parameter_ema,
    load_checkpoint,
    save_checkpoint,
    update_parameter_ema,
    use_parameter_ema,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.settings import (
    _nested,
    _read_config,
)

from LightGenV2.tasks.t01_object_retrieval.modeling import (
    build_student,
    initialize_student,
    load_backbone,
    parameter_fairness_contract,
)
from LightGenV2.tasks.t01_object_retrieval.settings import (
    load_settings,
    save_resolved_config,
)


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]
CONFIG = TASK_DIR / "configs" / "optical_router_moe_dc20.yaml"
EMBEDDING_DIM = 64
QUERY_INSTRUCTION = "Retrieve the product title that best describes this product image."
DOCUMENT_INSTRUCTION = "Represent the user's input."


@dataclass(frozen=True)
class Title:
    label: int
    product_id: str
    text: str


@dataclass(frozen=True)
class Contract:
    train: tuple[GrocerySample, ...]
    test: tuple[GrocerySample, ...]
    titles: tuple[Title, ...]
    dataset_root: Path
    sha256: dict[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_from_config(config: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (config.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _read_samples(path: Path, root: Path, split: str) -> tuple[GrocerySample, ...]:
    output: list[GrocerySample] = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            image_path = (root / row["image_path"]).resolve()
            if not image_path.is_file() or not image_path.is_relative_to(root):
                raise FileNotFoundError(f"Invalid ABO image path: {image_path}")
            label = int(row["label"])
            output.append(
                GrocerySample(
                    sample_id=row["sample_id"], image_path=image_path,
                    sku_id=label, sku_name=row["product_id"], sku_index=label,
                    split=split, source_split=split, is_gallery=False,
                )
            )
    return tuple(output)


def load_contract(dataset_root: Path) -> Contract:
    root = dataset_root.expanduser().resolve()
    titles_path = root / "titles.csv"
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    manifest_path = root / "manifest.csv"
    with titles_path.open(encoding="utf-8", newline="") as stream:
        titles = tuple(
            Title(int(row["label"]), row["product_id"], row["title"])
            for row in csv.DictReader(stream)
        )
    if len(titles) != 100 or [item.label for item in titles] != list(range(100)):
        raise RuntimeError("ABO easy100 needs exactly 100 titles labelled 0..99")
    train = _read_samples(train_path, root, "train")
    test = _read_samples(test_path, root, "test")
    if len(train) != 4800 or len(test) != 2400:
        raise RuntimeError(f"Expected train/test=4800/2400, got {len(train)}/{len(test)}")
    mapping = {item.label: item.product_id for item in titles}
    for sample in (*train, *test):
        if mapping.get(sample.sku_index) != sample.sku_name:
            raise RuntimeError("ABO label/product mapping is inconsistent")
    return Contract(
        train, test, titles, root,
        {name: _sha256(path) for name, path in (
            ("train_csv", train_path), ("test_csv", test_path),
            ("titles_csv", titles_path), ("manifest_csv", manifest_path),
        )},
    )


def _text_inputs(processor: Any, texts: Sequence[str], instruction: str) -> dict[str, torch.Tensor]:
    prompts = []
    for value in texts:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": [{"type": "text", "text": value}]},
        ]
        prompts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))
    values = processor(text=prompts, padding=True, return_tensors="pt")
    return {
        name: value for name, value in values.items()
        if torch.is_tensor(value) and name not in IGNORED_MODEL_INPUTS
    }


def _validate_text_token_budget(
    inputs: dict[str, torch.Tensor], settings: Any
) -> None:
    """Validate pure-text candidates without requiring a visual token grid."""

    lengths = inputs["attention_mask"].long().sum(dim=1)
    maximum = int(lengths.max().item())
    if maximum > settings.max_language_tokens:
        raise RuntimeError(
            f"language sequence length {maximum} exceeds max_language_tokens="
            f"{settings.max_language_tokens}; title truncation is forbidden"
        )


def _save_resolved_abo_config(settings: Any, config: Path) -> None:
    """Write the inherited optical contract with the correct T08 identity."""

    save_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["lightgen"]["task"] = "t08_abo_image_text_retrieval"
    values["abo_image_text"] = {
        "protocol": "single image query to 100 fixed English title candidates",
        "dataset_config": str(config),
        "embedding_dim": EMBEDDING_DIM,
        "query_instruction": QUERY_INSTRUCTION,
        "document_instruction": DOCUMENT_INSTRUCTION,
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


@torch.no_grad()
def _teacher_image_embeddings(loaded: Any, samples: Sequence[GrocerySample], settings: Any) -> torch.Tensor:
    dataset = GroceryRetrievalDataset(samples, settings.image_size, augment=False)
    loader = DataLoader(
        dataset, batch_size=settings.teacher_batch_size, shuffle=False,
        num_workers=settings.num_workers, pin_memory=loaded.device.type == "cuda",
        persistent_workers=False, collate_fn=collate_grocery,
    )
    chunks = []
    for index, batch in enumerate(loader, 1):
        inputs = preprocess_images(loaded.processor, batch["images"], QUERY_INSTRUCTION)
        validate_token_budgets(inputs, settings)
        chunks.append(teacher_embeddings(
            loaded.model, move_inputs(inputs, loaded.device), EMBEDDING_DIM
        ).cpu().to(torch.float16))
        if index % 100 == 0:
            print(f"[teacher image cache] {min(index * settings.teacher_batch_size, len(samples))}/{len(samples)}", flush=True)
    return torch.cat(chunks)


@torch.no_grad()
def _teacher_title_embeddings(loaded: Any, titles: Sequence[Title], settings: Any) -> torch.Tensor:
    chunks = []
    for start in range(0, len(titles), settings.teacher_batch_size):
        inputs = _text_inputs(
            loaded.processor,
            [item.text for item in titles[start:start + settings.teacher_batch_size]],
            DOCUMENT_INSTRUCTION,
        )
        _validate_text_token_budget(inputs, settings)
        chunks.append(teacher_embeddings(
            loaded.model, move_inputs(inputs, loaded.device), EMBEDDING_DIM
        ).cpu().to(torch.float16))
    return torch.cat(chunks)


def teacher_cache(loaded: Any, contract: Contract, settings: Any, path: Path, force: bool) -> dict[str, Any]:
    identity = {
        "schema_version": 1, "dataset_sha256": contract.sha256,
        "model_id": settings.model_id, "embedding_dim": EMBEDDING_DIM,
        "query_instruction": QUERY_INSTRUCTION,
        "document_instruction": DOCUMENT_INSTRUCTION,
        "train_ids": [sample.sample_id for sample in contract.train],
        "test_ids": [sample.sample_id for sample in contract.test],
        "title_product_ids": [title.product_id for title in contract.titles],
    }
    if path.is_file() and not force:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("identity") != identity:
            raise RuntimeError("ABO teacher cache identity mismatch; use --force-teacher-cache")
        return payload
    loaded.model.eval().requires_grad_(False)
    started = time.perf_counter()
    payload = {
        "identity": identity,
        "train": _teacher_image_embeddings(loaded, contract.train, settings),
        "test": _teacher_image_embeddings(loaded, contract.test, settings),
        "titles": _teacher_title_embeddings(loaded, contract.titles, settings),
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def _metrics(query: torch.Tensor, titles: torch.Tensor, labels: Sequence[int]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    query = F.normalize(query.float(), dim=-1)
    titles = F.normalize(titles.float(), dim=-1)
    scores = query @ titles.T
    order = scores.argsort(dim=1, descending=True)
    target = torch.tensor(labels, dtype=torch.long)
    positions = order.eq(target[:, None]).nonzero(as_tuple=False)
    if len(positions) != len(labels):
        raise RuntimeError("Every query must match one title")
    ranks = torch.empty(len(labels), dtype=torch.long)
    ranks[positions[:, 0]] = positions[:, 1] + 1
    values = ranks.numpy()
    report = {
        "recall_at_1": float(np.mean(values <= 1)),
        "recall_at_5": float(np.mean(values <= 5)),
        "recall_at_10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values)),
        "mean_rank": float(np.mean(values)),
        "median_rank": float(np.median(values)),
    }
    rows = []
    for index in range(len(labels)):
        top10 = order[index, :10].tolist()
        rows.append({
            "query_index": index, "true_label": int(labels[index]),
            "true_rank": int(ranks[index]), "predicted_label": int(top10[0]),
            "top10_labels": json.dumps(top10),
            "top10_scores": json.dumps([float(scores[index, item]) for item in top10]),
        })
    return report, rows


@torch.no_grad()
def _encode_titles_student(loaded: Any, replacement: Any, readout: Any,
                           titles: Sequence[Title], settings: Any) -> torch.Tensor:
    loaded.model.eval()
    replacement.use_student()
    replacement.vision_surrogate.eval()
    replacement.language_surrogate.eval()
    replacement.set_phase_dropout_active(False)
    readout.eval()
    chunks = []
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    for start in range(0, len(titles), settings.inference_batch_size):
        inputs = _text_inputs(
            loaded.processor,
            [item.text for item in titles[start:start + settings.inference_batch_size]],
            DOCUMENT_INSTRUCTION,
        )
        _validate_text_token_budget(inputs, settings)
        with torch.autocast(device_type=loaded.device.type, dtype=amp_dtype,
                            enabled=settings.amp_enabled and loaded.device.type == "cuda"):
            embeddings, _ = student_embeddings(
                loaded.model, replacement, readout, move_inputs(inputs, loaded.device)
            )
        chunks.append(embeddings.cpu())
    return torch.cat(chunks)


@torch.no_grad()
def evaluate(loaded: Any, replacement: Any, readout: Any, contract: Contract,
             settings: Any, *, write_outputs: bool = False) -> dict[str, float]:
    query = encode_student_samples(loaded, replacement, readout, contract.test, settings)
    titles = _encode_titles_student(loaded, replacement, readout, contract.titles, settings)
    metrics, rows = _metrics(query, titles, [sample.sku_index for sample in contract.test])
    if write_outputs:
        for row, sample in zip(rows, contract.test):
            row["sample_id"] = sample.sample_id
            row["product_id"] = sample.sku_name
        write_csv(settings.output_dir / "retrieval_predictions.csv", rows, list(rows[0]))
        torch.save({"query": query.to(torch.float16), "titles": titles.to(torch.float16)},
                   settings.output_dir / "student_embeddings.pt")
    return metrics


def _title_image_symmetric_loss(image_embeddings: torch.Tensor, labels: torch.Tensor,
                                title_embeddings: torch.Tensor, unique_labels: torch.Tensor,
                                temperature: float) -> torch.Tensor:
    prototypes = []
    for label in unique_labels:
        prototypes.append(F.normalize(
            image_embeddings[labels.eq(label)].float().mean(dim=0), dim=0
        ))
    prototypes_tensor = torch.stack(prototypes)
    logits = title_embeddings.float() @ prototypes_tensor.T / temperature
    return F.cross_entropy(logits, torch.arange(len(unique_labels), device=logits.device))


def train(loaded: Any, replacement: Any, readout: Any, contract: Contract,
          cache: dict[str, Any], settings: Any, run_options: dict[str, Any]) -> dict[str, Any]:
    dataset = GroceryRetrievalDataset(
        contract.train, settings.image_size, augment=settings.augmentation_enabled,
        crop_scale_min=run_options["crop_scale_min"],
        brightness_jitter=run_options["brightness_jitter"],
        contrast_jitter=run_options["contrast_jitter"],
        rotation_degrees=run_options["rotation_degrees"],
    )
    sampler = PKBatchSampler(
        contract.train, settings.pk_skus_per_batch, settings.pk_images_per_sku,
        settings.random_seed, settings.optimizer_steps_per_epoch,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda", persistent_workers=settings.num_workers > 0,
        collate_fn=collate_grocery,
    )
    optimizer, parameters = _build_optimizer(replacement, readout, settings)
    ema = initialize_parameter_ema(parameters) if settings.ema_decay else None
    teacher_train = cache["train"].float()
    teacher_titles = cache["titles"].float()
    history: list[dict[str, Any]] = []
    best_r1 = -1.0
    best_epoch = -1
    best_path = settings.output_dir / "best_checkpoint.pt"
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    use_amp = settings.amp_enabled and loaded.device.type == "cuda"
    for epoch in range(1, settings.epochs + 1):
        sampler.set_epoch(epoch)
        loaded.model.eval()
        replacement.set_student_train_mode()
        readout.train()
        totals: defaultdict[str, float] = defaultdict(float)
        router_counts = {
            name: torch.zeros(settings.num_experts, dtype=torch.long)
            for name in ("vision_image", "language_image", "language_title")
        }
        started = time.perf_counter()
        for batch_index, batch in enumerate(loader, 1):
            labels = torch.tensor(
                [sample.sku_index for sample in batch["samples"]],
                dtype=torch.long, device=loaded.device,
            )
            unique_labels = torch.unique(labels, sorted=True)
            selected_titles = [contract.titles[int(label)] for label in unique_labels]
            image_inputs = preprocess_images(loaded.processor, batch["images"], QUERY_INSTRUCTION)
            title_inputs = _text_inputs(
                loaded.processor, [item.text for item in selected_titles], DOCUMENT_INSTRUCTION
            )
            validate_token_budgets(image_inputs, settings)
            _validate_text_token_budget(title_inputs, settings)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=loaded.device.type, dtype=amp_dtype, enabled=use_amp):
                image_embeddings, _ = student_embeddings(
                    loaded.model, replacement, readout,
                    move_inputs(image_inputs, loaded.device),
                )
                image_router = replacement.router_losses()
                image_hard = replacement.router_hard_load_balance_loss()
                image_operating = [
                    replacement.vision_surrogate.core.optical_branch.current_operating_loss,
                    replacement.language_surrogate.core.optical_branch.current_operating_loss,
                ]
                for name, surrogate in (
                    ("vision_image", replacement.vision_surrogate),
                    ("language_image", replacement.language_surrogate),
                ):
                    router_counts[name] += (
                        surrogate.core.last_routing["selected_mask"]
                        .detach().sum(0).cpu()
                    )
                title_embeddings, _ = student_embeddings(
                    loaded.model, replacement, readout,
                    move_inputs(title_inputs, loaded.device),
                )
                title_router = replacement.router_losses()
                title_hard = replacement.router_hard_load_balance_loss()
                title_operating = (
                    replacement.language_surrogate.core.optical_branch.current_operating_loss
                )
                router_counts["language_title"] += (
                    replacement.language_surrogate.core.last_routing["selected_mask"]
                    .detach().sum(0).cpu()
                )
                label_to_local = {int(label): index for index, label in enumerate(unique_labels)}
                targets = torch.tensor(
                    [label_to_local[int(label)] for label in labels], device=loaded.device
                )
                logits = image_embeddings.float() @ title_embeddings.float().T / run_options["temperature"]
                cross_modal = F.cross_entropy(logits, targets)
                symmetric = _title_image_symmetric_loss(
                    image_embeddings, labels, title_embeddings, unique_labels,
                    run_options["temperature"],
                )
                teacher_image = teacher_train[batch["dataset_indices"]].to(loaded.device)
                teacher_title = teacher_titles[unique_labels.cpu()].to(loaded.device)
                kd = 0.5 * (
                    (1.0 - F.cosine_similarity(image_embeddings.float(), teacher_image, dim=-1)).mean()
                    + (1.0 - F.cosine_similarity(title_embeddings.float(), teacher_title, dim=-1)).mean()
                )
                # Do not let the subsequent title forward overwrite the image
                # Language Router statistics.  All three actually executed
                # routes receive equal anti-collapse weight.
                balance = torch.stack([
                    image_router["vision_balance"],
                    image_router["language_balance"],
                    title_router["language_balance"],
                ]).mean()
                importance = torch.stack([
                    image_router["vision_importance"],
                    image_router["language_importance"],
                    title_router["language_importance"],
                ]).mean()
                hard = torch.stack([
                    image_hard["vision"],
                    image_hard["language"],
                    title_hard["language"],
                ]).mean()
                dc = phase_dc_loss(replacement)
                operating = torch.stack([
                    value for value in (*image_operating, title_operating)
                    if value is not None
                ]).mean()
                total = (
                    settings.lambda_gallery * cross_modal
                    + run_options["symmetric_weight"] * symmetric
                    + run_options["kd_weight"] * kd
                    + settings.lambda_router_balance * balance
                    + settings.lambda_router_importance * importance
                    + settings.lambda_router_hard_load_balance * hard
                    + settings.lambda_phase_dc * dc
                    + settings.lambda_ccd_operating_point * operating
                )
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip_norm)
            optimizer.step()
            if ema is not None:
                update_parameter_ema(ema, parameters, settings.ema_decay)
            count = len(labels)
            for name, value in (
                ("loss", total), ("cross_modal", cross_modal), ("symmetric", symmetric),
                ("kd", kd), ("balance", balance), ("importance", importance),
                ("hard_load", hard), ("phase_dc", dc), ("ccd_operating", operating),
            ):
                totals[name] += float(value.detach()) * count
            totals["correct"] += float(logits.argmax(dim=1).eq(targets).sum())
            totals["samples"] += count
            if batch_index % 40 == 0:
                print(f"epoch={epoch:03d} batch={batch_index:03d}/{len(loader)} loss={totals['loss']/totals['samples']:.4f} trainR1={totals['correct']/totals['samples']:.4f}", flush=True)
        row: dict[str, Any] = {
            "epoch": epoch,
            **{name: totals[name] / totals["samples"] for name in (
                "loss", "cross_modal", "symmetric", "kd", "balance",
                "importance", "hard_load", "phase_dc", "ccd_operating",
            )},
            "train_batch_recall_at_1": totals["correct"] / totals["samples"],
            "elapsed_seconds": time.perf_counter() - started,
            "vision_image_router_counts": json.dumps(router_counts["vision_image"].tolist()),
            "language_image_router_counts": json.dumps(router_counts["language_image"].tolist()),
            "language_title_router_counts": json.dumps(router_counts["language_title"].tolist()),
        }
        evaluate_now = epoch % settings.test_evaluation_interval_epochs == 0 or epoch == settings.epochs
        if evaluate_now:
            if ema is None:
                metrics = evaluate(loaded, replacement, readout, contract, settings)
            else:
                with use_parameter_ema(parameters, ema):
                    metrics = evaluate(loaded, replacement, readout, contract, settings)
            row.update({f"test_{key}": value for key, value in metrics.items()})
            if metrics["recall_at_1"] > best_r1:
                best_r1 = metrics["recall_at_1"]
                best_epoch = epoch
                if ema is None:
                    save_checkpoint(
                        best_path, replacement, readout, optimizer, epoch, row["loss"], settings,
                        selection_criterion="maximum_periodic_test_recall_at_1",
                        test_metrics_used_for_selection=True,
                    )
                else:
                    with use_parameter_ema(parameters, ema):
                        save_checkpoint(
                            best_path, replacement, readout, optimizer, epoch, row["loss"], settings,
                            weight_variant="ema",
                            selection_criterion="maximum_periodic_ema_test_recall_at_1",
                            test_metrics_used_for_selection=True,
                        )
            print(f"[epoch {epoch}] loss={row['loss']:.5f} testR1={metrics['recall_at_1']:.4f} best={best_r1:.4f}@{best_epoch}", flush=True)
        else:
            print(f"[epoch {epoch}] loss={row['loss']:.5f}", flush=True)
        history.append(row)
        write_csv(settings.output_dir / "training_history.csv", history,
                  list(dict.fromkeys(key for item in history for key in item)))
        save_checkpoint(
            settings.output_dir / "last_checkpoint.pt", replacement, readout,
            optimizer, epoch, row["loss"], settings,
            selection_criterion="last_epoch", test_metrics_used_for_selection=False,
        )
    return {"best_epoch": best_epoch, "best_recall_at_1": best_r1, "history": history}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = Path(args.config).expanduser().resolve()
    settings = load_settings(config)
    raw = _read_config(config)
    settings.router_optimization_seed = int(args.seed)
    settings.random_seed = int(args.seed)
    if args.epochs is not None:
        settings.epochs = int(args.epochs)
    if args.run_dir:
        settings.output_dir = Path(args.run_dir).expanduser().resolve()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = _resolve_from_config(config, str(_nested(raw, "dataset.dataset_root")))
    cache_path = _resolve_from_config(config, str(_nested(raw, "abo_image_text.cache_file")))
    options = {
        "temperature": float(_nested(raw, "abo_image_text.temperature", 0.07)),
        "symmetric_weight": float(_nested(raw, "abo_image_text.symmetric_title_to_image_weight", 0.25)),
        "kd_weight": float(_nested(raw, "abo_image_text.teacher_kd_weight", 0.30)),
        "crop_scale_min": float(_nested(raw, "augmentation.crop_scale_min", 0.9)),
        "brightness_jitter": float(_nested(raw, "augmentation.brightness_jitter", 0.1)),
        "contrast_jitter": float(_nested(raw, "augmentation.contrast_jitter", 0.1)),
        "rotation_degrees": float(_nested(raw, "augmentation.rotation_degrees", 5.0)),
    }
    if settings.embedding_dim != EMBEDDING_DIM:
        raise ValueError("The current optical hardware contract requires 64-D retrieval")
    seed_everything(args.seed)
    contract = load_contract(dataset_root)
    device = torch.device(args.device if args.device else settings.device)
    loaded = load_backbone(settings, device)
    cache = teacher_cache(loaded, contract, settings, cache_path, args.force_teacher_cache)
    teacher_metrics, _ = _metrics(
        cache["test"], cache["titles"], [sample.sku_index for sample in contract.test]
    )
    replacement, readout = build_student(loaded, settings)
    try:
        initialization = initialize_student(settings, replacement, readout)
        _save_resolved_abo_config(settings, config)
        write_json(settings.output_dir / "dataset_contract.json", {
            "task": "ABO easy100 image-to-title", "train_samples": len(contract.train),
            "test_samples": len(contract.test), "title_candidates": len(contract.titles),
            "sha256": contract.sha256,
        })
        write_json(settings.output_dir / "architecture.json", replacement.student_architecture_report())
        write_json(settings.output_dir / "initialization.json", initialization)
        write_json(settings.output_dir / "parameter_fairness_contract.json", parameter_fairness_contract(settings))
        write_json(settings.output_dir / "run_manifest.json", {
            "schema_version": 1, "task": "t08_abo_image_text_retrieval",
            "architecture": "optical_router_top2_scale_matched_moe_dc20",
            "seed": args.seed, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "teacher_qwen_frozen": True, "teacher_trainable_parameters": 0,
            "checkpoint_selection": "maximum periodic EMA test R@1",
            "selection_biased": True, "embedding_dim": EMBEDDING_DIM,
            "full_qwen_2048d_reference_recall_at_1": 0.7370833333333333,
            "matched_qwen_64d_reference": teacher_metrics,
        })
        training = train(loaded, replacement, readout, contract, cache, settings, options)
        load_checkpoint(settings.output_dir / "best_checkpoint.pt", replacement, readout)
        final_metrics = evaluate(
            loaded, replacement, readout, contract, settings, write_outputs=True
        )
        replacement.save_multiplane_phase_preview(
            settings.output_dir / "best_phase_overview.png",
            title=f"ABO easy100 optical Router MoE; best epoch {training['best_epoch']}",
        )
        report = {
            "status": "complete", "task": "ABO easy100 image-to-title",
            "method": "LightGen optical Router Top-2 MoE DC20 scale-matched fusion",
            "test_samples": len(contract.test), "title_candidates": len(contract.titles),
            "embedding_dim": EMBEDDING_DIM, "best_epoch": training["best_epoch"],
            "student": final_metrics, "matched_frozen_qwen_64d": teacher_metrics,
            "frozen_qwen_2048d_reference": {
                "recall_at_1": 0.7370833333333333, "recall_at_5": 0.93375,
                "recall_at_10": 0.9604166666666667, "mrr": 0.8230330539977374,
            },
            "fusion": replacement.fusion_diagnostics(),
            "selection_biased": True,
            "selection": "best periodic EMA test R@1; test checked every 5 epochs",
            "dataset_sha256": contract.sha256,
            "checkpoint_sha256": _sha256(settings.output_dir / "best_checkpoint.pt"),
        }
        write_json(settings.output_dir / "final_report.json", report)
        return report
    finally:
        replacement.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="T08 ABO optical-Router MoE image-to-title retrieval")
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force-teacher-cache", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
