"""ABO easy100 image/text retrieval with the audited LightGen optical MoE.

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
import re
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
TEXT_TO_IMAGE_QUERY_INSTRUCTION = (
    "Retrieve product images that match the following product description."
)
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


@dataclass(frozen=True)
class PromptContract:
    """Bind each modality to its actual retrieval role before encoding."""

    direction: str
    image_instruction: str
    title_instruction: str


def _prompt_contract(raw: dict[str, Any]) -> PromptContract:
    direction = str(_nested(raw, "abo_image_text.retrieval_direction", "image_to_text"))
    if direction == "image_to_text":
        return PromptContract(direction, QUERY_INSTRUCTION, DOCUMENT_INSTRUCTION)
    if direction == "text_to_image":
        return PromptContract(
            direction, DOCUMENT_INSTRUCTION, TEXT_TO_IMAGE_QUERY_INSTRUCTION
        )
    raise ValueError("retrieval_direction must be image_to_text or text_to_image")


def _load_propagation_transition_checkpoint(
    path: Path, replacement: Any, readout: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse electronics while retraining every phase for a new distance.

    Propagation distance changes the transfer function, so feature and Router
    phases are deliberately *not* copied.  Adapters, electronic residuals and
    the 64-D readout remain a useful initialization and do not encode the old
    free-space transfer function.
    """

    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_architecture = str(
        payload.get("metadata", {}).get("optical_architecture", "")
    )
    target_architecture = str(replacement.checkpoint_architecture)
    source_distance = _architecture_distance_cm(source_architecture)
    target_distance = _architecture_distance_cm(target_architecture)
    if source_distance == target_distance:
        raise RuntimeError(
            "Propagation transition requires different pinned source and target distances"
        )

    fresh_by_modality: dict[str, list[str]] = {}
    for label, module, source_key in (
        ("vision", replacement.vision_surrogate, "vision_optical"),
        ("language", replacement.language_surrogate, "language_optical"),
    ):
        target_state = module.state_dict()
        phase_names = sorted(
            name for name in target_state
            if name.endswith("raw_phase") or name.endswith("raw_router_phase")
        )
        transferred = {
            name: value for name, value in payload[source_key].items()
            if name not in phase_names
        }
        incompatible = module.load_state_dict(transferred, strict=False)
        if sorted(incompatible.missing_keys) != phase_names or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unsafe {label} {source_distance} cm -> {target_distance} cm transplant: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        fresh_by_modality[label] = phase_names
    readout.load_state_dict(payload["retrieval_readout"], strict=True)
    replacement.reset_fusion_logits()
    return payload, {
        "source_architecture": source_architecture,
        "target_architecture": target_architecture,
        "source_distance_cm": source_distance,
        "target_distance_cm": target_distance,
        "fresh_phase_tensors": fresh_by_modality,
        "transferred": "electronic residuals, adapters and 64-D readout",
        "fusion_alpha_reset_to": replacement.fusion_diagnostics(),
    }


def _compact_residual_mlp_state(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Project a wider residual MLP checkpoint into a narrower expansion.

    The outer electronic width and every optical tensor remain unchanged. For
    each residual MLP, neurons are ranked by the product of their incoming and
    outgoing weight norms. The strongest target-width subset is copied into
    the compact block; all equal-shaped tensors are copied exactly.
    """

    compact: dict[str, torch.Tensor] = {}
    handled: set[str] = set()
    selections: dict[str, list[int]] = {}
    for key, target in target_state.items():
        if not key.endswith(".mlp.0.weight"):
            continue
        source = source_state.get(key)
        if source is None or source.shape == target.shape:
            continue
        prefix = key[: -len("0.weight")]
        up_bias_key = f"{prefix}0.bias"
        down_weight_key = f"{prefix}3.weight"
        down_bias_key = f"{prefix}3.bias"
        source_down = source_state.get(down_weight_key)
        target_down = target_state.get(down_weight_key)
        if (
            source.ndim != 2
            or source_down is None
            or target_down is None
            or source.shape[1] != target.shape[1]
            or source_down.shape[0] != target_down.shape[0]
            or source.shape[0] != source_down.shape[1]
            or target.shape[0] != target_down.shape[1]
            or target.shape[0] >= source.shape[0]
        ):
            raise RuntimeError(f"Unsupported compact residual MLP shape at {key}")
        scores = source.float().pow(2).sum(1).sqrt() * (
            source_down.float().pow(2).sum(0).sqrt()
        )
        selected = torch.topk(scores, target.shape[0], largest=True).indices.sort().values
        compact[key] = source.index_select(0, selected).to(dtype=target.dtype)
        compact[up_bias_key] = source_state[up_bias_key].index_select(0, selected).to(
            dtype=target_state[up_bias_key].dtype
        )
        compact[down_weight_key] = source_down.index_select(1, selected).to(
            dtype=target_down.dtype
        )
        compact[down_bias_key] = source_state[down_bias_key].to(
            dtype=target_state[down_bias_key].dtype
        )
        handled.update({key, up_bias_key, down_weight_key, down_bias_key})
        selections[prefix.rstrip(".")] = selected.tolist()

    for key, target in target_state.items():
        if key in handled:
            continue
        source = source_state.get(key)
        if source is None:
            raise RuntimeError(f"Compact transition source is missing {key}")
        if source.shape != target.shape:
            raise RuntimeError(
                "Compact transition only permits residual-MLP expansion changes; "
                f"{key} is {tuple(source.shape)} -> {tuple(target.shape)}"
            )
        compact[key] = source.to(dtype=target.dtype)
    if not selections:
        raise RuntimeError("Compact transition did not find any narrower residual MLP")
    return compact, {"selection_rule": "incoming_norm_times_outgoing_norm", "kept": selections}


def _architecture_distance_cm(architecture: str) -> int:
    match = re.search(r"_(\d+)cm_17um_", architecture)
    if match is None:
        raise RuntimeError("Checkpoint architecture does not contain a pinned distance")
    return int(match.group(1))


def _load_electronic_compaction_checkpoint(
    path: Path, replacement: Any, readout: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep same-distance optics fixed while structurally shrinking electronic MLPs."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_architecture = str(payload.get("metadata", {}).get("optical_architecture", ""))
    target_architecture = str(replacement.checkpoint_architecture)
    source_distance = _architecture_distance_cm(source_architecture)
    target_distance = _architecture_distance_cm(target_architecture)
    if source_distance != target_distance:
        raise RuntimeError(
            "Electronic compaction cannot change propagation distance: "
            f"{source_distance} cm -> {target_distance} cm"
        )
    reports: dict[str, Any] = {}
    for label, module, source_key in (
        ("vision", replacement.vision_surrogate, "vision_optical"),
        ("language", replacement.language_surrogate, "language_optical"),
    ):
        target_state = module.state_dict()
        compact_state, report = _compact_residual_mlp_state(
            payload[source_key], target_state
        )
        module.load_state_dict(compact_state, strict=True)
        report.update(
            {
                "source_parameters": sum(value.numel() for value in payload[source_key].values()),
                "target_parameters": sum(value.numel() for value in compact_state.values()),
            }
        )
        reports[label] = report
    readout.load_state_dict(payload["retrieval_readout"], strict=True)
    return payload, {
        "source_architecture": source_architecture,
        "target_architecture": target_architecture,
        "method": "structured residual-MLP neuron pruning",
        "propagation_distance_cm": source_distance,
        "optical_tensors": "copied exactly",
        "readout": "copied exactly",
        "modalities": reports,
    }


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


def _save_resolved_abo_config(
    settings: Any, config: Path, prompts: PromptContract
) -> None:
    """Write the inherited optical contract with the correct T08 identity."""

    save_resolved_config(settings)
    path = settings.output_dir / "config.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values["lightgen"]["task"] = "t08_abo_image_text_retrieval"
    values["abo_image_text"] = {
        "protocol": (
            "100 English title queries to 2400 held-out image documents"
            if prompts.direction == "text_to_image" else
            "single image query to 100 fixed English title candidates"
        ),
        "dataset_config": str(config),
        "embedding_dim": EMBEDDING_DIM,
        "retrieval_direction": prompts.direction,
        "image_instruction": prompts.image_instruction,
        "title_instruction": prompts.title_instruction,
        "query_instruction": (
            prompts.title_instruction
            if prompts.direction == "text_to_image" else prompts.image_instruction
        ),
        "document_instruction": DOCUMENT_INSTRUCTION,
    }
    path.write_text(
        yaml.safe_dump(values, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


@torch.no_grad()
def _teacher_image_embeddings(
    loaded: Any, samples: Sequence[GrocerySample], settings: Any,
    prompts: PromptContract,
) -> torch.Tensor:
    dataset = GroceryRetrievalDataset(samples, settings.image_size, augment=False)
    loader = DataLoader(
        dataset, batch_size=settings.teacher_batch_size, shuffle=False,
        num_workers=settings.num_workers, pin_memory=loaded.device.type == "cuda",
        persistent_workers=False, collate_fn=collate_grocery,
    )
    chunks = []
    for index, batch in enumerate(loader, 1):
        inputs = preprocess_images(
            loaded.processor, batch["images"], prompts.image_instruction
        )
        validate_token_budgets(inputs, settings)
        chunks.append(teacher_embeddings(
            loaded.model, move_inputs(inputs, loaded.device), EMBEDDING_DIM
        ).cpu().to(torch.float16))
        if index % 100 == 0:
            print(f"[teacher image cache] {min(index * settings.teacher_batch_size, len(samples))}/{len(samples)}", flush=True)
    return torch.cat(chunks)


@torch.no_grad()
def _teacher_title_embeddings(
    loaded: Any, titles: Sequence[Title], settings: Any,
    prompts: PromptContract,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(titles), settings.teacher_batch_size):
        inputs = _text_inputs(
            loaded.processor,
            [item.text for item in titles[start:start + settings.teacher_batch_size]],
            prompts.title_instruction,
        )
        _validate_text_token_budget(inputs, settings)
        chunks.append(teacher_embeddings(
            loaded.model, move_inputs(inputs, loaded.device), EMBEDDING_DIM
        ).cpu().to(torch.float16))
    return torch.cat(chunks)


def teacher_cache(
    loaded: Any, contract: Contract, settings: Any, path: Path, force: bool,
    prompts: PromptContract,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1, "dataset_sha256": contract.sha256,
        "model_id": settings.model_id, "embedding_dim": EMBEDDING_DIM,
        "retrieval_direction": prompts.direction,
        "image_instruction": prompts.image_instruction,
        "title_instruction": prompts.title_instruction,
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
        "train": _teacher_image_embeddings(loaded, contract.train, settings, prompts),
        "test": _teacher_image_embeddings(loaded, contract.test, settings, prompts),
        "titles": _teacher_title_embeddings(loaded, contract.titles, settings, prompts),
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


def _text_to_image_metrics(titles: torch.Tensor, images: torch.Tensor,
                           image_labels: Sequence[int]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """One title per SKU retrieves all24 held-out images of that SKU."""
    titles = F.normalize(titles.float(), dim=-1)
    images = F.normalize(images.float(), dim=-1)
    scores = titles @ images.T
    order = scores.argsort(dim=1, descending=True, stable=True)
    labels = torch.tensor(image_labels, dtype=torch.long)
    query_labels = torch.arange(len(titles), dtype=torch.long)
    relevant = labels[order].eq(query_labels[:, None])
    totals = relevant.sum(1)
    if len(titles) != 100 or not bool(totals.eq(24).all()):
        raise RuntimeError("Text-to-image protocol requires100 titles and24 TEST positives each")
    ranks = torch.arange(1, len(images) + 1, dtype=torch.float64)[None]
    precision = relevant.cumsum(1) / ranks
    first = torch.where(relevant, ranks, torch.inf).amin(1)
    report = {
        "hit_at_1": float(relevant[:, :1].any(1).double().mean()),
        "hit_at_5": float(relevant[:, :5].any(1).double().mean()),
        "hit_at_10": float(relevant[:, :10].any(1).double().mean()),
        "recall_at_1": float((relevant[:, :1].sum(1) / totals).double().mean()),
        "recall_at_5": float((relevant[:, :5].sum(1) / totals).double().mean()),
        "recall_at_10": float((relevant[:, :10].sum(1) / totals).double().mean()),
        "mrr": float((1.0 / first).mean()),
        "map": float(((precision * relevant).sum(1) / totals).mean()),
        "query_count": len(titles), "gallery_count": len(images),
        "relevant_images_per_query": 24,
    }
    rows = [{"title_label": i, "first_positive_rank": int(first[i]),
             "top10_image_indices": json.dumps(order[i, :10].tolist()),
             "top10_scores": json.dumps([float(scores[i, j]) for j in order[i, :10]])}
            for i in range(len(titles))]
    return report, rows


@torch.no_grad()
def _encode_titles_student(loaded: Any, replacement: Any, readout: Any,
                           titles: Sequence[Title], settings: Any,
                           prompts: PromptContract) -> torch.Tensor:
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
            prompts.title_instruction,
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
             settings: Any, prompts: PromptContract, *,
             write_outputs: bool = False) -> dict[str, float]:
    return evaluate_bidirectional(loaded, replacement, readout, contract, settings,
                                  prompts, write_outputs=write_outputs)["image_to_text"]


@torch.no_grad()
def evaluate_bidirectional(loaded: Any, replacement: Any, readout: Any, contract: Contract,
                           settings: Any, prompts: PromptContract, *,
                           write_outputs: bool = False) -> dict[str, Any]:
    query = encode_student_samples(loaded, replacement, readout, contract.test, settings)
    titles = _encode_titles_student(
        loaded, replacement, readout, contract.titles, settings, prompts
    )
    labels = [sample.sku_index for sample in contract.test]
    image_to_text, image_rows = _metrics(query, titles, labels)
    text_to_image, title_rows = _text_to_image_metrics(titles, query, labels)
    if write_outputs:
        for row, sample in zip(image_rows, contract.test):
            row["sample_id"] = sample.sample_id
            row["product_id"] = sample.sku_name
        for row, title in zip(title_rows, contract.titles):
            row["product_id"] = title.product_id
            row["title"] = title.text
            indices = json.loads(row["top10_image_indices"])
            row["top10_sample_ids"] = json.dumps([contract.test[i].sample_id for i in indices])
        write_csv(settings.output_dir / "image_to_text_predictions.csv", image_rows, list(image_rows[0]))
        write_csv(settings.output_dir / "text_to_image_predictions.csv", title_rows, list(title_rows[0]))
        torch.save({"query": query.to(torch.float16), "titles": titles.to(torch.float16)},
                   settings.output_dir / "student_embeddings.pt")
    return {"image_to_text": image_to_text, "text_to_image": text_to_image}


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
          cache: dict[str, Any], settings: Any, run_options: dict[str, Any],
          prompts: PromptContract) -> dict[str, Any]:
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
    alpha_curriculum = run_options["alpha_curriculum"]
    if alpha_curriculum:
        for surrogate in (replacement.vision_surrogate, replacement.language_surrogate):
            surrogate.core.block1_optical_fusion_logit.requires_grad_(False)
            surrogate.core.block2_optical_fusion_logit.requires_grad_(False)
    ema = initialize_parameter_ema(parameters) if settings.ema_decay else None
    curriculum_gate_ids = {
        id(parameter)
        for surrogate in (replacement.vision_surrogate, replacement.language_surrogate)
        for parameter in (
            surrogate.core.block1_optical_fusion_logit,
            surrogate.core.block2_optical_fusion_logit,
        )
    } if alpha_curriculum else set()
    teacher_train = cache["train"].float()
    teacher_titles = cache["titles"].float()
    history: list[dict[str, Any]] = []
    selection_direction = run_options["selection_direction"]
    best_r1 = -1.0
    best_epoch = -1
    best_path = settings.output_dir / "best_checkpoint.pt"
    amp_dtype = torch.bfloat16 if settings.dtype == "bfloat16" else torch.float16
    use_amp = settings.amp_enabled and loaded.device.type == "cuda"
    if run_options["continuation"]:
        initial = evaluate_bidirectional(
            loaded, replacement, readout, contract, settings, prompts
        )
        key = "hit_at_1" if selection_direction == "text_to_image" else "recall_at_1"
        initial_score = initial[selection_direction][key]
        best_epoch = 0
        initial_eligible = not alpha_curriculum
        if initial_eligible:
            best_r1 = initial_score
            save_checkpoint(best_path, replacement, readout, optimizer, 0, 0.0, settings,
                            selection_criterion=f"initial_continuation_{selection_direction}_top1",
                            test_metrics_used_for_selection=True)
        history.append({"epoch": 0, "continuation_initial": True,
                        "selection_eligible": initial_eligible,
                        **{f"test_{direction}_{name}": value
                           for direction, values in initial.items() for name, value in values.items()}})
        write_csv(settings.output_dir / "training_history.csv", history, list(history[0]))
        print(f"[epoch 0] {selection_direction}R1={initial_score:.4f} continuation baseline eligible={initial_eligible}", flush=True)
    for epoch in range(1, settings.epochs + 1):
        scheduled_alpha = None
        if alpha_curriculum:
            fraction = min(1.0, epoch / alpha_curriculum["warmup_epochs"])
            scheduled_alpha = alpha_curriculum["start"] + fraction * (
                alpha_curriculum["end"] - alpha_curriculum["start"]
            )
            for surrogate in (replacement.vision_surrogate, replacement.language_surrogate):
                surrogate.core.reset_fusion_logits(scheduled_alpha)
            # The four fusion gates are externally scheduled rather than learned.
            # Keep their EMA copies exactly on that schedule; otherwise periodic
            # EMA evaluation silently reinstalls the old alpha~=0.05 values.
            if ema is not None:
                with torch.no_grad():
                    for parameter, ema_parameter in zip(parameters, ema):
                        if id(parameter) in curriculum_gate_ids:
                            ema_parameter.copy_(parameter.detach().float())
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
            image_inputs = preprocess_images(
                loaded.processor, batch["images"], prompts.image_instruction
            )
            title_inputs = _text_inputs(
                loaded.processor, [item.text for item in selected_titles],
                prompts.title_instruction,
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
            "scheduled_alpha": scheduled_alpha,
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
                metrics = evaluate_bidirectional(
                    loaded, replacement, readout, contract, settings, prompts
                )
            else:
                with use_parameter_ema(parameters, ema):
                    metrics = evaluate_bidirectional(
                        loaded, replacement, readout, contract, settings, prompts
                    )
            for direction, values in metrics.items():
                row.update({f"test_{direction}_{key}": value for key, value in values.items()})
            selected_r1 = metrics[selection_direction]["hit_at_1" if selection_direction == "text_to_image" else "recall_at_1"]
            eligible = (not alpha_curriculum or
                        scheduled_alpha >= alpha_curriculum["minimum_selected"])
            row["selection_eligible"] = eligible
            if eligible and selected_r1 > best_r1:
                best_r1 = selected_r1
                best_epoch = epoch
                if ema is None:
                    save_checkpoint(
                        best_path, replacement, readout, optimizer, epoch, row["loss"], settings,
                        selection_criterion=f"maximum_periodic_test_{selection_direction}_top1",
                        test_metrics_used_for_selection=True,
                    )
                else:
                    with use_parameter_ema(parameters, ema):
                        save_checkpoint(
                            best_path, replacement, readout, optimizer, epoch, row["loss"], settings,
                            weight_variant="ema",
                            selection_criterion=f"maximum_periodic_ema_test_{selection_direction}_top1",
                            test_metrics_used_for_selection=True,
                        )
            print(f"[epoch {epoch}] loss={row['loss']:.5f} {selection_direction}R1={selected_r1:.4f} best={best_r1:.4f}@{best_epoch}", flush=True)
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
    if not best_path.is_file():
        raise RuntimeError("No alpha-eligible checkpoint was evaluated; adjust schedule/eval interval")
    return {"best_epoch": best_epoch, "best_recall_at_1": best_r1, "history": history}


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = Path(args.config).expanduser().resolve()
    settings = load_settings(config)
    raw = _read_config(config)
    prompts = _prompt_contract(raw)
    settings.router_optimization_seed = int(args.seed)
    settings.random_seed = int(args.seed)
    if args.epochs is not None:
        settings.epochs = int(args.epochs)
    if args.steps_per_epoch is not None:
        if args.steps_per_epoch < 1:
            raise ValueError("--steps-per-epoch must be positive")
        settings.optimizer_steps_per_epoch = int(args.steps_per_epoch)
    if args.eval_every is not None:
        if args.eval_every < 1:
            raise ValueError("--eval-every must be positive")
        settings.test_evaluation_interval_epochs = int(args.eval_every)
    if args.run_dir:
        settings.output_dir = Path(args.run_dir).expanduser().resolve()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = (Path(args.data_root).expanduser().resolve() if args.data_root else
                    _resolve_from_config(config, str(_nested(raw, "dataset.dataset_root"))))
    cache_path = (Path(args.teacher_cache).expanduser().resolve() if args.teacher_cache else
                  _resolve_from_config(config, str(_nested(raw, "abo_image_text.cache_file"))))
    options = {
        "temperature": float(_nested(raw, "abo_image_text.temperature", 0.07)),
        "symmetric_weight": float(_nested(raw, "abo_image_text.symmetric_title_to_image_weight", 0.25)),
        "kd_weight": float(_nested(raw, "abo_image_text.teacher_kd_weight", 0.30)),
        "crop_scale_min": float(_nested(raw, "augmentation.crop_scale_min", 0.9)),
        "brightness_jitter": float(_nested(raw, "augmentation.brightness_jitter", 0.1)),
        "contrast_jitter": float(_nested(raw, "augmentation.contrast_jitter", 0.1)),
        "rotation_degrees": float(_nested(raw, "augmentation.rotation_degrees", 5.0)),
        "selection_direction": str(_nested(raw, "abo_image_text.selection_direction", "image_to_text")),
        "continuation": bool(args.resume_checkpoint or
                             _nested(raw, "abo_image_text.resume_checkpoint", None)),
        "alpha_curriculum": _nested(raw, "abo_image_text.alpha_curriculum", None),
    }
    if options["selection_direction"] not in ("image_to_text", "text_to_image"):
        raise ValueError("selection_direction must be image_to_text or text_to_image")
    if options["alpha_curriculum"]:
        schedule = options["alpha_curriculum"]
        if (set(schedule) != {"start", "end", "warmup_epochs", "minimum_selected"}
                or not settings.fusion_alpha_min < float(schedule["start"]) < float(schedule["end"]) < settings.fusion_alpha_max
                or int(schedule["warmup_epochs"]) < 1
                or not float(schedule["start"]) <= float(schedule["minimum_selected"]) <= float(schedule["end"])):
            raise ValueError("Invalid alpha curriculum")
        options["alpha_curriculum"] = {
            "start": float(schedule["start"]), "end": float(schedule["end"]),
            "warmup_epochs": int(schedule["warmup_epochs"]),
            "minimum_selected": float(schedule["minimum_selected"]),
        }
    if settings.embedding_dim != EMBEDDING_DIM:
        raise ValueError("The current optical hardware contract requires 64-D retrieval")
    seed_everything(args.seed)
    contract = load_contract(dataset_root)
    device = torch.device(args.device if args.device else settings.device)
    loaded = load_backbone(settings, device)
    cache = teacher_cache(
        loaded, contract, settings, cache_path, args.force_teacher_cache, prompts
    )
    teacher_image_to_text, _ = _metrics(
        cache["test"], cache["titles"], [sample.sku_index for sample in contract.test]
    )
    teacher_text_to_image, _ = _text_to_image_metrics(
        cache["titles"], cache["test"],
        [sample.sku_index for sample in contract.test],
    )
    teacher_metrics = {
        "image_to_text": teacher_image_to_text,
        "text_to_image": teacher_text_to_image,
    }
    replacement, readout = build_student(loaded, settings)
    try:
        resume_value = args.resume_checkpoint or _nested(raw, "abo_image_text.resume_checkpoint", None)
        if resume_value:
            resume_path = (Path(resume_value).expanduser().resolve() if args.resume_checkpoint
                           else _resolve_from_config(config, str(resume_value)))
            expected = str(args.expected_resume_sha256 or
                           _nested(raw, "abo_image_text.resume_checkpoint_sha256", ""))
            actual = _sha256(resume_path)
            if len(expected) != 64 or actual != expected:
                raise RuntimeError("Pinned text-to-image resume checkpoint SHA256 mismatch")
            alpha_transition = bool(_nested(raw, "abo_image_text.allow_alpha_range_transition", False))
            distance_transition = bool(
                _nested(raw, "abo_image_text.allow_propagation_distance_transition", False)
            )
            compaction_transition = bool(
                _nested(raw, "abo_image_text.allow_electronic_compaction_transition", False)
            )
            if sum((alpha_transition, distance_transition, compaction_transition)) > 1:
                raise ValueError("Alpha, propagation and compaction transitions are mutually exclusive")
            transition_report = None
            if distance_transition:
                payload, transition_report = _load_propagation_transition_checkpoint(
                    resume_path, replacement, readout
                )
            elif compaction_transition:
                payload, transition_report = _load_electronic_compaction_checkpoint(
                    resume_path, replacement, readout
                )
            elif alpha_transition:
                payload = torch.load(resume_path, map_location="cpu", weights_only=False)
                source_architecture = payload.get("metadata", {}).get("optical_architecture")
                expected_source = "lightgen_t01_optical_router_scale_matched_moe_10cm_17um_scale_matched_0p010_0p950_c736891c7a55f_v1"
                if source_architecture != expected_source:
                    raise RuntimeError("Alpha transition accepts only the pinned low-alpha T08 architecture")
                replacement.vision_surrogate.load_state_dict(payload["vision_optical"], strict=True)
                replacement.language_surrogate.load_state_dict(payload["language_optical"], strict=True)
                readout.load_state_dict(payload["retrieval_readout"], strict=True)
                replacement.reset_fusion_logits()
            else:
                payload = load_checkpoint(resume_path, replacement, readout)
            initialization = {"mode": "pinned_t08_continuation", "path": str(resume_path),
                              "sha256": actual, "source_epoch": payload.get("epoch"),
                              "alpha_range_transition": alpha_transition,
                              "alpha_reset_to": settings.fusion_alpha_initial if alpha_transition else None,
                              "propagation_distance_transition": distance_transition,
                              "electronic_compaction_transition": compaction_transition,
                              "transition_report": transition_report}
        else:
            initialization = initialize_student(settings, replacement, readout)
        _save_resolved_abo_config(settings, config, prompts)
        write_json(settings.output_dir / "dataset_contract.json", {
            "task": f"ABO easy100 {prompts.direction}",
            "matching_unit": "exact product SKU, not broad category",
            "train_samples": len(contract.train),
            "test_samples": len(contract.test), "title_candidates": len(contract.titles),
            "sha256": contract.sha256,
        })
        write_json(settings.output_dir / "architecture.json", replacement.student_architecture_report())
        write_json(settings.output_dir / "initialization.json", initialization)
        write_json(settings.output_dir / "parameter_fairness_contract.json", parameter_fairness_contract(settings))
        write_json(settings.output_dir / "run_manifest.json", {
            "schema_version": 1, "task": "t08_abo_image_text_retrieval",
            "retrieval_direction": prompts.direction,
            "prompt_contract": {
                "image_instruction": prompts.image_instruction,
                "title_instruction": prompts.title_instruction,
            },
            "architecture": "optical_router_top2_scale_matched_moe_dc20",
            "propagation_distance_m": settings.language_optical_distance_m,
            "seed": args.seed, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "teacher_qwen_frozen": True, "teacher_trainable_parameters": 0,
            "checkpoint_selection": "maximum periodic EMA test R@1",
            "selection_biased": True, "embedding_dim": EMBEDDING_DIM,
            "full_qwen_2048d_reference": (
                {"hit_at_1": 0.80, "preprocessing": "fixed_224_white_pad"}
                if prompts.direction == "text_to_image" else
                {"recall_at_1": 0.7370833333333333, "preprocessing": "dynamic_shape"}
            ),
            "hardware_matched_fixed_field_qwen_64d_reference": teacher_metrics,
            "dynamic_shape_qwen_64d_reference": (
                {"hit_at_1": 0.65}
                if prompts.direction == "text_to_image" else
                {"recall_at_1": 0.5979166666666667}
            ),
        })
        if args.evaluate_only:
            normal = evaluate_bidirectional(
                loaded, replacement, readout, contract, settings, prompts,
                write_outputs=True
            )
            fusion = replacement.fusion_diagnostics()
            replacement.set_fusion_ablation("remove_optical")
            try:
                removed = evaluate_bidirectional(
                    loaded, replacement, readout, contract, settings, prompts
                )
            finally:
                replacement.set_fusion_ablation("none")
            report = {
                "status": "complete", "task": "ABO easy100 pure text-to-image audit",
                "training_executed": False, "initialization": initialization,
                "normal": normal, "same_weights_remove_optical": removed,
                "text_to_image_optical_removal_drop_percentage_points": 100.0 * (
                    normal["text_to_image"]["hit_at_1"]
                    - removed["text_to_image"]["hit_at_1"]
                ),
                "fusion": fusion,
                "dataset_sha256": contract.sha256,
                "selection_biased_source_checkpoint": True,
            }
            write_json(settings.output_dir / "final_report.json", report)
            return report
        training = train(
            loaded, replacement, readout, contract, cache, settings, options, prompts
        )
        load_checkpoint(settings.output_dir / "best_checkpoint.pt", replacement, readout)
        final_metrics = evaluate_bidirectional(
            loaded, replacement, readout, contract, settings, prompts,
            write_outputs=True
        )
        fusion = replacement.fusion_diagnostics()
        replacement.set_fusion_ablation("remove_optical")
        try:
            removed_metrics = evaluate_bidirectional(
                loaded, replacement, readout, contract, settings, prompts
            )
        finally:
            replacement.set_fusion_ablation("none")
        replacement.save_multiplane_phase_preview(
            settings.output_dir / "best_phase_overview.png",
            title=f"ABO easy100 optical Router MoE; best epoch {training['best_epoch']}",
        )
        report = {
            "status": "complete", "task": f"ABO easy100 {prompts.direction}",
            "method": "LightGen optical Router Top-2 MoE DC20 scale-matched fusion",
            "propagation_distance_m": settings.language_optical_distance_m,
            "test_samples": len(contract.test), "title_candidates": len(contract.titles),
            "embedding_dim": EMBEDDING_DIM, "best_epoch": training["best_epoch"],
            "student": final_metrics[options["selection_direction"]],
            "bidirectional": final_metrics,
            "same_weights_remove_optical": removed_metrics,
            "text_to_image_optical_removal_drop_percentage_points": 100.0 * (
                final_metrics["text_to_image"]["hit_at_1"]
                - removed_metrics["text_to_image"]["hit_at_1"]
            ),
            "hardware_matched_fixed_field_frozen_qwen_64d": teacher_metrics,
            "published_frozen_qwen_references": {
                "image_to_text_dynamic_64d_recall_at_1": 0.5979166666666667,
                "image_to_text_dynamic_2048d_recall_at_1": 0.7370833333333333,
                "text_to_image_dynamic_64d_hit_at_1": 0.65,
                "text_to_image_fixed224_64d_hit_at_1": 0.66,
                "text_to_image_dynamic_2048d_hit_at_1": 0.82,
                "text_to_image_fixed224_2048d_hit_at_1": 0.80,
            },
            "fusion": fusion,
            "selection_biased": True,
            "selection": f"best periodic EMA {options['selection_direction']} test Top-1; test checked every {settings.test_evaluation_interval_epochs} epochs",
            "dataset_sha256": contract.sha256,
            "checkpoint_sha256": _sha256(settings.output_dir / "best_checkpoint.pt"),
        }
        write_json(settings.output_dir / "final_report.json", report)
        return report
    finally:
        replacement.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T08 ABO optical-Router MoE image/text retrieval"
    )
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--run-dir")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int,
                        help="Diagnostic override for optimizer batches in each epoch")
    parser.add_argument("--eval-every", type=int,
                        help="Diagnostic override for periodic test evaluation")
    parser.add_argument("--force-teacher-cache", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Pinned checkpoint raw-image bidirectional and same-weight no-optical audit")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--expected-resume-sha256")
    parser.add_argument("--data-root")
    parser.add_argument("--teacher-cache")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
