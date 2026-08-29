from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .offline_tail import LanguageGlobalOfflineTail


EXPECTED_CACHE_KEYS = {
    "packed_block2_inputs",
    "offsets",
    "lengths",
    "labels",
    "split_codes",
    "orders",
}
SUPPORTED_CHECKPOINT_ARCHITECTURES = {
    "vision2_language2_moe4_10cm_robust_bounded_fusion_v2": 0.10,
    "vision2_language2_moe4_10cm_warmstart5_stage_b_v1": 0.05,
}

# The CCD-noise continuation deliberately keeps the warmstart5 checkpoint
# architecture label so old tensors load strictly, while changing only the
# construction-time optical-fusion floor from 5% to 1%.  Both values are
# audited contracts; no arbitrary floor is accepted.
AUDITED_ALTERNATE_FUSION_FLOORS = {
    "vision2_language2_moe4_10cm_warmstart5_stage_b_v1": {0.01},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_stage_dir(session_dir: Path) -> Path:
    value = session_dir.expanduser().resolve()
    if (value / "offline_downstream" / "contract.json").is_file():
        stage = value
    else:
        stage = value / "04_language_global"
    if not (stage / "offline_downstream" / "contract.json").is_file():
        raise FileNotFoundError(
            "Language-global offline contract is missing below "
            f"{stage / 'offline_downstream'}"
        )
    return stage


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"order", "key", "split", "sku_index", "sku_name"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Manifest {path} lacks required columns {sorted(required)}")
    return rows


def _require_sha256(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        )


@dataclass(frozen=True)
class OfflineQuickData:
    stage_dir: Path
    contract_path: Path
    contract: dict[str, Any]
    rows: tuple[dict[str, str], ...]
    keys: tuple[str, ...]
    groups: tuple[torch.Tensor, ...]
    labels: torch.Tensor
    split_codes: torch.Tensor
    ccd_uint8: torch.Tensor
    ccd_sha256: tuple[str, ...]

    def indexes_for_split(self, split: str) -> torch.Tensor:
        mapping = self.contract["split_codes"]
        if split not in mapping:
            raise ValueError(f"Unknown split {split!r}")
        return torch.nonzero(
            self.split_codes.eq(int(mapping[split])), as_tuple=False
        ).flatten()

    def batch(
        self, indexes: Iterable[int], device: torch.device
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        selected = [int(index) for index in indexes]
        if not selected:
            raise ValueError("Offline batch cannot be empty")
        groups = [self.groups[index].to(device=device, dtype=torch.float32) for index in selected]
        index_tensor = torch.tensor(selected, dtype=torch.long)
        ccd = self.ccd_uint8.index_select(0, index_tensor).to(
            device=device, dtype=torch.float32
        )
        labels = self.labels.index_select(0, index_tensor).to(device)
        return groups, ccd, labels


def _validate_manifest_and_cache(
    contract: dict[str, Any],
    rows: list[dict[str, str]],
    cache: dict[str, torch.Tensor],
) -> tuple[tuple[str, ...], tuple[torch.Tensor, ...]]:
    if set(cache) != EXPECTED_CACHE_KEYS:
        raise RuntimeError(
            f"Offline cache keys must be exactly {sorted(EXPECTED_CACHE_KEYS)}, "
            f"got {sorted(cache)}"
        )
    count = int(contract["sample_count"])
    if len(rows) != count:
        raise RuntimeError(f"Manifest has {len(rows)} rows; contract requires {count}")
    expected_tensor_specs = {
        "offsets": (torch.int64, (count + 1,)),
        "lengths": (torch.int64, (count,)),
        "labels": (torch.int64, (count,)),
        "split_codes": (torch.uint8, (count,)),
        "orders": (torch.int64, (count,)),
    }
    for name, (dtype, shape) in expected_tensor_specs.items():
        value = cache[name]
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
            raise RuntimeError(
                f"Cache tensor {name} must be {dtype} {shape}, got "
                f"{getattr(value, 'dtype', None)} {getattr(value, 'shape', None)}"
            )
    packed = cache["packed_block2_inputs"]
    construction = contract["tail_construction"]
    width = int(construction["width"])
    max_tokens = int(construction["max_tokens"])
    if (
        not isinstance(packed, torch.Tensor)
        or packed.dtype != torch.float32
        or packed.ndim != 2
        or packed.shape[1] != width
        or not torch.isfinite(packed).all()
    ):
        raise RuntimeError(f"Packed Block-2 input must be finite float32 [sumL,{width}]")
    offsets = cache["offsets"]
    lengths = cache["lengths"]
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(packed):
        raise RuntimeError("Offline cache offsets do not span the packed tensor")
    if not torch.equal(offsets[1:] - offsets[:-1], lengths):
        raise RuntimeError("Offline cache lengths disagree with offsets")
    if torch.any(lengths <= 0) or torch.any(lengths > max_tokens):
        raise RuntimeError("Offline cache contains an invalid token length")
    if not torch.equal(cache["orders"], torch.arange(count, dtype=torch.int64)):
        raise RuntimeError("Offline cache order is not contiguous")

    split_codes = {str(name): int(code) for name, code in contract["split_codes"].items()}
    if set(split_codes) != {"train", "gallery", "test"} or len(set(split_codes.values())) != 3:
        raise RuntimeError("Offline split-code contract must uniquely define train/gallery/test")
    keys: list[str] = []
    observed_split_counts = {name: 0 for name in split_codes}
    observed_class_splits: dict[int, dict[str, int]] = defaultdict(
        lambda: {name: 0 for name in split_codes}
    )
    observed_class_names: dict[int, str] = {}
    for index, row in enumerate(rows):
        try:
            order = int(row["order"])
            label = int(row["sku_index"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"Manifest row {index} has non-integer metadata") from error
        key = row["key"]
        split = row["split"]
        if order != index or int(cache["orders"][index]) != index:
            raise RuntimeError(f"Manifest/cache order mismatch at row {index}")
        if not key or Path(key).name != key or key in keys:
            raise RuntimeError(f"Manifest contains invalid or duplicate key {key!r}")
        if split not in split_codes:
            raise RuntimeError(f"Manifest contains unsupported split {split!r}")
        if int(cache["labels"][index]) != label:
            raise RuntimeError(f"Manifest/cache label mismatch at row {index}")
        if int(cache["split_codes"][index]) != split_codes[split]:
            raise RuntimeError(f"Manifest/cache split mismatch at row {index}")
        if label in observed_class_names and observed_class_names[label] != row["sku_name"]:
            raise RuntimeError(f"Class {label} has multiple names")
        observed_class_names[label] = row["sku_name"]
        observed_split_counts[split] += 1
        observed_class_splits[label][split] += 1
        keys.append(key)
    if len(set(keys)) != count:
        raise RuntimeError("Offline manifest keys are not unique")
    if _sha256_text("\n".join(keys)) != contract["ordered_keys_sha256"]:
        raise RuntimeError("Offline ordered key digest does not match the contract")
    if observed_split_counts != {
        name: int(value) for name, value in contract["split_counts"].items()
    }:
        raise RuntimeError("Offline split counts do not match the contract")
    class_names = list(contract["class_names"])
    if sorted(observed_class_names) != list(range(len(class_names))):
        raise RuntimeError("Offline class labels must be contiguous from zero")
    if [observed_class_names[index] for index in range(len(class_names))] != class_names:
        raise RuntimeError("Offline class names do not match the contract")
    expected_class_splits = {
        int(label): {name: int(value) for name, value in counts.items()}
        for label, counts in contract["class_split_counts"].items()
    }
    if dict(observed_class_splits) != expected_class_splits:
        raise RuntimeError("Offline per-class split counts do not match the contract")
    profile = str(contract.get("profile", ""))
    if profile not in {"quick210", "generic"}:
        raise RuntimeError(f"Unsupported offline dataset profile {profile!r}")
    if profile == "quick210":
        expected_per_class = {"train": 10, "gallery": 1, "test": 10}
        if (
            count != 210
            or len(class_names) != 10
            or observed_split_counts
            != {"train": 100, "gallery": 10, "test": 100}
            or any(
                observed_class_splits[index] != expected_per_class
                for index in range(10)
            )
        ):
            raise RuntimeError(
                "quick210 must contain 10 classes and exactly 10/1/10 "
                "train/gallery/test samples per class"
            )
        if (
            contract.get("upstream_source") != "simulation"
            or contract.get("measured_upstream_stages") != []
        ):
            raise RuntimeError(
                "quick210 requires simulation upstream and only the fourth layer measured"
            )

    groups = tuple(
        packed[int(offsets[index]) : int(offsets[index + 1])]
        for index in range(count)
    )
    return tuple(keys), groups


def _load_strict_ccd(
    ccd_dir: Path,
    keys: tuple[str, ...],
    ccd_contract: dict[str, Any],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if ccd_contract.get("background_subtraction") is not False:
        raise RuntimeError("Offline quick path forbids background subtraction")
    if ccd_contract.get("resizing") is not False:
        raise RuntimeError("Offline quick path forbids CCD resizing")
    if ccd_contract.get("mode") != "L" or ccd_contract.get("dtype") != "uint8":
        raise RuntimeError("Offline quick path requires mode-L uint8 CCD transport")
    shape = tuple(int(value) for value in ccd_contract["shape_hw"])
    if shape != (478, 478):
        raise RuntimeError(f"Offline quick path requires exact 478x478 CCD, got {shape}")
    if not ccd_dir.is_dir():
        raise FileNotFoundError(f"CCD capture directory is missing: {ccd_dir}")
    expected_names = {f"{key}.png" for key in keys}
    observed_names = {path.name for path in ccd_dir.glob("*.png") if path.is_file()}
    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)
    if missing or unexpected:
        raise RuntimeError(
            "CCD key set does not match the manifest: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    values: list[torch.Tensor] = []
    digests: list[str] = []
    for key in keys:
        path = ccd_dir / f"{key}.png"
        with Image.open(path) as image:
            if image.mode != "L":
                raise RuntimeError(f"CCD {path} must be 8-bit mode L, got {image.mode}")
            if image.size != (shape[1], shape[0]):
                raise RuntimeError(f"CCD {path} must be 478x478, got {image.size}")
            array = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array)
        if bool(ccd_contract["flip_vertical_after_load"]):
            tensor = torch.flip(tensor, (-2,))
        if bool(ccd_contract["flip_horizontal_after_load"]):
            tensor = torch.flip(tensor, (-1,))
        values.append(tensor.contiguous())
        digests.append(_sha256(path))
    return torch.stack(values, dim=0), tuple(digests)


def load_offline_quick_data(
    session_dir: str | Path, *, ccd_dir: str | Path | None = None
) -> OfflineQuickData:
    stage_dir = _resolve_stage_dir(Path(session_dir))
    offline_dir = stage_dir / "offline_downstream"
    contract_path = offline_dir / "contract.json"
    contract = _read_json(contract_path)
    if (
        int(contract.get("schema_version", -1)) != 1
        or contract.get("type") != "language_global_quick_offline_full_parity"
        or contract.get("stage") != "language_global"
    ):
        raise RuntimeError("Unsupported Language-global offline contract")
    architecture = str(contract.get("checkpoint_architecture", ""))
    if architecture not in SUPPORTED_CHECKPOINT_ARCHITECTURES:
        raise RuntimeError(
            "Offline quick tail requires an explicitly audited 10 cm checkpoint "
            f"architecture, got {architecture!r}"
        )
    construction = contract.get("tail_construction")
    if not isinstance(construction, dict):
        raise RuntimeError("Offline contract has no tail construction")
    expected_floor = SUPPORTED_CHECKPOINT_ARCHITECTURES[architecture]
    observed_floor = float(construction.get("minimum_optical_fusion", -1.0))
    allowed_floors = {expected_floor} | AUDITED_ALTERNATE_FUSION_FLOORS.get(
        architecture, set()
    )
    if not any(abs(observed_floor - value) <= 1.0e-12 for value in allowed_floors):
        raise RuntimeError(
            "Offline fusion floor does not match its checkpoint architecture: "
            f"expected one of {sorted(allowed_floors)}, got {observed_floor}"
        )
    ccd_contract = contract.get("ccd_contract")
    if not isinstance(ccd_contract, dict):
        raise RuntimeError("Offline contract has no CCD contract")
    if (
        contract.get("manifest_relative_path") != "../../manifest.csv"
        or contract.get("cache_file") != "cache.pt"
        or contract.get("state_file") != "downstream_state.pt"
        or ccd_contract.get("directory_relative_to_stage") != "ccd_captured"
    ):
        raise RuntimeError("Offline contract contains an unsupported payload path")
    cache_path = offline_dir / "cache.pt"
    state_path = offline_dir / "downstream_state.pt"
    _require_sha256(cache_path, str(contract["cache_sha256"]), "offline cache")
    _require_sha256(state_path, str(contract["state_sha256"]), "offline tail state")
    manifest_path = stage_dir.parent / "manifest.csv"
    _require_sha256(
        manifest_path, str(contract["manifest_sha256"]), "hardware manifest"
    )
    rows = _read_manifest(manifest_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cache, dict):
        raise RuntimeError("Offline cache must be a tensor dictionary")
    keys, groups = _validate_manifest_and_cache(contract, rows, cache)
    capture_dir = (
        Path(ccd_dir).expanduser().resolve()
        if ccd_dir is not None
        else stage_dir / str(ccd_contract["directory_relative_to_stage"])
    )
    ccd_uint8, ccd_sha256 = _load_strict_ccd(capture_dir, keys, ccd_contract)
    return OfflineQuickData(
        stage_dir=stage_dir,
        contract_path=contract_path,
        contract=contract,
        rows=tuple(rows),
        keys=keys,
        groups=groups,
        labels=cache["labels"],
        split_codes=cache["split_codes"],
        ccd_uint8=ccd_uint8,
        ccd_sha256=ccd_sha256,
    )


def load_offline_tail(data: OfflineQuickData, device: torch.device) -> LanguageGlobalOfflineTail:
    tail = LanguageGlobalOfflineTail(**data.contract["tail_construction"]).to(device)
    state_path = data.stage_dir / "offline_downstream" / str(data.contract["state_file"])
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("Offline tail state must contain tensors only")
    nonfinite = [
        name
        for name, value in state.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise RuntimeError(f"Offline tail state contains non-finite tensors {nonfinite}")
    tail.load_state_dict(state, strict=True)
    observed = sum(parameter.numel() for parameter in tail.parameters())
    expected = int(data.contract["tail_trainable_parameter_count"])
    if observed != expected:
        raise RuntimeError(
            f"Offline tail parameter count mismatch: expected {expected}, observed {observed}"
        )
    tail.requires_grad_(True)
    # This matches hardware_bridge._set_hardware_finetune_mode: Block 2 keeps
    # gradients but remains in eval mode, so its configured dropout is disabled.
    tail.eval()
    return tail


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    embeddings = F.normalize(embeddings.float(), dim=-1)
    logits = embeddings @ embeddings.T / float(temperature)
    identity = torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    if not torch.all(positive.any(dim=1)):
        raise RuntimeError("Every contrastive anchor requires a same-class positive")
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_denominator = torch.logsumexp(logits.masked_fill(identity, -torch.inf), dim=1)
    log_probability = logits - log_denominator[:, None]
    return -(
        (log_probability * positive).sum(dim=1) / positive.sum(dim=1)
    ).mean()


def episodic_prototype_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    classes = torch.unique(labels, sorted=True)
    normalized = F.normalize(embeddings.float(), dim=-1)
    prototypes: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for target, class_id in enumerate(classes):
        indexes = torch.nonzero(labels.eq(class_id), as_tuple=False).flatten()
        if len(indexes) < 2:
            raise RuntimeError("Every episodic class requires support and query samples")
        support_offset = int(torch.randint(len(indexes), (), device=indexes.device).item())
        prototypes.append(normalized[indexes[support_offset]])
        query_indexes = torch.cat(
            (indexes[:support_offset], indexes[support_offset + 1 :])
        )
        queries.append(normalized[query_indexes])
        targets.append(
            torch.full(
                (len(query_indexes),), target, dtype=torch.long, device=labels.device
            )
        )
    logits = torch.cat(queries) @ F.normalize(torch.stack(prototypes), dim=-1).T
    return F.cross_entropy(logits / float(temperature), torch.cat(targets))


def _split_batches(indexes: torch.Tensor, batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(indexes), batch_size):
        yield [int(value) for value in indexes[start : start + batch_size]]


@torch.no_grad()
def _embeddings(
    tail: LanguageGlobalOfflineTail,
    data: OfflineQuickData,
    indexes: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    tail.eval()
    chunks = []
    for selected in _split_batches(indexes, batch_size):
        groups, ccd, _ = data.batch(selected, device)
        chunks.append(tail(groups, ccd).detach().cpu())
    if not chunks:
        raise RuntimeError("Cannot embed an empty split")
    return torch.cat(chunks, dim=0)


def _evaluate(
    tail: LanguageGlobalOfflineTail,
    data: OfflineQuickData,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    gallery_indexes = data.indexes_for_split("gallery")
    test_indexes = data.indexes_for_split("test")
    gallery = F.normalize(
        _embeddings(tail, data, gallery_indexes, device, batch_size).float(), dim=-1
    )
    query = F.normalize(
        _embeddings(tail, data, test_indexes, device, batch_size).float(), dim=-1
    )
    gallery_labels = data.labels.index_select(0, gallery_indexes)
    truth = data.labels.index_select(0, test_indexes)
    class_names = list(data.contract["class_names"])
    classes = torch.arange(len(class_names), dtype=torch.long)
    missing = [int(label) for label in classes if not gallery_labels.eq(label).any()]
    if missing:
        raise RuntimeError(f"Offline gallery is missing class prototypes {missing}")
    prototypes = torch.stack(
        [F.normalize(gallery[gallery_labels.eq(label)].mean(dim=0), dim=0) for label in classes]
    )
    scores = query @ prototypes.T
    ranking = scores.argsort(dim=1, descending=True)
    top1 = ranking[:, 0]
    top3 = ranking[:, : min(3, len(class_names))]
    correct1 = top1.eq(truth)
    correct3 = top3.eq(truth[:, None]).any(dim=1)
    ranks = torch.stack(
        [
            torch.nonzero(ranking[index].eq(truth[index]), as_tuple=False)[0, 0] + 1
            for index in range(len(truth))
        ]
    ).float()
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.long)
    for true, predicted in zip(truth, top1):
        confusion[int(true), int(predicted)] += 1
    per_class = {}
    for label, name in enumerate(class_names):
        selected = truth.eq(label)
        per_class[name] = {
            "query_count": int(selected.sum()),
            "top1_accuracy": float(correct1[selected].float().mean()),
            "top3_accuracy": float(correct3[selected].float().mean()),
        }
    return {
        "system": "quick210_offline_language_global_full_parity",
        "query_count": int(len(test_indexes)),
        "gallery_image_count": int(len(gallery_indexes)),
        "class_count": len(class_names),
        "gallery_aggregation": "mean_prototype",
        "top1_retrieval_accuracy": float(correct1.float().mean()),
        "top3_retrieval_accuracy": float(correct3.float().mean()),
        "mrr": float((1.0 / ranks).mean()),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in module.state_dict().items()
    }


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def finetune_offline_quick(
    *,
    session_dir: str | Path,
    ccd_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    device_name: str = "auto",
    epochs: int | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    data = load_offline_quick_data(session_dir, ccd_dir=ccd_dir)
    contract = data.contract
    training = contract["training_contract"]
    device = _select_device(device_name)
    tail = load_offline_tail(data, device)
    if validate_only:
        return {
            "validated": True,
            "samples": len(data.rows),
            "train": int(len(data.indexes_for_split("train"))),
            "gallery": int(len(data.indexes_for_split("gallery"))),
            "test": int(len(data.indexes_for_split("test"))),
            "device": str(device),
            "tail_parameters": sum(parameter.numel() for parameter in tail.parameters()),
        }

    epoch_count = int(training["recommended_epochs"] if epochs is None else epochs)
    if epoch_count <= 0:
        raise ValueError("Offline fine-tuning epochs must be positive")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_indexes = data.indexes_for_split("train")
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in train_indexes:
        grouped[int(data.labels[int(index)])].append(int(index))
    class_count = int(training["pk_classes"])
    images_per_class = int(training["pk_images_per_class"])
    if len(grouped) != class_count or any(
        len(indexes) < images_per_class for indexes in grouped.values()
    ):
        raise RuntimeError(
            "Offline PK contract does not match the captured training split"
        )
    batch_size = class_count * images_per_class
    steps = max(1, len(train_indexes) // batch_size)
    head_parameters = [
        *tail.retrieval_norm.parameters(),
        *tail.retrieval_projection.parameters(),
    ]
    head_ids = {id(parameter) for parameter in head_parameters}
    base_parameters = [
        parameter for parameter in tail.parameters() if id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": base_parameters,
                "lr": float(training["learning_rate"]),
                "group_name": "downstream_electronic",
            },
            {
                "params": head_parameters,
                "lr": float(training["readout_learning_rate"]),
                "group_name": "retrieval_readout",
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    best_loss = float("inf")
    best_epoch = 0
    best_state = _cpu_state(tail)
    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, epoch_count + 1):
        # Keep dropout disabled while retaining gradients, matching the server path.
        tail.eval()
        epoch_total = 0.0
        epoch_supcon = 0.0
        epoch_prototype = 0.0
        for _ in range(steps):
            selected: list[int] = []
            for label in sorted(grouped):
                permutation = torch.randperm(len(grouped[label]), generator=generator)
                selected.extend(
                    grouped[label][int(offset)]
                    for offset in permutation[:images_per_class]
                )
            groups, ccd, labels = data.batch(selected, device)
            embeddings = tail(groups, ccd)
            supcon = supervised_contrastive_loss(
                embeddings,
                labels,
                float(training["supervised_contrastive_temperature"]),
            )
            prototype = episodic_prototype_loss(
                embeddings,
                labels,
                float(training["episodic_prototype_temperature"]),
            )
            loss = supcon + prototype
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(tail.parameters()),
                float(training["gradient_clip_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            epoch_total += float(loss.detach())
            epoch_supcon += float(supcon.detach())
            epoch_prototype += float(prototype.detach())
        average = epoch_total / steps
        row = {
            "epoch": epoch,
            "train_loss": average,
            "supervised_contrastive": epoch_supcon / steps,
            "episodic_prototype": epoch_prototype / steps,
            "block2_optical_fusion": float(tail.block2_optical_fusion.detach()),
        }
        log_rows.append(row)
        print(
            f"[offline_language_global] epoch={epoch:03d}/{epoch_count:03d} "
            f"loss={average:.5f} supcon={row['supervised_contrastive']:.5f} "
            f"prototype={row['episodic_prototype']:.5f} "
            f"gate={row['block2_optical_fusion']:.4f}",
            flush=True,
        )
        if average < best_loss:
            best_loss = average
            best_epoch = epoch
            best_state = _cpu_state(tail)
    tail.load_state_dict(best_state, strict=True)
    metrics = _evaluate(tail, data, device, batch_size)
    result_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else data.stage_dir / "offline_results"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    state_path = result_dir / "best_offline_tail_state.pt"
    torch.save(best_state, state_path)
    _write_csv(result_dir / "train_log.csv", log_rows)
    ccd_fingerprint = _sha256_text(
        "\n".join(
            f"{key}:{digest}" for key, digest in zip(data.keys, data.ccd_sha256)
        )
    )
    _write_json(
        result_dir / "ccd_inventory.json",
        {
            "schema_version": 1,
            "sample_count": len(data.keys),
            "ordered_ccd_set_sha256": ccd_fingerprint,
            "transport": "unmodified 478x478 mode-L uint8 PNG before contract flips",
            "background_subtraction": False,
            "files": [
                {"order": index, "key": key, "sha256": digest}
                for index, (key, digest) in enumerate(
                    zip(data.keys, data.ccd_sha256)
                )
            ],
        },
    )
    report = {
        **metrics,
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "epochs": epoch_count,
        "optimizer_steps_per_epoch": steps,
        "device": str(device),
        "tail_trainable_parameters": sum(
            parameter.numel() for parameter in tail.parameters()
        ),
        "block2_optical_fusion": float(tail.block2_optical_fusion.detach()),
        "source_checkpoint_sha256": contract["source_checkpoint_sha256"],
        "contract_sha256": _sha256(data.contract_path),
        "ccd_set_sha256": ccd_fingerprint,
        "tail_state": str(state_path),
        "tail_state_sha256": _sha256(state_path),
        "background_subtraction": False,
        "test_selection_rule": "best checkpoint selected only by train loss",
    }
    _write_json(result_dir / "metrics.json", report)
    print(
        f"[offline_language_global] test_top1={metrics['top1_retrieval_accuracy']:.4f} "
        f"best_epoch={best_epoch} output={result_dir}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the cached Language-global electronic tail without loading "
            "Qwen, Transformers, the source images, or an optical simulator"
        )
    )
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--ccd-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report = finetune_offline_quick(
        session_dir=args.session_dir,
        ccd_dir=args.ccd_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        epochs=args.epochs,
        validate_only=args.validate_only,
    )
    if args.validate_only:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
