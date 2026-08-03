from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    forward_base_hidden,
    move_inputs,
    preprocess_images,
    teacher_embeddings,
    validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import (
    LoadedBackbone,
)

from .datasets import ABOBundle, ABOSample, collate_abo, load_product_image
from .io_utils import canonical_digest


class _UniqueSampleDataset(Dataset[dict[str, Any]]):
    def __init__(
        self, samples: Sequence[ABOSample], image_size: int = 224
    ) -> None:
        self.samples = tuple(samples)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        rgb = load_product_image(
            sample.image_path, self.image_size, image_id=sample.image_id
        )
        return {
            "image": rgb,
            "image_id": sample.image_id,
            "item_id": sample.item_id,
            "item_index": sample.item_index,
            "image_path": str(sample.image_path),
            "product_type": sample.product_type,
            "split": sample.split,
            "quality_score": sample.quality_score,
        }


def teacher_cache_identity(bundle: ABOBundle, settings: Any) -> dict[str, Any]:
    return {
        "version": (
            "abo_qwen_embedding_cache_v2_raw_hidden"
            if settings.teacher_embedding_mode == "adapted_head"
            else "abo_qwen_embedding_cache_v1"
        ),
        "dataset": "abo",
        "manifest_sha256": bundle.manifest_digest,
        "model_id": settings.model_id,
        "instruction": settings.instruction,
        "embedding_dim": settings.embedding_dim,
        "raw_hidden_dim": (
            settings.teacher_raw_hidden_dim
            if settings.teacher_embedding_mode == "adapted_head"
            else None
        ),
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
    }


@torch.no_grad()
def build_teacher_cache(
    loaded: LoadedBackbone,
    bundle: ABOBundle,
    settings: Any,
    *,
    force: bool = False,
) -> None:
    expected_identity = teacher_cache_identity(bundle, settings)
    if settings.teacher_cache_path.is_file() and not force:
        payload = torch.load(settings.teacher_cache_path, map_location="cpu")
        if payload.get("identity") != expected_identity:
            raise RuntimeError(
                "Teacher embedding cache metadata differs from the fixed ABO "
                "manifest/Qwen settings. Delete it or pass --force-teacher-cache."
            )
        print(f"Teacher cache already valid: {settings.teacher_cache_path}")
        return
    samples: dict[str, ABOSample] = {}
    for dataset in (
        bundle.stage1_train,
        bundle.stage2_train,
        bundle.gallery,
        bundle.query,
    ):
        for sample in dataset.samples:
            samples.setdefault(sample.image_id, sample)
    ordered = [samples[key] for key in sorted(samples)]
    loader = DataLoader(
        _UniqueSampleDataset(ordered, settings.image_size),
        batch_size=settings.teacher_batch_size,
        shuffle=False,
        num_workers=settings.num_workers,
        pin_memory=loaded.device.type == "cuda",
        collate_fn=collate_abo,
        persistent_workers=settings.num_workers > 0,
    )
    embeddings: list[torch.Tensor] = []
    raw_hidden: list[torch.Tensor] = []
    image_ids: list[str] = []
    for batch_index, batch in enumerate(loader, start=1):
        inputs = preprocess_images(
            loaded.processor, batch["images"], settings.instruction
        )
        validate_token_budgets(inputs, settings)
        inputs = move_inputs(inputs, loaded.device)
        if settings.teacher_embedding_mode == "adapted_head":
            hidden = forward_base_hidden(loaded.model, inputs)
            positions = torch.arange(
                hidden.shape[1], device=hidden.device
            ).unsqueeze(0).expand_as(inputs["attention_mask"])
            positions = positions.masked_fill(inputs["attention_mask"].eq(0), -1)
            last_positions = positions.max(dim=1).values
            batch_rows = torch.arange(hidden.shape[0], device=hidden.device)
            pooled = hidden[batch_rows, last_positions].float()
            if not torch.isfinite(pooled).all():
                raise RuntimeError("Teacher raw pooled hidden contains NaN/Inf")
            raw_hidden.append(pooled.detach().cpu().to(torch.float16))
            values = torch.nn.functional.normalize(
                pooled[:, : settings.embedding_dim], p=2, dim=-1
            )
        else:
            values = teacher_embeddings(
                loaded.model, inputs, settings.embedding_dim
            )
        embeddings.append(values.detach().cpu().to(torch.float16))
        image_ids.extend(batch["image_ids"])
        if batch_index % settings.log_interval_batches == 0:
            print(
                f"[teacher_cache] cached={len(image_ids):,}/{len(ordered):,}"
            )
    packed = torch.cat(embeddings, dim=0)
    if packed.shape != (len(image_ids), settings.embedding_dim):
        raise RuntimeError(
            f"Teacher cache shape {tuple(packed.shape)} is inconsistent"
        )
    settings.teacher_cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "identity": expected_identity,
        "identity_sha256": canonical_digest(expected_identity),
        "image_ids": image_ids,
        "embeddings": packed,
    }
    if raw_hidden:
        packed_raw = torch.cat(raw_hidden, dim=0)
        if packed_raw.shape != (len(image_ids), settings.teacher_raw_hidden_dim):
            raise RuntimeError(
                f"Teacher raw hidden shape {tuple(packed_raw.shape)} is inconsistent"
            )
        payload["raw_hidden"] = packed_raw
    torch.save(payload, settings.teacher_cache_path)
    print(
        f"Saved {len(image_ids):,} Teacher embeddings "
        f"{tuple(packed.shape)} to {settings.teacher_cache_path}"
    )


class TeacherEmbeddingStore:
    def __init__(
        self,
        bundle: ABOBundle,
        settings: Any,
        *,
        apply_adapter: bool | None = None,
    ) -> None:
        if not settings.teacher_cache_path.is_file():
            raise FileNotFoundError(
                f"Teacher cache is missing: {settings.teacher_cache_path}. "
                "Run --phase cache_teacher first."
            )
        payload = torch.load(settings.teacher_cache_path, map_location="cpu")
        expected = teacher_cache_identity(bundle, settings)
        if payload.get("identity") != expected:
            raise RuntimeError(
                "Teacher cache identity mismatch; it cannot be reused for this manifest."
            )
        self.embeddings = payload["embeddings"].float()
        self.raw_hidden = (
            payload["raw_hidden"].float()
            if "raw_hidden" in payload
            else None
        )
        self.index = {
            str(image_id): index
            for index, image_id in enumerate(payload["image_ids"])
        }
        if len(self.index) != len(payload["image_ids"]):
            raise RuntimeError("Teacher cache contains duplicate image IDs")
        use_adapter = (
            settings.teacher_embedding_mode == "adapted_head"
            if apply_adapter is None
            else bool(apply_adapter)
        )
        if use_adapter:
            if self.raw_hidden is None:
                raise RuntimeError(
                    "Adapted Teacher requires cached 2048-D raw hidden states"
                )
            if not settings.teacher_adapter_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Teacher adapter checkpoint is missing: "
                    f"{settings.teacher_adapter_checkpoint}. Run --phase train_teacher_adapter."
                )
            from .teacher_adapter import NormalizedTeacherAdapter

            adapter = NormalizedTeacherAdapter(
                self.raw_hidden.shape[1], settings.embedding_dim
            )
            checkpoint = torch.load(
                settings.teacher_adapter_checkpoint, map_location="cpu"
            )
            adapter.load_state_dict(checkpoint["teacher_adapter"])
            adapter.eval()
            with torch.no_grad():
                chunks = [
                    adapter(chunk)
                    for chunk in self.raw_hidden.split(4096)
                ]
            self.embeddings = torch.cat(chunks, dim=0)

    def get(
        self, image_ids: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        missing = [image_id for image_id in image_ids if image_id not in self.index]
        if missing:
            raise KeyError(f"Teacher cache is missing image IDs: {missing[:5]}")
        rows = torch.tensor(
            [self.index[image_id] for image_id in image_ids], dtype=torch.long
        )
        return self.embeddings.index_select(0, rows).to(
            device, non_blocking=True
        )

    def get_raw(
        self, image_ids: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        if self.raw_hidden is None:
            raise RuntimeError("Teacher cache does not contain raw pooled hidden states")
        missing = [image_id for image_id in image_ids if image_id not in self.index]
        if missing:
            raise KeyError(f"Teacher cache is missing image IDs: {missing[:5]}")
        rows = torch.tensor(
            [self.index[image_id] for image_id in image_ids], dtype=torch.long
        )
        return self.raw_hidden.index_select(0, rows).to(
            device, non_blocking=True
        )
