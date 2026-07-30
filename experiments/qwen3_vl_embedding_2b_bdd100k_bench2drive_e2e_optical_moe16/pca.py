from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.qwen3_vl_embedding_2b_coco_duts_vision_optical_moe16_pretrain.pca import (
    FixedPCAProjection,
    fit_pca_matrix,
)

from .io_utils import atomic_torch_save, torch_load, write_json
from .modeling import preprocess_vision


@torch.no_grad()
def fit_bdd_pca(
    teacher: nn.Module,
    processor: Any,
    loader: Any,
    settings: Any,
) -> dict[str, Any]:
    samples: list[torch.Tensor] = []
    image_count = 0
    token_count = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(settings.random_seed)
    for batch in loader:
        if image_count >= settings.pca_calibration_images:
            break
        inputs = preprocess_vision(processor, batch["images"], teacher.device)
        packed, lengths = teacher.extract(
            inputs["pixel_values"], inputs["image_grid_thw"]
        )
        for group in packed.split(lengths):
            if image_count >= settings.pca_calibration_images:
                break
            count = min(
                settings.pca_tokens_per_image,
                group.shape[0],
                settings.pca_max_tokens - token_count,
            )
            if count <= 0:
                break
            if count < group.shape[0]:
                indices = torch.randperm(group.shape[0], generator=generator)[:count]
                group = group.index_select(0, indices.to(group.device))
            samples.append(group[:count].detach().cpu().to(torch.float16))
            token_count += count
            image_count += 1
            if token_count >= settings.pca_max_tokens:
                break
    if token_count < settings.pca_rank:
        raise RuntimeError(
            f"PCA collected only {token_count} tokens for rank={settings.pca_rank}"
        )
    matrix = torch.cat(samples, dim=0)[: settings.pca_max_tokens].float()
    device = torch.device(
        "cuda"
        if settings.pca_device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    projection, metrics = fit_pca_matrix(
        matrix.to(device),
        rank=settings.pca_rank,
        oversample=settings.pca_oversample,
        niter=settings.pca_niter,
        seed=settings.random_seed,
    )
    metadata = {
        **metrics,
        "dataset": "BDD100K",
        "source_hidden": "Qwen final native Vision block before merger",
        "student_contains_pca": False,
        "calibration_images": image_count,
        "calibration_tokens": token_count,
        "processor_min_pixels": settings.processor_min_pixels,
        "processor_max_pixels": settings.processor_max_pixels,
        "model_id": settings.model_id,
    }
    atomic_torch_save(
        settings.pca_path,
        {
            "mean": projection.mean.cpu(),
            "components": projection.components.cpu(),
            "metadata": metadata,
        },
    )
    write_json(settings.output_dir / "metrics" / "bdd_pca.json", metadata)
    return metadata


def load_pca(path: Path, device: torch.device) -> tuple[FixedPCAProjection, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"BDD Vision PCA is missing: {path}; run fit_pca")
    payload = torch_load(path)
    projection = FixedPCAProjection(payload["mean"], payload["components"]).to(device)
    if projection.input_dim != 1024 or projection.rank != 224:
        raise RuntimeError(
            f"Expected PCA 1024->224, got {projection.input_dim}->{projection.rank}"
        )
    return projection, dict(payload.get("metadata", {}))
