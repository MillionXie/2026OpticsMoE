from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .io_utils import atomic_torch_save, torch_load, write_json
from .modeling import preprocess_vision


class FixedPCAProjection(nn.Module):
    """Frozen 1024<->224 coordinate map used only for teacher targets."""

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
    ) -> None:
        super().__init__()
        if mean.ndim != 1:
            raise ValueError(f"PCA mean must be [D], got {tuple(mean.shape)}")
        if components.ndim != 2 or components.shape[0] != mean.shape[0]:
            raise ValueError(
                "PCA components must be [teacher_dim,rank], got "
                f"{tuple(components.shape)} for mean {tuple(mean.shape)}"
            )
        self.register_buffer("mean", mean.detach().float().contiguous())
        self.register_buffer(
            "components",
            components.detach().float().contiguous(),
        )

    @property
    def input_dim(self) -> int:
        return int(self.mean.numel())

    @property
    def rank(self) -> int:
        return int(self.components.shape[1])

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"PCA encode expected hidden dim {self.input_dim}, got "
                f"{hidden.shape[-1]}"
            )
        return (hidden.float() - self.mean) @ self.components

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.rank:
            raise RuntimeError(
                f"PCA decode expected latent dim {self.rank}, got "
                f"{latent.shape[-1]}"
            )
        return latent.float() @ self.components.transpose(0, 1) + self.mean

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, rank={self.rank}, trainable=False"


def fit_pca_matrix(
    samples: torch.Tensor,
    *,
    rank: int,
    oversample: int,
    niter: int,
    seed: int,
) -> tuple[FixedPCAProjection, dict[str, Any]]:
    if samples.ndim != 2:
        raise ValueError(f"PCA samples must be [N,D], got {tuple(samples.shape)}")
    if samples.shape[0] < rank or samples.shape[1] < rank:
        raise ValueError(
            f"PCA rank={rank} requires at least {rank} samples and dimensions, "
            f"got {tuple(samples.shape)}"
        )
    if not torch.isfinite(samples).all():
        raise RuntimeError("PCA calibration samples contain NaN or Inf")
    torch.manual_seed(int(seed))
    if samples.is_cuda:
        torch.cuda.manual_seed_all(int(seed))
    values = samples.float()
    mean = values.mean(dim=0)
    centered = values - mean
    q = min(
        int(rank) + max(0, int(oversample)),
        int(centered.shape[0]),
        int(centered.shape[1]),
    )
    _, singular_values, vectors = torch.pca_lowrank(
        centered,
        q=q,
        center=False,
        niter=int(niter),
    )
    components = vectors[:, :rank].contiguous()
    projection = FixedPCAProjection(mean, components).to(values.device)
    with torch.no_grad():
        latent = projection.encode(values)
        reconstructed = projection.decode(latent)
        residual = values - reconstructed
        total_variance = centered.square().sum().clamp_min(1e-12)
        retained_variance = singular_values[:rank].square().sum()
        relative_error = (
            residual.norm() / values.norm().clamp_min(1e-12)
        ).item()
        centered_relative_error = (
            residual.norm() / centered.norm().clamp_min(1e-12)
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            values,
            reconstructed,
            dim=-1,
        ).mean().item()
    metadata = {
        "algorithm": "torch.pca_lowrank",
        "sample_count": int(samples.shape[0]),
        "teacher_dim": int(samples.shape[1]),
        "rank": int(rank),
        "oversample": int(oversample),
        "q": int(q),
        "niter": int(niter),
        "seed": int(seed),
        "explained_variance_ratio_total": float(
            (retained_variance / total_variance).item()
        ),
        "explained_variance_ratio_per_component": [
            float(value)
            for value in (
                singular_values[:rank].square() / total_variance
            ).detach().cpu()
        ],
        "relative_reconstruction_error": float(relative_error),
        "centered_relative_reconstruction_error": float(
            centered_relative_error
        ),
        "reconstruction_cosine": float(cosine),
    }
    return projection.cpu(), metadata


@torch.no_grad()
def fit_vision_pca(
    teacher: nn.Module,
    processor: Any,
    loader: Any,
    settings: Any,
    *,
    coco_manifest_digest: str,
) -> dict[str, Any]:
    calibration: list[torch.Tensor] = []
    collected = 0
    images_seen = 0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(settings.random_seed))
    for batch in loader:
        if images_seen >= settings.pca_calibration_images:
            break
        inputs = preprocess_vision(processor, batch["images"], teacher.device)
        packed, lengths = teacher.extract_packed(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
        groups = packed.split(lengths)
        for group in groups:
            if images_seen >= settings.pca_calibration_images:
                break
            count = min(
                int(settings.pca_tokens_per_image),
                int(group.shape[0]),
                int(settings.pca_max_tokens - collected),
            )
            if count <= 0:
                break
            if count == group.shape[0]:
                selected = group
            else:
                indices = torch.randperm(
                    group.shape[0],
                    generator=generator,
                )[:count].to(group.device)
                selected = group.index_select(0, indices)
            calibration.append(selected.detach().to("cpu", torch.float16))
            collected += count
            images_seen += 1
            if collected >= settings.pca_max_tokens:
                break
        if collected >= settings.pca_max_tokens:
            break
        if images_seen % 256 == 0:
            print(
                f"[fit_pca] images={images_seen:,} tokens={collected:,}",
                flush=True,
            )
    if collected < settings.pca_rank:
        raise RuntimeError(
            f"PCA calibration collected only {collected} tokens; "
            f"rank={settings.pca_rank} cannot be fitted"
        )
    matrix = torch.cat(calibration, dim=0)[: settings.pca_max_tokens]
    pca_device = _resolve_pca_device(settings.pca_device)
    matrix = matrix.to(pca_device, torch.float32)
    projection, metrics = fit_pca_matrix(
        matrix,
        rank=settings.pca_rank,
        oversample=settings.pca_oversample,
        niter=settings.pca_niter,
        seed=settings.random_seed,
    )
    state_digest = _projection_digest(projection)
    metadata = {
        **metrics,
        "dataset": "COCO train2017",
        "coco_manifest_sha256": coco_manifest_digest,
        "calibration_images": int(images_seen),
        "calibration_tokens": int(matrix.shape[0]),
        "processor_min_pixels": int(settings.processor_min_pixels),
        "processor_max_pixels": int(settings.processor_max_pixels),
        "model_id": settings.model_id,
        "source_hidden": "final_native_qwen_vision_block_pre_merger",
        "student_contains_pca": False,
        "projection_sha256": state_digest,
    }
    atomic_torch_save(
        settings.pca_path,
        {
            "mean": projection.mean.cpu(),
            "components": projection.components.cpu(),
            "metadata": metadata,
        },
    )
    write_json(settings.output_dir / "metrics" / "pca_fit.json", metadata)
    print(
        "[fit_pca] "
        f"tokens={matrix.shape[0]:,} rank={projection.rank} "
        f"explained={metrics['explained_variance_ratio_total']:.4f} "
        f"relative_error={metrics['relative_reconstruction_error']:.4f}",
        flush=True,
    )
    return metadata


def load_pca(path: Path, device: torch.device | str) -> tuple[FixedPCAProjection, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Vision PCA checkpoint is missing: {path}. Run --phase fit_pca."
        )
    payload = torch_load(path)
    required = {"mean", "components", "metadata"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError(
            f"Invalid PCA checkpoint {path}; required keys are {sorted(required)}"
        )
    projection = FixedPCAProjection(
        payload["mean"],
        payload["components"],
    ).to(device)
    if projection.input_dim != 1024 or projection.rank != 224:
        raise RuntimeError(
            f"PCA checkpoint has shape {projection.input_dim}->{projection.rank}; "
            "expected 1024->224"
        )
    if any(parameter.requires_grad for parameter in projection.parameters()):
        raise RuntimeError("Fixed PCA unexpectedly exposes trainable parameters")
    return projection, dict(payload["metadata"])


@torch.no_grad()
def pca_oracle_metrics(
    teacher: nn.Module,
    projection: FixedPCAProjection,
    processor: Any,
    loader: Any,
    settings: Any,
    *,
    max_batches: int = 16,
) -> dict[str, Any]:
    mse_sum = 0.0
    cosine_sum = 0.0
    relative_sum = 0.0
    token_count = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        inputs = preprocess_vision(processor, batch["images"], teacher.device)
        packed, _ = teacher.extract_packed(
            inputs["pixel_values"],
            inputs["image_grid_thw"],
        )
        reconstructed = projection.decode(projection.encode(packed))
        count = int(packed.shape[0])
        mse_sum += float(
            (packed.float() - reconstructed.float()).square().mean(dim=-1).sum()
        )
        cosine_sum += float(
            torch.nn.functional.cosine_similarity(
                packed.float(),
                reconstructed.float(),
                dim=-1,
            ).sum()
        )
        relative_sum += float(
            (
                (packed.float() - reconstructed.float()).norm(dim=-1)
                / packed.float().norm(dim=-1).clamp_min(1e-12)
            ).sum()
        )
        token_count += count
    if not token_count:
        raise RuntimeError("PCA oracle check received no visual tokens")
    result = {
        "tokens": token_count,
        "hidden_mse": mse_sum / token_count,
        "hidden_cosine": cosine_sum / token_count,
        "relative_reconstruction_error_mean_token": relative_sum / token_count,
        "pca_is_student_module": False,
    }
    write_json(settings.output_dir / "metrics" / "pca_oracle.json", result)
    if result["hidden_cosine"] < 0.8:
        warnings.warn(
            "PCA oracle cosine is below 0.8. The fixed teacher target may lose "
            "substantial information; no trainable adapter was inserted.",
            RuntimeWarning,
            stacklevel=2,
        )
    return result


def _resolve_pca_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "pca.device=cuda but CUDA is unavailable; falling back to CPU",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return torch.device(name)


def _projection_digest(projection: FixedPCAProjection) -> str:
    digest = hashlib.sha256()
    digest.update(projection.mean.detach().cpu().numpy().tobytes())
    digest.update(projection.components.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
