from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PCA_SCHEMA_VERSION = 1


class FixedPCAProjection(nn.Module):
    """One immutable PCA coordinate system shared by a complete Qwen stack."""

    def __init__(
        self,
        mean: torch.Tensor,
        components: torch.Tensor,
        explained_variance_ratio: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if mean.ndim != 1 or components.ndim != 2 or components.shape[0] != mean.numel():
            raise ValueError("PCA mean/components dimensions are inconsistent")
        self.register_buffer("mean", mean.detach().float().clone())
        self.register_buffer("components", components.detach().float().clone())
        ratio = (
            explained_variance_ratio.detach().float().clone()
            if explained_variance_ratio is not None
            else torch.empty(0, dtype=torch.float32)
        )
        self.register_buffer("explained_variance_ratio", ratio)

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def latent_dim(self) -> int:
        return int(self.components.shape[1])

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.input_dim:
            raise ValueError(f"Expected hidden dimension {self.input_dim}, got {hidden.shape[-1]}")
        normalized = F.layer_norm(hidden.float(), (self.input_dim,))
        return (normalized - self.mean) @ self.components

    def encode_additive_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Project a native DeepStack additive delta without adding PCA mean.

        A zero visual injection must remain exactly zero. Applying the normal
        encode formula to a delta would incorrectly turn zero rows into
        ``-mean @ components``.
        """
        if delta.shape[-1] != self.input_dim:
            raise ValueError(f"Expected delta dimension {self.input_dim}, got {delta.shape[-1]}")
        return delta.float() @ self.components

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected PCA latent dimension {self.latent_dim}, got {latent.shape[-1]}")
        return latent.float() @ self.components.transpose(0, 1) + self.mean


@dataclass
class StreamingPCAFitter:
    input_dim: int
    latent_dim: int
    max_tokens: int
    device: torch.device

    def __post_init__(self) -> None:
        if self.latent_dim > self.input_dim:
            raise ValueError("latent_dim cannot exceed input_dim")
        self.count = 0
        self.sum = torch.zeros(self.input_dim, dtype=torch.float64, device=self.device)
        self.gram = torch.zeros(
            self.input_dim,
            self.input_dim,
            dtype=torch.float64,
            device=self.device,
        )

    @torch.no_grad()
    def update(self, hidden: torch.Tensor) -> int:
        remaining = self.max_tokens - self.count
        if remaining <= 0:
            return 0
        flat = hidden.detach().reshape(-1, self.input_dim)
        if len(flat) > remaining:
            flat = flat[:remaining]
        normalized = F.layer_norm(flat.float(), (self.input_dim,)).to(self.device, torch.float64)
        self.sum += normalized.sum(0)
        self.gram += normalized.transpose(0, 1) @ normalized
        self.count += len(normalized)
        return len(normalized)

    @torch.no_grad()
    def finalize(self) -> tuple[FixedPCAProjection, dict[str, Any]]:
        if self.count <= self.latent_dim:
            raise RuntimeError(
                f"PCA needs more than {self.latent_dim} calibration tokens, got {self.count}"
            )
        mean = self.sum / self.count
        centered_gram = self.gram - self.count * torch.outer(mean, mean)
        covariance = centered_gram / max(self.count - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        components = eigenvectors[:, order[: self.latent_dim]]
        total_variance = eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
        ratio = eigenvalues[: self.latent_dim] / total_variance
        retained = ratio.sum()
        relative_error = torch.sqrt((1.0 - retained).clamp_min(0.0))
        projection = FixedPCAProjection(
            mean.float().cpu(),
            components.float().cpu(),
            ratio.float().cpu(),
        )
        report = {
            "calibration_token_count": self.count,
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "explained_variance_ratio": ratio.float().cpu().tolist(),
            "explained_variance_ratio_sum": float(retained),
            "relative_reconstruction_error_from_spectrum": float(relative_error),
        }
        return projection, report


def save_projection(
    path: Path,
    projection: FixedPCAProjection,
    report: dict[str, Any],
    *,
    stack: str,
    seed: int,
    source_taps: list[int | str],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PCA_SCHEMA_VERSION,
        "stack": stack,
        "seed": int(seed),
        "source_taps": source_taps,
        "mean": projection.mean.cpu(),
        "components": projection.components.cpu(),
        "explained_variance_ratio": projection.explained_variance_ratio.cpu(),
        "report": report,
    }
    torch.save(payload, path)
    digest = file_sha256(path)
    metadata = {
        **report,
        "schema_version": PCA_SCHEMA_VERSION,
        "stack": stack,
        "seed": int(seed),
        "source_taps": source_taps,
        "path": str(path),
        "sha256": digest,
        "mean_requires_grad": False,
        "components_requires_grad": False,
    }
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_projection(path: Path, expected_dim: int | None = None) -> FixedPCAProjection:
    if not path.is_file():
        raise FileNotFoundError(f"Fixed PCA projection is missing: {path}. Run --phase fit_pca.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if int(payload.get("schema_version", -1)) != PCA_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported PCA schema in {path}")
    projection = FixedPCAProjection(
        payload["mean"],
        payload["components"],
        payload.get("explained_variance_ratio"),
    )
    if expected_dim is not None and projection.input_dim != expected_dim:
        raise RuntimeError(
            f"PCA input dimension mismatch: saved={projection.input_dim}, expected={expected_dim}"
        )
    return projection


def projection_paths(settings: Any) -> tuple[Path, Path]:
    root = Path(settings.precompute_cache_dir) / "pca" / str(settings.manifest_digest)
    return root / "vision.pt", root / "language.pt"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
