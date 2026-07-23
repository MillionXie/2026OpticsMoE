from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .settings import ExperimentSettings


def import_official_clip(settings: ExperimentSettings):
    repository = settings.clip.local_clip_repository
    if repository.is_dir():
        value = str(repository)
        if value not in sys.path:
            sys.path.insert(0, value)
    try:
        module = importlib.import_module("clip")
    except Exception as error:
        raise RuntimeError(
            "The official OpenAI CLIP package is required. Either keep the repository "
            f"at {repository} or install it with `pip install -e CLIP`."
        ) from error
    if not hasattr(module, "load") or not hasattr(module, "tokenize"):
        raise RuntimeError(f"Imported module {module!r} is not the official OpenAI CLIP package")
    return module


class FrozenClipTeacher(nn.Module):
    def __init__(self, settings: ExperimentSettings, device: torch.device) -> None:
        super().__init__()
        clip = import_official_clip(settings)
        download_root = (
            str(settings.clip.cache_dir) if settings.clip.cache_dir is not None else None
        )
        try:
            model, _ = clip.load(
                settings.clip.model_name,
                device=str(device),
                jit=False,
                download_root=download_root,
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load frozen CLIP teacher {settings.clip.model_name!r}. "
                "Check network access or pre-populate clip.cache_dir."
            ) from error
        self.clip_module = clip
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        dtype = next(self.model.parameters()).dtype
        features = self.model.encode_image(images.to(self.device, dtype=dtype, non_blocking=True))
        return F.normalize(features.float(), dim=-1)

    @torch.no_grad()
    def build_text_prototypes(
        self, class_names: list[str], templates: Iterable[str], batch_size: int = 256
    ) -> torch.Tensor:
        self.model.eval()
        templates = list(templates)
        if not templates:
            raise ValueError("At least one CLIP text template is required")
        texts = [
            template.format(name)
            for name in class_names
            for template in templates
        ]
        encoded = []
        for start in range(0, len(texts), batch_size):
            tokens = self.clip_module.tokenize(texts[start : start + batch_size]).to(
                self.device
            )
            encoded.append(F.normalize(self.model.encode_text(tokens).float(), dim=-1))
        features = torch.cat(encoded).reshape(len(class_names), len(templates), -1)
        return F.normalize(features.mean(1), dim=-1)

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.detach().float().exp().clamp(max=100).cpu())


def save_text_prototypes(
    path: Path,
    prototypes: torch.Tensor,
    class_names: list[str],
    settings: ExperimentSettings,
    logit_scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes.detach().cpu().float(),
            "class_names": list(class_names),
            "model_name": settings.clip.model_name,
            "templates": list(settings.clip.text_prompt_templates),
            "logit_scale": float(logit_scale),
            "schema_version": 1,
        },
        path,
    )
    report_path = path.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "model_name": settings.clip.model_name,
                "shape": list(prototypes.shape),
                "class_count": len(class_names),
                "templates": settings.clip.text_prompt_templates,
                "logit_scale": float(logit_scale),
                "path": str(path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_text_prototypes(
    path: Path, class_names: list[str], settings: ExperimentSettings, device: torch.device
) -> tuple[torch.Tensor, float]:
    if not path.is_file():
        raise FileNotFoundError(
            f"CLIP text prototypes are missing: {path}. Run --phase clip_cache first."
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mismatches = {}
    if payload.get("model_name") != settings.clip.model_name:
        mismatches["model_name"] = (payload.get("model_name"), settings.clip.model_name)
    if payload.get("class_names") != class_names:
        mismatches["class_names"] = "cached ImageNet class names differ"
    if payload.get("templates") != settings.clip.text_prompt_templates:
        mismatches["templates"] = "cached prompt templates differ"
    if mismatches:
        raise RuntimeError(
            f"CLIP text prototype metadata mismatch: {mismatches}. Delete {path} and rebuild."
        )
    return payload["prototypes"].to(device=device, dtype=torch.float32), float(
        payload["logit_scale"]
    )
