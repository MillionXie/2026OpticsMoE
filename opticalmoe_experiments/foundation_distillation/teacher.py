from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenImageTeacher(nn.Module):
    """Base class for normalized, frozen image-encoder features."""

    def __init__(
        self,
        model: nn.Module,
        *,
        backend: str,
        teacher_type: str,
        feature_type: str,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.backend = str(backend)
        self.teacher_type = str(teacher_type)
        self.feature_type = str(feature_type)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        features = self.encode_features(images)
        return F.normalize(features.float(), dim=-1)


class FrozenCLIPImageTeacher(FrozenImageTeacher):
    def __init__(self, model: nn.Module, backend: str) -> None:
        super().__init__(
            model,
            backend=backend,
            teacher_type="clip_image_encoder",
            feature_type="image_embedding",
        )

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(images)


class FrozenDINOv2ImageTeacher(FrozenImageTeacher):
    def __init__(self, model: nn.Module, backend: str = "transformers", feature_type: str = "cls") -> None:
        feature_type = str(feature_type).lower()
        if feature_type not in {"cls", "patch_mean"}:
            raise ValueError("DINOv2 feature_type must be 'cls' or 'patch_mean'.")
        super().__init__(
            model,
            backend=backend,
            teacher_type="dinov2_image_encoder",
            feature_type=feature_type,
        )

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=images)
        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError("DINOv2 model output must provide last_hidden_state.")
        tokens = outputs.last_hidden_state
        if tokens.ndim != 3 or tokens.shape[1] < 1:
            raise ValueError("DINOv2 last_hidden_state must have shape [B, 1+N, D].")
        if self.feature_type == "cls":
            return tokens[:, 0]
        if tokens.shape[1] < 2:
            raise ValueError("DINOv2 patch_mean requires at least one patch token.")
        return tokens[:, 1:].mean(dim=1)


class FrozenImageEncoder(FrozenCLIPImageTeacher):
    """Backward-compatible name for the original frozen CLIP wrapper."""


def _processor_image_size(processor) -> int:
    for value in (getattr(processor, "crop_size", None), getattr(processor, "size", None)):
        if isinstance(value, Mapping):
            for key in ("height", "width", "shortest_edge"):
                if value.get(key) is not None:
                    return int(value[key])
        if isinstance(value, int):
            return int(value)
    return 224


def _load_clip_backend(model_name: str, device: torch.device, requested_backend: str = "auto"):
    requested = str(requested_backend or "auto").lower()
    if requested not in {"auto", "open_clip", "clip"}:
        raise ValueError("CLIP teacher.backend must be auto, open_clip, or clip.")
    errors = []
    if requested in {"auto", "open_clip"}:
        try:
            import open_clip

            open_clip_name = str(model_name).replace("/", "-")
            model = open_clip.create_model(open_clip_name, pretrained="openai", device=device)
            return model, "open_clip"
        except Exception as exc:
            errors.append(f"open_clip: {exc}")
            if requested == "open_clip":
                raise RuntimeError(
                    "CLIP teacher backend open_clip is unavailable. Install with: pip install open_clip_torch\n"
                    f"Backend error: {exc}"
                ) from exc
    if requested in {"auto", "clip"}:
        try:
            import clip

            model, _unused_preprocess = clip.load(str(model_name), device=device, jit=False)
            return model, "clip"
        except Exception as exc:
            errors.append(f"clip: {exc}")
    details = "\n".join(errors)
    raise RuntimeError(
        "A CLIP image-encoder backend is required only when building the teacher cache. "
        "Install one with `pip install open_clip_torch` (recommended) or install OpenAI CLIP.\n"
        f"Backend errors:\n{details}"
    )


def _load_dinov2_backend(model_name: str, device: torch.device):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "DINOv2 teacher requires transformers.\nInstall with: pip install transformers"
        ) from exc
    processor = AutoImageProcessor.from_pretrained(str(model_name))
    model = AutoModel.from_pretrained(str(model_name)).to(device)
    preprocess_info = {
        "teacher_image_size": _processor_image_size(processor),
        "teacher_image_mean": list(getattr(processor, "image_mean", [0.485, 0.456, 0.406])),
        "teacher_image_std": list(getattr(processor, "image_std", [0.229, 0.224, 0.225])),
    }
    return model, "transformers", preprocess_info


def load_frozen_image_teacher(
    teacher_cfg: Dict,
    device: torch.device,
) -> Tuple[FrozenImageTeacher, Dict]:
    cfg = dict(teacher_cfg or {})
    teacher_type = str(cfg.get("type", "clip_image_encoder")).lower()
    input_mode = str(cfg.get("input_mode", "grayscale_replicated_rgb"))
    if not bool(cfg.get("freeze", True)):
        raise ValueError("Foundation-distillation image teachers must remain frozen.")
    if input_mode != "grayscale_replicated_rgb":
        raise ValueError("Teacher input_mode must be grayscale_replicated_rgb.")

    if teacher_type == "clip_image_encoder":
        model_name = str(cfg.get("model_name", "ViT-B/32"))
        model, backend = _load_clip_backend(model_name, device, cfg.get("backend", "auto"))
        teacher = FrozenCLIPImageTeacher(model, backend=backend).to(device)
        preprocess_info = {
            "teacher_image_size": 224,
            "teacher_image_mean": [0.48145466, 0.4578275, 0.40821073],
            "teacher_image_std": [0.26862954, 0.26130258, 0.27577711],
        }
        feature_type = "image_embedding"
    elif teacher_type == "dinov2_image_encoder":
        backend_requested = str(cfg.get("backend", "transformers")).lower()
        if backend_requested != "transformers":
            raise ValueError("DINOv2 teacher.backend must be transformers.")
        model_name = str(cfg.get("model_name", "facebook/dinov2-small"))
        feature_type = str(cfg.get("feature_type", "cls")).lower()
        model, backend, preprocess_info = _load_dinov2_backend(model_name, device)
        teacher = FrozenDINOv2ImageTeacher(
            model,
            backend=backend,
            feature_type=feature_type,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown teacher.type={teacher_type!r}; expected clip_image_encoder or dinov2_image_encoder."
        )

    teacher.eval()
    info = {
        "teacher_type": teacher_type,
        "teacher_backend": backend,
        "teacher_model_name": model_name,
        "feature_type": feature_type,
        "input_mode": input_mode,
        "teacher_text_encoder_used": False,
        **preprocess_info,
    }
    return teacher, info


def load_clip_image_encoder(model_name: str, device: torch.device) -> Tuple[FrozenImageTeacher, str]:
    """Backward-compatible CLIP loader returning ``(teacher, backend)``."""
    teacher, info = load_frozen_image_teacher(
        {
            "type": "clip_image_encoder",
            "model_name": model_name,
            "input_mode": "grayscale_replicated_rgb",
            "freeze": True,
        },
        device,
    )
    return teacher, str(info["teacher_backend"])
