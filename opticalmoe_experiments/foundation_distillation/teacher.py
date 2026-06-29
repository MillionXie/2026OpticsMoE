from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenImageEncoder(nn.Module):
    def __init__(self, model: nn.Module, backend: str) -> None:
        super().__init__()
        self.model = model.eval()
        self.backend = backend
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(images)
        return F.normalize(features.float(), dim=-1)


def load_clip_image_encoder(model_name: str, device: torch.device) -> Tuple[FrozenImageEncoder, str]:
    """Load only a CLIP image encoder through an optional backend."""
    errors = []
    try:
        import open_clip

        open_clip_name = str(model_name).replace("/", "-")
        model = open_clip.create_model(open_clip_name, pretrained="openai", device=device)
        encoder = FrozenImageEncoder(model, backend="open_clip").to(device)
        return encoder, "open_clip"
    except Exception as exc:
        errors.append(f"open_clip: {exc}")
    try:
        import clip

        model, _unused_preprocess = clip.load(str(model_name), device=device, jit=False)
        encoder = FrozenImageEncoder(model, backend="clip").to(device)
        return encoder, "clip"
    except Exception as exc:
        errors.append(f"clip: {exc}")
    details = "\n".join(errors)
    raise RuntimeError(
        "A CLIP image-encoder backend is required only when building the teacher cache. "
        "Install one with `pip install open_clip_torch` (recommended) or install OpenAI CLIP.\n"
        f"Backend errors:\n{details}"
    )

