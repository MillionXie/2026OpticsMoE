from __future__ import annotations

from torch import nn


def locate_visual(model: nn.Module) -> nn.Module:
    candidates = (getattr(model, "visual", None), getattr(getattr(model, "model", None), "visual", None))
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "patch_embed") and hasattr(candidate, "blocks"):
            return candidate
    raise RuntimeError("Unable to locate Qwen3-VL visual patch embedding")

