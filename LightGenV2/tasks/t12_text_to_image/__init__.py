"""T12: single-pass, text-conditioned object image generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .modeling import TextConditionedVAE


def __getattr__(name: str) -> Any:
    """Avoid importing Torch when callers only need configuration modules."""

    if name in {"TextConditionedVAE", "build_model"}:
        from .modeling import TextConditionedVAE, build_model

        return {"TextConditionedVAE": TextConditionedVAE, "build_model": build_model}[name]
    raise AttributeError(name)

__all__ = ["TextConditionedVAE", "build_model"]
