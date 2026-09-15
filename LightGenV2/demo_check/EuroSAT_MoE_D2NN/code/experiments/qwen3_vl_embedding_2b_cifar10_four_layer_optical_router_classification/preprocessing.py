from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    IGNORED_MODEL_INPUTS,
    apply_embedding_template,
)


_REQUIRED_INPUTS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
_INSTANCES: dict[tuple[int, str], "CachedCIFAR10QwenPreprocessor"] = {}


def _tensor_inputs(values: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in values.items()
        if torch.is_tensor(tensor) and name not in IGNORED_MODEL_INPUTS
    }


def legacy_preprocess_images(
    processor: Any,
    images: Sequence[Any],
    instruction: str,
) -> dict[str, torch.Tensor]:
    """Reference implementation used when a processor cannot be safely cached."""

    texts = [apply_embedding_template(processor, image, instruction) for image in images]
    values = _tensor_inputs(
        processor(
            text=texts,
            images=list(images),
            padding=True,
            return_tensors="pt",
        )
    )
    missing = [name for name in _REQUIRED_INPUTS if name not in values]
    if missing:
        raise RuntimeError(f"Qwen3-VL processor did not return: {missing}")
    return values


class CachedCIFAR10QwenPreprocessor:
    """Cache the invariant text half of CIFAR-10 Qwen preprocessing.

    All CIFAR-10 images enter the Qwen processor with one fixed pixel budget, so
    their ``image_grid_thw`` values and the rendered classification prompt are
    invariant.  The first batch is intentionally processed through the legacy
    path and checked against an image-only processor call.  Caching is enabled
    only when those runtime checks prove that the processor output decomposes
    exactly into invariant text tensors plus image tensors.  Otherwise every
    call falls back to the legacy implementation.
    """

    def __init__(self, processor: Any, instruction: str) -> None:
        self.processor = processor
        self.instruction = str(instruction)
        self._static_template: str | None = None
        self._text_rows: dict[str, torch.Tensor] | None = None
        self._grid: tuple[int, ...] | None = None
        self._vision_keys: frozenset[str] | None = None
        self._cache_disabled = False
        self._announced = False

    @property
    def cache_enabled(self) -> bool:
        return self._text_rows is not None and not self._cache_disabled

    def _render_all(self, images: Sequence[Any]) -> list[str]:
        if self._static_template is not None:
            return [self._static_template] * len(images)
        texts = [
            apply_embedding_template(self.processor, image, self.instruction)
            for image in images
        ]
        if texts and all(text == texts[0] for text in texts[1:]):
            self._static_template = texts[0]
        return texts

    def _full(self, images: Sequence[Any]) -> dict[str, torch.Tensor]:
        texts = self._render_all(images)
        values = _tensor_inputs(
            self.processor(
                text=texts,
                images=list(images),
                padding=True,
                return_tensors="pt",
            )
        )
        missing = [name for name in _REQUIRED_INPUTS if name not in values]
        if missing:
            raise RuntimeError(f"Qwen3-VL processor did not return: {missing}")
        return values

    def _images_only(self, images: Sequence[Any]) -> dict[str, torch.Tensor]:
        image_processor = getattr(self.processor, "image_processor", None)
        if callable(image_processor):
            # Bypass ProcessorMixin's construction of one long replacement
            # string per image.  The first-batch tensor check below proves that
            # this direct component call is identical for the installed Qwen
            # processor before its result is ever used by the model.
            return _tensor_inputs(
                image_processor(
                    images=list(images),
                    return_tensors="pt",
                )
            )
        return _tensor_inputs(
            self.processor(
                images=list(images),
                padding=True,
                return_tensors="pt",
            )
        )

    @staticmethod
    def _identical_rows(tensor: torch.Tensor, batch_size: int) -> bool:
        if tensor.ndim == 0 or tensor.shape[0] != batch_size:
            return False
        if batch_size <= 1:
            return True
        return bool(torch.equal(tensor, tensor[0:1].expand_as(tensor)))

    def _try_warm_cache(
        self,
        images: Sequence[Any],
        full: Mapping[str, torch.Tensor],
    ) -> None:
        if self._static_template is None or not images:
            self._cache_disabled = True
            return
        try:
            vision = self._images_only(images)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self._cache_disabled = True
            return
        if "image_grid_thw" not in vision or "pixel_values" not in vision:
            self._cache_disabled = True
            return
        if any(
            name not in full or not torch.equal(tensor, full[name])
            for name, tensor in vision.items()
        ):
            self._cache_disabled = True
            return

        grids = vision["image_grid_thw"]
        batch_size = len(images)
        if grids.ndim != 2 or grids.shape[0] != batch_size:
            self._cache_disabled = True
            return
        if not self._identical_rows(grids, batch_size):
            self._cache_disabled = True
            return

        vision_keys = frozenset(vision)
        text = {name: tensor for name, tensor in full.items() if name not in vision_keys}
        if not text or any(
            not self._identical_rows(tensor, batch_size) for tensor in text.values()
        ):
            self._cache_disabled = True
            return

        self._grid = tuple(int(value) for value in grids[0].tolist())
        self._vision_keys = vision_keys
        self._text_rows = {
            name: tensor[0:1].clone() for name, tensor in text.items()
        }

    def _cached(self, images: Sequence[Any]) -> dict[str, torch.Tensor] | None:
        if not self.cache_enabled:
            return None
        try:
            vision = self._images_only(images)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return None
        if frozenset(vision) != self._vision_keys:
            return None
        grids = vision.get("image_grid_thw")
        if grids is None or grids.ndim != 2 or grids.shape[0] != len(images):
            return None
        expected = self._grid
        if expected is None or any(
            tuple(int(value) for value in row.tolist()) != expected for row in grids
        ):
            return None
        output = dict(vision)
        assert self._text_rows is not None
        for name, row in self._text_rows.items():
            output[name] = row.expand(len(images), *row.shape[1:]).clone()
        return output

    def __call__(self, images: Sequence[Any]) -> dict[str, torch.Tensor]:
        if not images:
            raise ValueError("CIFAR-10 preprocessing requires a non-empty image batch")
        cached = self._cached(images)
        if cached is not None:
            return cached
        full = self._full(images)
        if self._text_rows is None and not self._cache_disabled:
            self._try_warm_cache(images, full)
            if self.cache_enabled and not self._announced:
                print(
                    "[preprocess] exact static prompt/token cache enabled after "
                    "runtime tensor-equivalence checks",
                    flush=True,
                )
                self._announced = True
        return full


def cached_preprocess_images(
    processor: Any,
    images: Sequence[Any],
    instruction: str,
) -> dict[str, torch.Tensor]:
    """Drop-in replacement for the shared ``preprocess_images`` function."""

    key = (id(processor), str(instruction))
    cached = _INSTANCES.get(key)
    if cached is None or cached.processor is not processor:
        cached = CachedCIFAR10QwenPreprocessor(processor, instruction)
        _INSTANCES[key] = cached
    return cached(images)


__all__ = [
    "CachedCIFAR10QwenPreprocessor",
    "cached_preprocess_images",
    "legacy_preprocess_images",
]
