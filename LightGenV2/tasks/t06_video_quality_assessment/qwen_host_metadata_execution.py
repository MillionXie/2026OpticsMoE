"""Exact native Qwen execution with integer host metadata, not feature caching.

All learned operators and their original precision/attention implementation stay
in the installed native forward. Integer position preparation is recomputed at
its original point inside every forward; no model work is moved before timing.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import torch

OFFICIAL_SOURCE_SHA256 = "685b10cbc48556343df4e159199c5884d02db9ad227640f0f4fbe75ddf0b2a99"
METADATA_KEYS = ("input_ids", "attention_mask", "video_grid_thw")


class HostMetadataQwenModel(torch.nn.Module):
    """Compose the original model without changing its class, weights or source.

    The official forward is invoked with this delegating module as `self`. Its
    child modules are the SAME objects. Only get_rope_index changes the location
    of integer operations; video grids/attention split lengths remain on CPU.
    No learned embedding, rotary values, hidden states or logits are cached.
    """

    def __init__(self, wrapped, cpu_inputs, resident_inputs):
        super().__init__()
        module_path = Path(inspect.getfile(type(wrapped)))
        if hashlib.sha256(module_path.read_bytes()).hexdigest() != OFFICIAL_SOURCE_SHA256:
            raise RuntimeError("Unreviewed native Qwen source; fail closed")
        if type(wrapped).__name__ != "Qwen3VLModel":
            raise TypeError("Only the audited native Qwen3VLModel is supported")
        if wrapped.training or any(p.requires_grad for p in wrapped.parameters()):
            raise RuntimeError("Frozen evaluation model required")
        if wrapped.config._attn_implementation != "sdpa":
            raise RuntimeError("Preserve original SDPA implementation")
        if set(cpu_inputs) != {"input_ids", "attention_mask", "video_grid_thw", "pixel_values_videos"}:
            raise ValueError("Only the audited video-only input contract is supported")
        if any(cpu_inputs[k].device.type != "cpu" for k in METADATA_KEYS):
            raise ValueError("Host metadata must come from original CPU preparation")
        if any(cpu_inputs[k].dtype.is_floating_point for k in METADATA_KEYS):
            raise ValueError("Only integer metadata may move to CPU")
        if cpu_inputs["video_grid_thw"].ndim != 2 or cpu_inputs["video_grid_thw"].shape[1] != 3:
            raise ValueError("Invalid grid shape")
        self.wrapped = wrapped
        self.host_metadata = {k: cpu_inputs[k].detach().clone() for k in METADATA_KEYS}
        self.resident_inputs = resident_inputs
        self.rope_deltas = None
        self.training = False
        if {id(p) for p in self.parameters()} != {id(p) for p in wrapped.parameters()}:
            raise RuntimeError("Adapter must not create or duplicate parameters")

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("wrapped"), name)

    def get_rope_index(self, input_ids=None, image_grid_thw=None,
                       video_grid_thw=None, attention_mask=None):
        if image_grid_thw is not None or input_ids is not self.resident_inputs["input_ids"]:
            raise ValueError("Unsupported or changed input")
        if video_grid_thw is not self.host_metadata["video_grid_thw"]:
            raise ValueError("Unexpected grid metadata")
        positions, deltas = self.wrapped.get_rope_index(
            self.host_metadata["input_ids"], None,
            self.host_metadata["video_grid_thw"],
            attention_mask=self.host_metadata["attention_mask"],
        )
        # This method runs at the original native point, AFTER vision forward.
        # Transfers stay inside forward/measurement. No setup-only position cache.
        return positions.to(input_ids.device), deltas.to(input_ids.device)

    def forward(self, **kwargs):
        if kwargs.get("use_cache") is not False or kwargs.get("past_key_values") is not None:
            raise ValueError("Only the original full prefill, no KV cache")
        if kwargs.get("position_ids") is not None:
            raise ValueError("Do not cache or supply setup-only position IDs")
        for key, value in self.resident_inputs.items():
            if kwargs.get(key) is not value:
                raise ValueError("Input identity changed; rebuild metadata adapter")
        inputs = dict(kwargs, video_grid_thw=self.host_metadata["video_grid_thw"])
        # Call the ORIGINAL method, not a copied/reimplemented model forward.
        return type(self.wrapped).forward(self, **inputs)


def validate_metadata_unchanged(adapter, original_cpu_inputs):
    return all(torch.equal(adapter.host_metadata[k], original_cpu_inputs[k]) for k in METADATA_KEYS)
