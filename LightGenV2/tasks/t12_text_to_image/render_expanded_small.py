"""Render a balanced three-category, three-mode grid from a saved small editor."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset

from .qwen_mini_small import PromptEmbeddingLookup, QwenMiniConfig, QwenMiniTextEncoder, save_samples
from .small_fullframe import SmallEditorConfig, SmallFullFrameEditor, build_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "data-dir", "instruction-cache", "embedding-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--gallery", choices=("balanced", "four-designs"), default="balanced")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_values = dict(payload["editor_config"])
    config_values["widths"] = tuple(config_values["widths"])
    config = SmallEditorConfig(**config_values)
    model = SmallFullFrameEditor(config)
    model.text = QwenMiniTextEncoder(QwenMiniConfig(**payload["qwen_mini_config"]))
    model.load_state_dict(payload["model"])
    device = torch.device(args.device)
    model = model.to(device).eval()
    dataset = build_dataset("unified_expanded", args.data_dir, args.split,
                            config.image_size, args.instruction_cache)
    if args.gallery == "four-designs":
        first = {}
        for source_index, source in enumerate(dataset.base.sources):
            first.setdefault(source["category"], source_index)
        indices = [first[category] * dataset.base.targets_per_source + offset
                   for category in dataset.base.supported_categories
                   for offset in range(4, 8)]
        dataset = Subset(dataset, indices)
    lookup = PromptEmbeddingLookup(args.embedding_cache)
    save_samples(model, dataset, lookup, args.output, device, 2026)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
