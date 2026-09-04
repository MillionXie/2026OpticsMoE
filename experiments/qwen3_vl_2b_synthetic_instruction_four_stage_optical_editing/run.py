from __future__ import annotations

import argparse
import json

import torch

from .prompt_cache import build_prompt_cache
from .scenes import prepare_dataset
from .settings import load_settings
from .training import test, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt-conditioned Language2+Vision2 optical synthetic editor"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("prepare_data", "cache_prompts", "train", "test", "all"),
    )
    parser.add_argument("--device", default=None, help="cuda, cuda:0, cpu; defaults to CUDA when available")
    return parser.parse_args()


def _device(value: str | None) -> torch.device:
    if value is not None:
        device = torch.device(value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    device = _device(args.device)
    result = None
    if args.phase in {"prepare_data", "all"}:
        result = prepare_dataset(settings)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.phase in {"cache_prompts", "all"}:
        result = build_prompt_cache(settings, device)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.phase in {"train", "all"}:
        result = train(settings, device)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.phase in {"test", "all"}:
        result = test(settings, device)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
