from __future__ import annotations

import argparse
from pathlib import Path

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.train import (
    Context,
    load_config,
    run,
)

from .model import QwenStemSlimMixerOpticalImageNetBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train P09 Qwen-stem eight-stage width-96 slim-mixer optical backbone"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = Context()
    try:
        run(
            load_config(args.config),
            context,
            resume=args.resume,
            model_class=QwenStemSlimMixerOpticalImageNetBackbone,
            experiment_name=(
                "P09 Qwen static patch stem + eight-stage optical backbone + "
                "per-stage width-96 slim spatial token mixers"
            ),
        )
    finally:
        context.close()


if __name__ == "__main__":
    main()
