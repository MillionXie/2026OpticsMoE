from __future__ import annotations

import argparse
from pathlib import Path

from experiments.qwen3_vl_patch_stem_8stage_optical_imagenet_backbone.train import (
    Context,
    load_config,
    run,
)

from .model import QwenStemSeparableOpticalImageNetBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train P11 separable token/channel-axis optical ImageNet backbone"
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
            model_class=QwenStemSeparableOpticalImageNetBackbone,
            experiment_name=(
                "P11 Qwen static stem + four separable token/channel-axis "
                "optical macro blocks + width-96 electronic mixers"
            ),
        )
    finally:
        context.close()


if __name__ == "__main__":
    main()
