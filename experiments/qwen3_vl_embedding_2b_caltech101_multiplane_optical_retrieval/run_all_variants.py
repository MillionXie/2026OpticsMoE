from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


VARIANTS = (
    "d2nn_continuous",
    "d2nn_oeo_sigmoid",
    "moe_continuous_fixed_router",
    "moe_oeo_dynamic_router",
    # Supplemental ablation, not one of the four primary comparisons.
    "moe_oeo_fixed_router",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run four primary comparisons plus fixed-router OEO supplement"
    )
    parser.add_argument("--phase", default="all", choices=("train", "evaluate", "all"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start-at", choices=VARIANTS, default=VARIANTS[0])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    started = False
    for variant in VARIANTS:
        if variant == args.start_at:
            started = True
        if not started:
            continue
        config = root / "configs" / ("smoke" if args.smoke else "release") / f"{variant}.yaml"
        command = [
            sys.executable,
            "-m",
            "experiments.qwen3_vl_embedding_2b_caltech101_multiplane_optical_retrieval",
            "--config",
            str(config),
            "--phase",
            args.phase,
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
