from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import queue
from .phase_only import load_phase_only_settings


PHASE_ONLY_MODULE = (
    "experiments."
    "qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.phase_only"
)


def configure_queue() -> None:
    queue.MODULE = PHASE_ONLY_MODULE
    queue.resolved_settings = lambda config, key: load_phase_only_settings(
        config,
        task=key.task,
        method=key.method,
        seed=key.seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    configure_queue()
    return queue.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
