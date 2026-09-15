"""Sealed-test Caltech101 four-layer hybrid with a 5% fusion floor.

The package entry point is deliberately lazy.  This keeps the laboratory
``offline_quick_finetune`` utility independent of Qwen and Transformers while
preserving ``python -m <package>`` for full server training.
"""


def main() -> int:
    from .run import main as run_main

    return run_main()


__all__ = ["main"]
