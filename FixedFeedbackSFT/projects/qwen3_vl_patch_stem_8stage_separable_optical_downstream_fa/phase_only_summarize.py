from __future__ import annotations

from typing import Sequence

from . import summarize
from .phase_only import load_phase_only_settings


def main(argv: Sequence[str] | None = None) -> int:
    original = summarize.load_settings
    summarize.load_settings = load_phase_only_settings
    try:
        return summarize.main(argv)
    finally:
        summarize.load_settings = original


if __name__ == "__main__":
    raise SystemExit(main())
