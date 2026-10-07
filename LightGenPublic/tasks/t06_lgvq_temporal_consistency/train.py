"""Standalone training/evaluation entry for LGVQ Temporal 0.8044."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from LightGenV2.tasks.t06_video_quality_assessment.multivideo import main


if __name__ == "__main__":
    raise SystemExit(main())
