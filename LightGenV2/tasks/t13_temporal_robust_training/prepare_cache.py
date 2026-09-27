"""Rebuild 4-frame 49-token front from local frozen Qwen and original LGVQ."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime"))
from cache_builder.cache_qwen_front import main

if __name__ == "__main__":
    raise SystemExit(main())
