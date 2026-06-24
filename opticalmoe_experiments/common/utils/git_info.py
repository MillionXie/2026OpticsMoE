import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

PathLike = Union[str, Path]

import torch


def _run_git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def collect_git_info(repo_root: PathLike) -> Dict:
    root = Path(repo_root)
    return {
        "commit": _run_git(["rev-parse", "HEAD"], root),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(_run_git(["status", "--porcelain"], root)),
    }


def collect_environment() -> Dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cwd": os.getcwd(),
    }
