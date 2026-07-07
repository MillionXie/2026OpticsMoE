from __future__ import annotations

import os
from pathlib import Path


def download_checkpoint(
    model_id: str,
    cache_dir: Path | None,
    max_workers: int = 2,
    disable_xet: bool = True,
) -> Path:
    """Download a checkpoint without constructing the model or requiring CUDA."""

    if disable_xet:
        # This must be set before huggingface_hub is imported.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        max_workers=max_workers,
    )
    return Path(snapshot)

