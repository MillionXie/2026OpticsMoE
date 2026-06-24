from pathlib import Path
from typing import Dict, Union

PathLike = Union[str, Path]

import torch


def save_checkpoint(path: PathLike, model, optimizer, epoch: int, metrics: Dict, config: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "config": config,
        },
        str(path),
    )


def load_checkpoint(path: PathLike, model, optimizer=None, map_location="cpu"):
    payload = torch.load(str(path), map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
