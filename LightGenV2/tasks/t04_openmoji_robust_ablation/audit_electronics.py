"""Count parameters in the actual OpenMoji checkpoint by electronic module."""

from collections import Counter
from pathlib import Path
import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    groups = Counter()
    detail = Counter()
    for name, tensor in state.items():
        if name.startswith("shared_readout."):
            groups[name.split(".")[1]] += tensor.numel()
            detail[".".join(name.split(".")[:3])] += tensor.numel()
    print(json.dumps({"checkpoint": str(args.checkpoint), "shared_readout": dict(groups),
                      "detail": dict(detail), "total": sum(groups.values())}, indent=2))


if __name__ == "__main__":
    main()
