"""Read-only training-split audit for the shared frozen visual input."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .train_eurosat import load_split, source_paths, sha256_file
from .train_other_tasks import load_task


def check_power(field):
    if field.shape != (2, 224, 224) or not torch.isfinite(field).all():
        raise AssertionError("invalid incident field")
    power = field.square().sum((1, 2))
    if not torch.allclose(power, torch.ones_like(power), atol=2e-4):
        raise AssertionError(f"incident power changed: {power.tolist()}")
    return power.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--clevr", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainval, holdout = source_paths(args.eurosat)
    euro = load_split(trainval, holdout, "train", args.vision_checkpoint, device)
    clevr = load_task(args.clevr, "clevr", "train", args.vision_checkpoint, device)
    efield = euro.get_batch(np.array([0, 1]), device)
    cfield = clevr.get_batch(np.array([0, 1]), device)
    old_euro = euro.base[np.array([0, 1])].to(device)
    old_clevr = clevr.base.get_batch(np.array([0, 1]), device)
    if not torch.equal(efield[:, 112:, 112:], old_euro[:, 112:, 112:]):
        raise AssertionError("paired SAR tile changed")
    if not torch.equal(cfield[:, 112:, 112:], old_clevr[:, 112:, 112:]):
        raise AssertionError("original query tile changed")
    if not torch.equal(cfield[0, :112, :112], cfield[1, :112, :112]):
        raise AssertionError("the same CLEVR image has different visual encoding")
    if torch.equal(cfield[0, 112:, 112:], cfield[1, 112:, 112:]):
        raise AssertionError("positive and negative original queries are identical")
    print(json.dumps({"status": "pass", "device": str(device),
                      "vision_sha256": sha256_file(args.vision_checkpoint),
                      "train_records": {"eurosat": len(euro), "clevr": len(clevr)},
                      "incident_power": {"eurosat": check_power(efield),
                                          "clevr": check_power(cfield)},
                      "sar_unchanged": True, "original_question_unchanged": True,
                      "same_image_same_visual_field": True}), flush=True)


if __name__ == "__main__":
    main()
