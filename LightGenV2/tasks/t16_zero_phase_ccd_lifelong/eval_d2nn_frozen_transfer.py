"""Frozen D2NN cross-task inference; no phase or readout adaptation."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from .model import DirectCCDOptics
from .train_eurosat import save_json, sha256_file, source_paths
from .train_lifelong_moe import load_dataset, validate_head
from .train_other_tasks import evaluate


TASKS = ("eurosat", "speech_binary")
CLASSES = {"eurosat": 10, "speech_binary": 2}


def check_source(checkpoint, task, protocol):
    config = checkpoint.get("config", {})
    if config.get("architecture") != "d2nn":
        raise ValueError("source checkpoint is not a D2NN")
    if task == "eurosat":
        trainval, holdout = source_paths(protocol)
        expected = {"trainval": sha256_file(trainval),
                    "holdout": sha256_file(holdout)}
        if config.get("task") != "eurosat_paired_rgb_sar" or config.get("source_sha256") != expected:
            raise ValueError("EuroSAT source checkpoint or data differs")
    elif config.get("task") != "speech_binary" or config.get("source_protocol_sha256") != sha256_file(protocol):
        raise ValueError("Speech source checkpoint or data differs")
    weights = checkpoint["model"]
    if "shared_head.weight" not in weights or tuple(weights["shared_head.weight"].shape) != (10, 784):
        raise ValueError("expected a single ten-output readout")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--eurosat-checkpoint", type=Path, required=True)
    parser.add_argument("--speech-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.batch <= 0 or ("runs", "simulation") not in list(zip(
            args.out.parts, args.out.parts[1:])):
        raise ValueError("expected positive batch and a runs/simulation output")
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "status.json", {"status": "running"})
    protocols = {"eurosat": args.eurosat, "speech_binary": args.speech}
    checkpoints = {"eurosat": args.eurosat_checkpoint,
                   "speech_binary": args.speech_checkpoint}
    config = {
        "rows": TASKS, "columns": TASKS,
        "meaning": "source D2NN optical phases and its own Linear fixed; target direct test inference",
        "adaptation": "none", "batch": args.batch,
        "protocol_sha256": {name: sha256_file(path) for name, path in protocols.items()},
        "checkpoint_sha256": {name: sha256_file(path) for name, path in checkpoints.items()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda},
    }
    save_json(args.out / "config.json", config)
    (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {name: load_dataset(protocols, name, "test") for name in TASKS}
    matrix = {}
    for source in TASKS:
        checkpoint = torch.load(checkpoints[source], map_location=device, weights_only=False)
        check_source(checkpoint, source, protocols[source])
        model = DirectCCDOptics("d2nn", activation_order="center_out").to(device)
        validate_head(model)
        model.load_state_dict(checkpoint["model"])
        matrix[source] = {}
        for target in TASKS:
            matrix[source][target] = evaluate(model, datasets[target], device,
                                               args.batch, CLASSES[target])
        save_json(args.out / "partial_result.json", matrix)
        print(json.dumps({"source": source,
                          "balanced_accuracy": {t: matrix[source][t]["balanced_accuracy"]
                                                for t in TASKS}}), flush=True)
    save_json(args.out / "result.json", {"matrix": matrix, "config": config})
    save_json(args.out / "status.json", {"status": "complete", "cells": 4})


if __name__ == "__main__":
    main()
