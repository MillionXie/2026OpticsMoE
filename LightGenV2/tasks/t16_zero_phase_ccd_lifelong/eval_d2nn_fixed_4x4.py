"""Direct 4x4 transfer of four independently trained frozen D2NN systems.

Rows retain their own optical phases AND their own single Linear head. Columns
are target datasets preprocessed by the one shared, fixed protocol. No target
adaptation, gradients, head fitting, or per-column checkpoint selection occur.
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from .model import DirectCCDOptics
from .train_eurosat import save_json, sha256_file, source_paths
from .train_lifelong_moe import load_dataset, score_all, validate_head


TASKS = ("eurosat", "clevr", "speech_binary", "physical_binary_raw")
SOURCE_TASK = {"eurosat": "eurosat_paired_rgb_sar", "clevr": "clevr",
               "speech_binary": "speech_binary",
               "physical_binary_raw": "physical_binary_raw"}


def check_checkpoint(checkpoint, name, protocol, vision_sha):
    config = checkpoint.get("config", {})
    if (config.get("task") != SOURCE_TASK[name] or
            config.get("architecture") != "d2nn" or
            config.get("activation_order") != "center_out"):
        raise ValueError(f"{name}: wrong independent D2NN source")
    if name == "eurosat":
        trainval, holdout = source_paths(protocol)
        expected = {"trainval": sha256_file(trainval), "holdout": sha256_file(holdout)}
        if config.get("source_sha256") != expected:
            raise ValueError("EuroSAT source data changed")
    elif config.get("source_protocol_sha256") != sha256_file(protocol):
        raise ValueError(f"{name}: source data changed")
    expected_vision = vision_sha if name in ("eurosat", "clevr") else None
    if config.get("vision_checkpoint_sha256") != expected_vision:
        raise ValueError(f"{name}: visual front differs from fixed target protocol")
    if ("shared_head.weight" not in checkpoint["model"] or
            tuple(checkpoint["model"]["shared_head.weight"].shape) != (10, 784)):
        raise ValueError(f"{name}: expected one ten-output Linear head")


def main():
    parser = argparse.ArgumentParser()
    for name in TASKS:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
        parser.add_argument("--" + name.replace("_", "-") + "-checkpoint",
                            type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if (args.batch <= 0 or
            ("runs", "simulation") not in list(zip(args.out.parts, args.out.parts[1:]))):
        raise ValueError("matrix must use a positive batch and runs/simulation")
    protocols = {name: getattr(args, name) for name in TASKS}
    paths = {name: getattr(args, name + "_checkpoint") for name in TASKS}
    config = {
        "rows": TASKS, "columns": TASKS,
        "meaning": "independent source D2NN optical phases and its own Linear; direct target inference",
        "target_adaptation": "none",
        "vision_checkpoint_sha256": sha256_file(args.vision_checkpoint),
        "source_protocol_sha256": {name: sha256_file(path) for name, path in protocols.items()},
        "source_checkpoint_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "batch": args.batch, "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    args.out.mkdir(parents=True, exist_ok=False)
    save_json(args.out / "config.json", config)
    (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    save_json(args.out / "status.json", {"status": "running", "cells": 0})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {name: load_dataset(protocols, name, "test", args.vision_checkpoint,
                                   device) for name in TASKS}
    matrix = {}
    for name in TASKS:
        checkpoint = torch.load(paths[name], map_location=device, weights_only=False)
        check_checkpoint(checkpoint, name, protocols[name],
                         config["vision_checkpoint_sha256"])
        model = DirectCCDOptics("d2nn", activation_order="center_out").to(device)
        model.configure_stage(0)
        validate_head(model)
        model.load_state_dict(checkpoint["model"])
        matrix[name] = score_all(model, datasets, TASKS, device, args.batch)
        save_json(args.out / "partial_result.json", matrix)
        save_json(args.out / "status.json", {"status": "running",
                                               "cells": len(matrix) * len(TASKS)})
        print(json.dumps({"source": name,
                          "balanced_accuracy": {target: row["balanced_accuracy"]
                                                for target, row in matrix[name].items()}}),
              flush=True)
    diagonal_checks = {}
    for name in TASKS:
        selected_path = paths[name].parent / "selected_test.json"
        if selected_path.exists():
            previous = json.loads(selected_path.read_text())
            new = matrix[name][name]["balanced_accuracy"]
            old = previous["test"]["balanced_accuracy"]
            if abs(new - old) > 1e-10:
                raise AssertionError(f"{name} matrix diagonal differs from one-time test")
            diagonal_checks[name] = True
    save_json(args.out / "result.json", {"matrix": matrix, "config": config,
                                          "diagonal_checks": diagonal_checks})
    save_json(args.out / "status.json", {"status": "complete", "cells": 16})


if __name__ == "__main__":
    main()
