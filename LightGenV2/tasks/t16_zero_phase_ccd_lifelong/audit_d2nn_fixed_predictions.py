"""Audit ten-output predictions behind the frozen D2NN transfer matrix.

This is read-only inference with the original checkpoints and test protocol.
It does not mask logits, fit a head, or change the published matrix scores.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .eval_d2nn_fixed_4x4 import TASKS, check_checkpoint
from .model import DirectCCDOptics
from .train_eurosat import save_json, sha256_file
from .train_lifelong_moe import load_dataset, validate_head
from .train_other_tasks import CLASSES, metrics


def main():
    parser = argparse.ArgumentParser()
    for name in TASKS:
        flag = name.replace("_", "-")
        parser.add_argument("--" + flag, type=Path, required=True)
        parser.add_argument("--" + flag + "-checkpoint", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--matrix-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.batch <= 0 or args.out.exists():
        raise ValueError("positive batch and new output path required")
    original = json.loads((args.matrix_run / "result.json").read_text())["matrix"]
    protocols = {name: getattr(args, name) for name in TASKS}
    paths = {name: getattr(args, name + "_checkpoint") for name in TASKS}
    vision_sha = sha256_file(args.vision_checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {name: load_dataset(protocols, name, "test", args.vision_checkpoint,
                                   device) for name in TASKS}
    args.out.mkdir(parents=True)
    save_json(args.out / "config.json", {
        "source_matrix": str(args.matrix_run),
        "source_checkpoint_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "source_protocol_sha256": {name: sha256_file(path) for name, path in protocols.items()},
        "vision_checkpoint_sha256": vision_sha,
        "batch": args.batch,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inference": "unchanged ten-output argmax; no target adaptation",
    })
    audit = {}
    for source in TASKS:
        checkpoint = torch.load(paths[source], map_location=device, weights_only=False)
        check_checkpoint(checkpoint, source, protocols[source], vision_sha)
        model = DirectCCDOptics("d2nn", activation_order="center_out").to(device)
        model.configure_stage(0)
        validate_head(model)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        audit[source] = {}
        for target in TASKS:
            data = datasets[target]
            labels = np.asarray(data.labels)
            predictions = []
            with torch.inference_mode():
                for start in range(0, len(data), args.batch):
                    indices = np.arange(start, min(start + args.batch, len(data)))
                    field = data.get_batch(indices, device)
                    predictions.append(model(field)["logits"].argmax(1).cpu().numpy())
            predicted = np.concatenate(predictions)
            classes = 10 if target == "eurosat" else CLASSES[target]
            measured = metrics(labels, predicted, classes)
            reference = original[source][target]
            for key in ("accuracy", "balanced_accuracy"):
                if abs(measured[key] - reference[key]) > 1e-10:
                    raise AssertionError(f"{source}->{target} {key} changed")
            cell = {
                "accuracy": measured["accuracy"],
                "balanced_accuracy": measured["balanced_accuracy"],
                "per_class_recall": measured["per_class_recall"],
                "prediction_histogram_0_to_9": np.bincount(predicted, minlength=10).tolist(),
                "valid_target_label_fraction": float((predicted < classes).mean()),
                "n": int(len(data)),
            }
            audit[source][target] = cell
            print(json.dumps({"source": source, "target": target,
                              "valid_fraction": cell["valid_target_label_fraction"],
                              "histogram": cell["prediction_histogram_0_to_9"]}), flush=True)
        save_json(args.out / "partial_result.json", audit)
    save_json(args.out / "result.json", audit)


if __name__ == "__main__":
    main()
