"""One-time held-out evaluation for a validation-selected non-EuroSAT run."""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from .model import DirectCCDOptics
from .train_eurosat import save_json, sha256_file
from .train_other_tasks import CLASSES, evaluate, load_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--eval-batch", type=int, default=16)
    args = parser.parse_args()
    output = args.run / "selected_test.json"
    if output.exists():
        raise FileExistsError("selected task checkpoint has already been tested")
    config = json.loads((args.run / "config.json").read_text())
    status = json.loads((args.run / "status.json").read_text())
    if status.get("status") != "complete" or config.get("test_policy") != "omitted":
        raise ValueError("test needs a complete validation-only run")
    if config["source_protocol_sha256"] != sha256_file(args.protocol):
        raise ValueError("source protocol differs from training")
    if config.get("vision_checkpoint_sha256") != (sha256_file(args.vision_checkpoint)
                                                    if args.vision_checkpoint else None):
        raise ValueError("frozen visual front differs from training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.run / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config"] != config:
        raise ValueError("selected checkpoint config differs from run")
    model = DirectCCDOptics(config["architecture"], activation_order="center_out").to(device)
    if config["architecture"] == "moe":
        model.configure_stage(config["moe_active_experts"] // 4 - 1)
    else:
        model.configure_stage(0)
    model.load_state_dict(checkpoint["model"])
    task = config["task"]
    test = load_task(args.protocol, task, "test", args.vision_checkpoint, device)
    result = {"task": task, "selected_epoch": checkpoint["epoch"],
              "validation": checkpoint["validation"],
              "checkpoint_sha256": sha256_file(checkpoint_path),
              "vision_checkpoint_sha256": config["vision_checkpoint_sha256"],
              "evaluation_git_commit": subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], text=True).strip(),
              "test": evaluate(model, test, device, args.eval_batch, CLASSES[task])}
    save_json(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
