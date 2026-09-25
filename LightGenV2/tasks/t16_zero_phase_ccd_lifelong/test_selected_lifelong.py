"""Evaluate one validation-selected T16 lifelong checkpoint on held-out test once."""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch

from .model import DirectCCDOptics
from .train_eurosat import save_json, sha256_file
from .train_lifelong_moe import ORDER, load_dataset, score_all, validate_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--clevr", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--eval-batch", type=int, default=16)
    args = parser.parse_args()
    if args.eval_batch <= 0:
        raise ValueError("eval batch must be positive")
    output = args.run / "selected_test.json"
    if output.exists():
        raise FileExistsError("selected checkpoint has already been tested")
    config = json.loads((args.run / "config.json").read_text())
    status = json.loads((args.run / "status.json").read_text())
    if status.get("status") != "complete" or config.get("test_policy") != "omitted":
        raise ValueError("test needs a complete validation-only run")
    stage = config["stage"]
    order = tuple(config["order"])
    names = order[:stage]
    if (stage not in (2, 3, 4) or tuple(config["learned_tasks"]) != names or
            order[:3] != ORDER[:3] or order[3] != "physical_binary_raw"):
        raise ValueError("unexpected lifelong stage")
    protocols = dict(zip(order, (args.eurosat, args.clevr,
                                 args.speech, args.physical)))
    if config.get("vision_checkpoint_sha256") != (sha256_file(args.vision_checkpoint)
                                                    if args.vision_checkpoint else None):
        raise ValueError("frozen visual front differs from the training run")
    if config["protocol_sha256"] != {name: sha256_file(protocols[name])
                                      for name in names}:
        raise ValueError("test protocols differ from the training run")
    checkpoint_path = args.run / "best_checkpoint.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not (0 < checkpoint["epoch"] <= config["epochs"]):
        raise ValueError("invalid selected epoch")
    model = DirectCCDOptics(config["architecture"], activation_order="center_out").to(device)
    model.configure_stage(stage - 1)
    validate_head(model)
    model.load_state_dict(checkpoint["model"])
    test = {name: load_dataset(protocols, name, "test", args.vision_checkpoint,
                               device) for name in names}
    result = {
        "stage": stage, "learned_tasks": names,
        "selected_epoch": checkpoint["epoch"],
        "selected_validation": checkpoint["validation"],
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "test": score_all(model, test, names, device, args.eval_batch),
        "evaluated_by_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {"python": platform.python_version(),
                        "torch": torch.__version__, "cuda": torch.version.cuda},
    }
    save_json(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
