"""Open held-out EuroSAT once for a validation-selected stage-A checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from .model import DirectCCDOptics
from .train_eurosat import evaluate, load_split, save_json, sha256_file, source_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--eval-batch", type=int, default=16)
    args = parser.parse_args()
    output = args.run / "selected_test.json"
    if output.exists():
        raise FileExistsError("selected EuroSAT checkpoint has already been tested")
    config = json.loads((args.run / "config.json").read_text())
    status = json.loads((args.run / "status.json").read_text())
    if status.get("status") != "complete" or not config.get("skip_test"):
        raise ValueError("test needs a complete validation-only run")
    trainval, holdout = source_paths(args.protocol)
    if config["source_sha256"] != {"trainval": sha256_file(trainval),
                                   "holdout": sha256_file(holdout)}:
        raise ValueError("EuroSAT source differs from training")
    if config.get("vision_checkpoint_sha256") != (sha256_file(args.vision_checkpoint)
                                                    if args.vision_checkpoint else None):
        raise ValueError("frozen visual front differs from training")
    checkpoint_path = args.run / "best_checkpoint.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["config"] != config:
        raise ValueError("selected checkpoint config differs from run")
    model = DirectCCDOptics(config["architecture"], activation_order="center_out").to(device)
    model.configure_stage(0)
    model.load_state_dict(checkpoint["model"])
    test = load_split(trainval, holdout, "test", args.vision_checkpoint, device)
    result = {"selected_epoch": checkpoint["epoch"],
              "validation": checkpoint["validation"],
              "checkpoint_sha256": sha256_file(checkpoint_path),
              "vision_checkpoint_sha256": config["vision_checkpoint_sha256"],
              "test": evaluate(model, test, device, args.eval_batch)}
    save_json(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
