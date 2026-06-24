import argparse
import sys
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.data.datasets import create_dataloaders
from common.training.eval_loop import evaluate
from common.utils.config import load_yaml, save_json
from common.utils.seed import choose_device, set_seed
from baselines.model_factory import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 7)))
    device = choose_device(args.device)
    bundle = create_dataloaders(config.get("dataset", {}), seed=int(config.get("seed", 7)))
    model = build_model(config, bundle.num_classes).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    result = evaluate(model, bundle.test_loader, torch.nn.CrossEntropyLoss(), device)
    print(result)
    if args.out:
        save_json(result, args.out)


if __name__ == "__main__":
    main()

