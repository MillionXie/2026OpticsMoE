"""Inference-only cross-task matrix for independently trained D2NN optics.

Each cell uses the optical tensors from the row task and the already-trained
single Linear(784, C) head from the column task.  No parameter is optimized.
"""
import argparse
import json
from pathlib import Path

import torch

from .run import TASK_ORDER, build_model, evaluate, load_tasks, save, selection_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", choices=TASK_ORDER, required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="TASK=best_checkpoint.pt; provide all four tasks")
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--clevr", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    checkpoint_paths = {}
    for item in args.checkpoint:
        task, raw_path = item.split("=", 1)
        checkpoint_paths[task] = Path(raw_path)
    missing = set(TASK_ORDER) - set(checkpoint_paths)
    if missing:
        raise ValueError(f"missing checkpoints: {sorted(missing)}")

    tasks = load_tasks({
        "eurosat": args.eurosat, "clevr": args.clevr,
        "speech": args.speech, "physical": args.physical,
    }, require_full=True)
    checkpoints = {
        name: torch.load(path, map_location=device, weights_only=False)
        for name, path in checkpoint_paths.items()
    }
    model = build_model(
        "d2nn", cfg, cfg["seed"],
        max_experts=int(cfg.get("single_task_d2nn_max_experts", 4))).to(device)
    model.load_state_dict(checkpoints[args.source]["model"])
    source_state = checkpoints[args.source]["model"]

    details = {}
    scores = {}
    for target in TASK_ORDER:
        target_state = checkpoints[target]["model"]
        head_state = {
            key.removeprefix(f"heads.{target}."): value
            for key, value in target_state.items()
            if key.startswith(f"heads.{target}.")
        }
        model.heads[target].load_state_dict(head_state)
        metrics, _, _ = evaluate(model, tasks[target], "test", device, cfg["eval_batch"])
        details[target] = metrics
        scores[target] = selection_score(target, metrics)
        print(json.dumps({"source_optics": args.source, "target": target,
                          "test_balanced_accuracy": scores[target]}), flush=True)

    result = {
        "protocol": "inference_only_source_optics_plus_pretrained_target_linear_head",
        "optimization_steps": 0,
        "electronic_readout": "single_linear_layer",
        "source": args.source,
        "source_optical_checkpoint": str(checkpoint_paths[args.source]),
        "target_head_checkpoints": {k: str(v) for k, v in checkpoint_paths.items()},
        "score_row": scores,
        "details": details,
        "source_optical_state_preserved": all(
            torch.equal(value, model.state_dict()[key])
            for key, value in source_state.items()
            if not key.startswith("heads.")
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / f"{args.source}.json", result)


if __name__ == "__main__":
    main()
