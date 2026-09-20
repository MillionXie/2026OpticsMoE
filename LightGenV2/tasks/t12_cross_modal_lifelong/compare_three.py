"""Build one auditable three-model comparison from completed run results."""
import argparse
import hashlib
import json
from pathlib import Path


TASKS = ("sen12ms", "clevr", "sonyc", "video")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(task, metrics):
    if task == "sen12ms":
        return metrics["macro_f1"]
    if task == "sonyc":
        return metrics["macro_ap"]
    return metrics["balanced_accuracy"]


def summarize(path):
    path = Path(path)
    result = json.loads(path.read_text())
    return {
        "source": str(path),
        "source_sha256": sha256(path),
        "training": result["training"],
        "selected_epoch": result["selected_epoch"],
        "validation": {task: score(task, result["validation"][task]) for task in TASKS},
        "test": {task: score(task, result["test"][task]) for task in TASKS},
        "mean_test": result["mean_test_selection_metric"],
        "continual": result.get("continual"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--sequential", type=Path, required=True)
    parser.add_argument("--moe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    models = {"joint_d2nn": summarize(args.joint),
              "sequential_d2nn": summarize(args.sequential),
              "lifelong_moe": summarize(args.moe)}
    deltas = {}
    for reference in ("joint_d2nn", "sequential_d2nn"):
        deltas[f"moe_minus_{reference}"] = {
            "mean_test": models["lifelong_moe"]["mean_test"] - models[reference]["mean_test"],
            "test": {task: models["lifelong_moe"]["test"][task] - models[reference]["test"][task]
                     for task in TASKS},
        }
    output = {"metric_contract": {"sen12ms":"macro_f1", "clevr":"balanced_accuracy",
                                   "sonyc":"macro_ap", "video":"balanced_accuracy"},
              "models":models, "deltas":deltas}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
