import argparse
import csv
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_nine_expert_as_multitask_moe as base_train
from opticalmoe.data import create_dataloaders
from opticalmoe.data.dsprites import DSpritesMultiLabelDataset
from opticalmoe.utils import load_config, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 9-expert fair134 AS global router on dSprites shape/scale."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--disable_visualization", action="store_true")
    return parser.parse_args()


def write_rows(path: Path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def _task_dataset_cfg(config, task_name: str):
    for task in config["training"]["multitask"]["tasks"]:
        if task["name"].lower() == task_name:
            return dict(task["dataset"])
    raise KeyError(f"Missing task config: {task_name}")


@torch.no_grad()
def same_input_task_switching_evaluation(model, config, run_dir: Path, device, max_samples: int):
    shape_cfg = _task_dataset_cfg(config, "shape")
    scale_cfg = _task_dataset_cfg(config, "scale")
    if shape_cfg.get("name", "").lower() != "dsprites" or scale_cfg.get("name", "").lower() != "dsprites":
        raise ValueError("dSprites same-input evaluation requires shape/scale dSprites tasks.")
    dataset = DSpritesMultiLabelDataset(
        root=shape_cfg.get("root", "./data"),
        input_size=int(shape_cfg.get("input_size", 134)),
        split="test",
        val_split=float(shape_cfg.get("val_split", 0.1)),
        test_split=float(shape_cfg.get("test_split", 0.1)),
        seed=int(config.get("seed", 7)),
        download=bool(shape_cfg.get("download", True)),
        max_samples=int(max_samples),
        npz_path=shape_cfg.get("npz_path"),
    )
    images, shape_targets, scale_targets = [], [], []
    for index in range(len(dataset)):
        image, labels = dataset[index]
        images.append(image)
        shape_targets.append(int(labels["shape"]))
        scale_targets.append(int(labels["scale"]))
    if not images:
        raise RuntimeError("dSprites same-input evaluation dataset is empty.")
    batch = torch.stack(images, dim=0).to(device)
    shape_target_tensor = torch.as_tensor(shape_targets, dtype=torch.long, device=device)
    scale_target_tensor = torch.as_tensor(scale_targets, dtype=torch.long, device=device)
    model.eval()
    logits_shape = model(batch, task_name="shape")
    logits_scale = model(batch, task_name="scale")
    pred_shape = logits_shape.argmax(dim=1)
    pred_scale = logits_scale.argmax(dim=1)
    shape_acc = float((pred_shape == shape_target_tensor).float().mean().item())
    scale_acc = float((pred_scale == scale_target_tensor).float().mean().item())
    rows = []
    for index in range(batch.shape[0]):
        rows.append(
            {
                "sample_index": index,
                "shape_target": int(shape_targets[index]),
                "shape_pred": int(pred_shape[index].cpu()),
                "shape_correct": int(pred_shape[index].item() == shape_targets[index]),
                "scale_target": int(scale_targets[index]),
                "scale_pred": int(pred_scale[index].cpu()),
                "scale_correct": int(pred_scale[index].item() == scale_targets[index]),
            }
        )
    write_rows(run_dir / "same_input_task_switching.csv", rows)
    payload = {
        "num_samples": int(batch.shape[0]),
        "shape_accuracy": shape_acc,
        "scale_accuracy": scale_acc,
        "shape_targets": shape_targets,
        "shape_predictions": pred_shape.cpu().tolist(),
        "scale_targets": scale_targets,
        "scale_predictions": pred_scale.cpu().tolist(),
        "note": "Same dSprites images evaluated as task_name=shape and task_name=scale.",
    }
    save_json(payload, str(run_dir / "same_input_task_switching.json"))
    if config.get("visualization", {}).get("enabled", True):
        plt = base_train.ensure_matplotlib()
        count = min(batch.shape[0], 12)
        fig, axes = plt.subplots(1, count, figsize=(2.4 * count, 3.0), squeeze=False)
        for index in range(count):
            axes[0, index].imshow(batch[index, 0].detach().cpu(), cmap="gray")
            axes[0, index].set_title(
                f"shape {shape_targets[index]}->{int(pred_shape[index])}\n"
                f"scale {scale_targets[index]}->{int(pred_scale[index])}"
            )
            axes[0, index].axis("off")
        fig.suptitle("Same Input, Different Task Prompt/Readout")
        fig.tight_layout()
        fig.savefig(run_dir / "same_input_task_switching_samples.png")
        plt.close(fig)
    return payload


def run_same_input_evaluation(args):
    config = load_config(args.config)
    run_name = args.run_name or config.get("experiment", {}).get(
        "run_name", "nine_expert_dsprites_shape_scale"
    )
    run_dir = PROJECT_ROOT / "runs" / run_name
    device = base_train.choose_device(args.device or config.get("device", "auto"))
    # Recreate class counts through the normal loader path so the model shape
    # matches the trained checkpoint without hardcoding future dSprites tasks.
    train_loaders, _val_loaders, _test_loaders, class_counts = base_train.create_task_loaders(
        config,
        int(config.get("seed", 7)),
        force_smoke=bool(args.smoke_test),
    )
    task_names = list(train_loaders.keys())
    model = base_train.build_model(config, task_names, class_counts).to(device)
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No best checkpoint found under {run_dir}.")
    payload = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    max_samples = int(config.get("visualization", {}).get("num_samples", 4)) * 8
    result = same_input_task_switching_evaluation(
        model,
        config,
        run_dir,
        device,
        max_samples=max_samples,
    )
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["same_input_task_switching"] = result
        save_json(summary, str(summary_path))
    print(
        "same-input task switching: "
        f"shape_acc={result['shape_accuracy']:.4f}, "
        f"scale_acc={result['scale_accuracy']:.4f}"
    )


def main():
    args = parse_args()
    # Let the generic 9-expert trainer handle all standard multitask training
    # artifacts. This script adds only the dSprites same-input evaluation after
    # training, keeping the generic path dataset-agnostic.
    base_train.main()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    run_same_input_evaluation(args)


if __name__ == "__main__":
    main()
