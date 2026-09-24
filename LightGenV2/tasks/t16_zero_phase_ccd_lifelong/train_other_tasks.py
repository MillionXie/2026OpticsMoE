"""Full single-task training for the three non-EuroSAT paired-modality tasks."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from LightGenV2.tasks.t14_shared_readout_lifelong.data import (
    ClevrRawPairs, PhysicalPermutedCandidates,
)

from .data import PhysicalBinaryPairs, SpeechBinaryPairs
from .model import DirectCCDOptics
from .train_eurosat import batches, save_json, sha256_file


EXPECTED = {
    "clevr": {"train": 140000, "val": 15000, "test": 15000},
    "speech_binary": {"train": 12526, "val": 1686, "test": 1734},
    "physical": {"train": 70400, "val": 14652, "test": 14948},
    "physical_binary": {"train": 140800, "val": 29304, "test": 29896},
}
CLASSES = {"clevr": 2, "speech_binary": 2, "physical": 10,
           "physical_binary": 2}
STORAGE = {
    "clevr": "clevr_lazy_v1",
    "speech_binary": "speech_commands_text_rank8_v2",
    "physical": "physical_video_text_rank10_v3",
    "physical_binary": "physical_video_text_rank10_v3",
}

RESUME_CONTRACT = (
    "task", "architecture", "activation_order", "readout", "class_count",
    "dataset_counts", "batch", "eval_batch", "lr", "seed",
    "source_protocol_sha256", "source_manifest_sha256", "test_policy",
    "clevr_pairwise_weight",
)


def validate_resume_contract(prior, current):
    changed = [key for key in RESUME_CONTRACT if prior.get(key) != current.get(key)]
    if changed:
        raise ValueError(f"resume run changes contract fields: {', '.join(changed)}")


def load_task(protocol_path, task, split):
    protocol = json.loads(protocol_path.read_text())
    if (protocol.get("storage") != STORAGE[task] or
            protocol.get("all_original_samples") is not True):
        raise ValueError("task source protocol does not match the declared full dataset")
    if task == "clevr":
        data = ClevrRawPairs(protocol_path.parent, split)
    elif task == "speech_binary":
        data = SpeechBinaryPairs(Path(protocol["source_root"]), split)
    elif task == "physical":
        roots = {key: Path(path) for key, path in protocol["source_roots"].items()}
        data = PhysicalPermutedCandidates(roots, split, seed=17)
    else:
        roots = {key: Path(path) for key, path in protocol["source_roots"].items()}
        data = PhysicalBinaryPairs(roots, split)
    labels = np.asarray(data.labels, dtype=np.int64)
    if (len(data) != EXPECTED[task][split] or
            not np.array_equal(np.unique(labels), np.arange(CLASSES[task]))):
        raise ValueError(f"unexpected {task}/{split} count or class set")
    return data


def metrics(labels, predictions, classes):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    correct = labels == predictions
    counts = np.bincount(labels, minlength=classes)
    recalls = np.bincount(labels[correct], minlength=classes) / counts
    return {"accuracy": float(correct.mean()), "balanced_accuracy": float(recalls.mean()),
            "per_class_recall": recalls.tolist(), "n": len(labels)}


def clevr_pairwise_loss(logits, targets):
    """Compare positive and negative queries of each *same* source image."""
    if logits.ndim != 2 or logits.shape[0] % 2 or logits.shape[1] != 10:
        raise ValueError("expected an even batch of ten-class CLEVR logits")
    labels = targets.reshape(-1, 2)
    if not torch.all(labels[:, 0] == 1) or not torch.all(labels[:, 1] == 0):
        raise ValueError("CLEVR pair order must be positive then negative")
    margins = (logits[:, 1] - logits[:, 0]).reshape(-1, 2)
    return F.softplus(margins[:, 1] - margins[:, 0]).mean()


@torch.no_grad()
def evaluate(model, data, device, batch_size, classes):
    model.eval()
    predictions, routing, captures = [], [], []
    for indices in batches(np.arange(len(data)), batch_size):
        output = model(data.get_batch(indices, device))
        predictions.append(output["logits"].argmax(1).cpu().numpy())
        if output["route_power"] is not None:
            routing.append(output["route_power"].cpu().numpy())
            captures.append(output["router_efficiency"].cpu().numpy())
    result = metrics(data.labels, np.concatenate(predictions), classes)
    if routing:
        q = np.concatenate(routing)
        active = model.active_indices[:int(model.active_count)].cpu().numpy()
        result["router"] = {
            "active_slot_ids": (active + 1).tolist(),
            "mean_power_by_slot": q.mean(0).tolist(),
            "std_power_by_slot": q.std(0).tolist(),
            "argmax_fraction_by_slot": (
                np.bincount(q.argmax(1), minlength=model.max_experts) / len(q)).tolist(),
            "mean_active_capture": float(np.concatenate(captures).mean()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--architecture", choices=("moe", "d2nn"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--clevr-pairwise-weight", type=float, default=0.0)
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="Continue an immutable prior run into a new run directory")
    args = parser.parse_args()
    if not (0 < args.min_epochs <= args.epochs and args.patience > 0 and
            args.batch > 0 and args.eval_batch > 0 and args.lr > 0 and
            args.clevr_pairwise_weight >= 0):
        raise ValueError("invalid training budget")
    if args.clevr_pairwise_weight and (args.task != "clevr" or args.batch % 2):
        raise ValueError("paired loss requires CLEVR and an even batch size")
    if ("runs", "simulation") not in list(zip(args.out.parts, args.out.parts[1:])):
        raise ValueError("formal runs must live under runs/simulation")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_task(args.protocol, args.task, "train")
    val = load_task(args.protocol, args.task, "val")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {
        "task": args.task, "architecture": args.architecture,
        "activation_order": "center_out", "readout": "one trainable Linear(784,10), bias=False",
        "class_count": CLASSES[args.task], "dataset_counts": EXPECTED[args.task],
        "selection": "maximum full-validation balanced accuracy",
        "test_policy": "omitted" if args.skip_test else "once after validation selection",
        "epochs": args.epochs, "min_epochs": args.min_epochs, "patience": args.patience,
        "batch": args.batch, "eval_batch": args.eval_batch, "lr": args.lr,
        "clevr_pairwise_weight": args.clevr_pairwise_weight,
        "seed": args.seed, "source_protocol": str(args.protocol),
        "source_protocol_sha256": sha256_file(args.protocol),
        "source_manifest_sha256": json.loads(args.protocol.read_text()).get(
            "source_manifest_sha256"),
        "model_git_commit": commit,
        "resume_checkpoint": (str(args.resume_checkpoint)
                              if args.resume_checkpoint else None),
        "resume_checkpoint_sha256": (sha256_file(args.resume_checkpoint)
                                     if args.resume_checkpoint else None),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device)},
    }
    save_json(args.out / "config.json", config)
    (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    save_json(args.out / "status.json", {"status": "running", "epoch": 0})
    model = DirectCCDOptics(args.architecture, activation_order="center_out").to(device)
    model.configure_stage(0)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best_score, best_epoch, wait = -1.0, 0, 0
    history = []
    start_epoch = 1
    if args.resume_checkpoint is not None:
        prior_dir = args.resume_checkpoint.parent
        prior = json.loads((prior_dir / "config.json").read_text())
        validate_resume_contract(prior, config)
        state = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        start_epoch = int(state["epoch"]) + 1
        if start_epoch > args.epochs:
            raise ValueError("resume target epochs must exceed checkpoint epoch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_score, best_epoch, wait = (float(state["best_score"]),
                                        int(state["best_epoch"]), int(state["wait"]))
        history = json.loads((prior_dir / "history.json").read_text())
        if not history or history[-1]["epoch"] != start_epoch - 1:
            raise ValueError("resume history does not match the last checkpoint")
        shutil.copy2(prior_dir / "best_checkpoint.pt", args.out / "best_checkpoint.pt")
        config["prior_model_git_commit"] = prior["model_git_commit"]
        config["prior_run"] = str(prior_dir)
        save_json(args.out / "config.json", config)
        save_json(args.out / "history.json", history)
    for epoch in range(start_epoch, args.epochs + 1):
        began = time.time()
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        if args.clevr_pairwise_weight:
            pair_order = rng.permutation(len(train) // 2)
            pair_batches = batches(pair_order, args.batch // 2)
            batch_indices = (
                np.column_stack((2 * pair_ids, 2 * pair_ids + 1)).reshape(-1)
                for pair_ids in pair_batches)
        else:
            batch_indices = batches(rng.permutation(len(train)), args.batch)
        loss_sum = 0.0
        for indices in batch_indices:
            amplitude = train.get_batch(indices, device)
            target = torch.as_tensor(train.labels[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(amplitude)
            loss = F.cross_entropy(output["logits"], target)
            if args.clevr_pairwise_weight:
                loss = loss + args.clevr_pairwise_weight * clevr_pairwise_loss(
                    output["logits"], target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                            if p.requires_grad], 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        validation = evaluate(model, val, device, args.eval_batch, CLASSES[args.task])
        score = validation["balanced_accuracy"]
        if score > best_score + 1e-5:
            best_score, best_epoch, wait = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation": validation, "config": config},
                       args.out / "best_checkpoint.pt")
        else:
            wait += 1
        row = {"epoch": epoch, "train_loss": loss_sum / len(train),
               "train_samples": len(train), "validation": validation,
               "seconds": time.time() - began, "best_epoch": best_epoch}
        history.append(row)
        save_json(args.out / "history.json", history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_score": best_score,
                    "best_epoch": best_epoch, "wait": wait},
                   args.out / "last_checkpoint.pt")
        save_json(args.out / "status.json", {"status": "running", "epoch": epoch,
                                               "best_epoch": best_epoch,
                                               "best_val_balanced_accuracy": best_score})
        print(json.dumps({"epoch": epoch, "train_loss": row["train_loss"],
                          "val_balanced_accuracy": score,
                          "best_val_balanced_accuracy": best_score,
                          "seconds": row["seconds"]}), flush=True)
        if epoch >= args.min_epochs and wait >= args.patience:
            break
    selected = torch.load(args.out / "best_checkpoint.pt", map_location=device,
                          weights_only=False)
    model.load_state_dict(selected["model"])
    result = {"selected_epoch": best_epoch, "validation": selected["validation"],
              "model_git_commit": commit,
              "selected_checkpoint_git_commit": selected["config"]["model_git_commit"]}
    if not args.skip_test:
        test = load_task(args.protocol, args.task, "test")
        result["test"] = evaluate(model, test, device, args.eval_batch, CLASSES[args.task])
    save_json(args.out / "result.json", result)
    save_json(args.out / "status.json", {"status": "complete", "epoch": history[-1]["epoch"],
                                           "best_epoch": best_epoch})
    print(json.dumps({"result": result}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--out" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--out") + 1])
            if out.is_dir():
                save_json(out / "status.json", {"status": "failed",
                                                "error": f"{type(error).__name__}: {error}"})
        raise
