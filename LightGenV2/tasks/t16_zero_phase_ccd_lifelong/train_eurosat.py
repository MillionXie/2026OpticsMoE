"""Full paired RGB/SAR EuroSAT training with the one-Linear CCD readout."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import PairedEuroSatFields
from .model import DirectCCDOptics


EXPECTED = {"train": 15998, "val": 5465, "test": 5429}


def source_paths(protocol_path):
    protocol = json.loads(protocol_path.read_text())
    source = protocol.get("source_protocol", protocol)
    if source.get("storage") != "eurosat_rgb_sar_v1" or source.get("all_original_samples") is not True:
        raise ValueError("expected audited, full original RGB/SAR source")
    return Path(source["trainval_npz"]), Path(source["holdout_npz"])


def load_split(trainval, holdout, split):
    data = PairedEuroSatFields(trainval, holdout, split)
    if len(data) != EXPECTED[split] or set(np.unique(data.labels)) != set(range(10)):
        raise ValueError(f"unexpected {split} count or class set")
    return data


def batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def accuracy_metrics(labels, predicted):
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    correct = predicted == labels
    counts = np.bincount(labels, minlength=10)
    recalls = np.bincount(labels[correct], minlength=10) / counts
    return {"accuracy": float(correct.mean()),
            "balanced_accuracy": float(recalls.mean()),
            "per_class_recall": recalls.tolist(), "n": int(len(labels))}


def routing_balance_penalty(route_power, active_indices):
    """Weak batch-level slot use penalty; never assign samples to a task group."""
    q = route_power.index_select(1, active_indices)
    return (q.mean(0) - 1.0 / len(active_indices)).square().sum()


@torch.no_grad()
def evaluate(model, data, device, batch_size):
    model.eval()
    guesses = []
    route_rows = []
    captures = []
    for indices in batches(np.arange(len(data)), batch_size):
        output = model(data[indices].to(device))
        guesses.append(output["logits"].argmax(1).cpu().numpy())
        if output["route_power"] is not None:
            route_rows.append(output["route_power"].cpu().numpy())
            captures.append(output["router_efficiency"].cpu().numpy())
    metrics = accuracy_metrics(data.labels, np.concatenate(guesses))
    if route_rows:
        q = np.concatenate(route_rows)
        active = model.active_indices[:int(model.active_count)].cpu().numpy()
        winners = q.argmax(1)
        metrics["router"] = {
            "active_slot_ids": (active + 1).tolist(),
            "mean_power_by_slot": q.mean(0).tolist(),
            "std_power_by_slot": q.std(0).tolist(),
            "argmax_fraction_by_slot": (
                np.bincount(winners, minlength=model.max_experts) / len(q)).tolist(),
            "mean_active_capture": float(np.concatenate(captures).mean()),
        }
    return metrics


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--architecture", choices=("moe", "d2nn"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--route-balance-weight", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (0 < args.min_epochs <= args.epochs and args.patience > 0 and
            args.batch > 0 and args.eval_batch > 0 and args.lr > 0 and
            args.route_balance_weight >= 0):
        raise ValueError("invalid training budget")
    if ("runs", "simulation") not in list(zip(args.out.parts, args.out.parts[1:])):
        raise ValueError("formal runs must live under runs/simulation")
    if args.resume:
        if not (args.out / "last_checkpoint.pt").exists():
            raise FileNotFoundError("resume requested but last checkpoint missing")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainval, holdout = source_paths(args.protocol)
    train = load_split(trainval, holdout, "train")
    val = load_split(trainval, holdout, "val")
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {"task": "eurosat_paired_rgb_sar", "architecture": args.architecture,
              "activation_order": "center_out", "readout": "one trainable Linear(784,10), bias=False",
              "dataset_counts": {"train": len(train), "val": len(val), "test": EXPECTED["test"]},
              "selection": "maximum full-validation balanced accuracy; test opened once afterward",
              "epochs": args.epochs, "min_epochs": args.min_epochs,
              "patience": args.patience, "batch": args.batch, "eval_batch": args.eval_batch,
              "lr": args.lr, "seed": args.seed,
              "route_balance_weight": args.route_balance_weight,
              "source_protocol": str(args.protocol),
              "source_sha256": {"trainval": sha256_file(trainval), "holdout": sha256_file(holdout)},
              "model_git_commit": code_commit,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "cuda": torch.version.cuda, "device": str(device)}}
    if args.resume:
        prior = json.loads((args.out / "config.json").read_text())
        if prior != config:
            raise ValueError("resume config differs from original run")
        history = json.loads((args.out / "history.json").read_text())
    else:
        save_json(args.out / "config.json", config)
        (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
        history = []
    save_json(args.out / "status.json", {"status": "running", "epoch": len(history)})
    model = DirectCCDOptics(args.architecture, activation_order="center_out").to(device)
    model.configure_stage(0)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best_score, best_epoch, wait, start_epoch = -1.0, 0, 0, 1
    if args.resume:
        checkpoint = torch.load(args.out / "last_checkpoint.pt", map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_score, best_epoch, wait = (checkpoint["best_score"],
                                        checkpoint["best_epoch"], checkpoint["wait"])
        start_epoch = checkpoint["epoch"] + 1
    for epoch in range(start_epoch, args.epochs + 1):
        began = time.time()
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        loss_sum, classification_sum, balance_sum, sample_count = 0.0, 0.0, 0.0, 0
        for indices in batches(order, args.batch):
            amplitude = train[indices].to(device)
            targets = torch.as_tensor(train.labels[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(amplitude)
            classification = F.cross_entropy(output["logits"], targets)
            balance = (routing_balance_penalty(output["route_power"],
                                                model.active_indices[:int(model.active_count)])
                       if args.architecture == "moe" and args.route_balance_weight > 0
                       else classification.new_zeros(()))
            loss = classification + args.route_balance_weight * balance
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                            if p.requires_grad], 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            classification_sum += float(classification.detach()) * len(indices)
            balance_sum += float(balance.detach()) * len(indices)
            sample_count += len(indices)
        validation = evaluate(model, val, device, args.eval_batch)
        score = validation["balanced_accuracy"]
        improved = score > best_score + 1e-5
        if improved:
            best_score, best_epoch, wait = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "validation": validation, "config": config},
                       args.out / "best_checkpoint.pt")
        else:
            wait += 1
        row = {"epoch": epoch, "train_loss": loss_sum / sample_count,
               "train_classification_loss": classification_sum / sample_count,
               "train_route_balance_penalty": balance_sum / sample_count,
               "train_samples": sample_count, "validation": validation,
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
    test = load_split(trainval, holdout, "test")
    final = {"selected_epoch": best_epoch,
             "validation": selected["validation"],
             "test": evaluate(model, test, device, args.eval_batch),
             "model_git_commit": code_commit}
    save_json(args.out / "result.json", final)
    save_json(args.out / "status.json", {"status": "complete", "epoch": history[-1]["epoch"],
                                           "best_epoch": best_epoch})
    print(json.dumps({"result": final}), flush=True)


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
