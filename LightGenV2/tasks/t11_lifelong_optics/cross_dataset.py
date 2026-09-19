"""Reproducible two-dataset continual-learning run for the optical MoE."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .data import sha
from .model import OpticalMoE, loss
from .run import save


OLD_MASK = [True] * 4 + [False] * 8
NEW_MASK = [False] * 4 + [True] * 4 + [False] * 4


def load_dataset(path, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("license") != "CC BY 4.0":
        raise ValueError(f"Unverified license in {manifest_path}")
    if manifest.get("cache_sha256") != sha(path):
        raise ValueError(f"Data hash mismatch for {path}")
    with np.load(path, allow_pickle=False) as z:
        required = {f"{split}_{kind}" for split in ("train", "val") for kind in ("images", "labels", "ids")}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"Missing arrays in {path}: {sorted(missing)}")
        data = {key: z[key].copy() for key in required}
    for split in ("train", "val"):
        x, y, ids = data[f"{split}_images"], data[f"{split}_labels"], data[f"{split}_ids"]
        if x.dtype != np.uint8 or x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"{path} must contain NHWC uint8 RGB images")
        if set(y.tolist()) != {0, 1}:
            raise ValueError(f"{path} must contain both binary labels")
        if len(x) != len(y) or len(y) != len(ids) or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"Invalid or repeated identities in {path}:{split}")
    if set(data["train_ids"].tolist()) & set(data["val_ids"].tolist()):
        raise ValueError(f"Train/validation identity leakage in {path}")
    return data, manifest


def balanced_subset(labels, per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for cls in (0, 1):
        ids = np.flatnonzero(labels == cls)
        if len(ids) < per_class:
            raise ValueError(f"Class {cls} has {len(ids)} samples, requested {per_class}")
        selected.extend(rng.permutation(ids)[:per_class])
    return np.asarray(selected, dtype=np.int64)


@torch.no_grad()
def evaluate(model, images, labels, batch_size, mask=None, warmup=False):
    model.eval(); probabilities = []; routes = []
    device = next(model.parameters()).device
    for start in range(0, len(labels), batch_size):
        out = model(images[start:start + batch_size].to(device), mask=mask, warmup=warmup)
        probabilities.append(out["probabilities"].cpu()); routes.append(out["routes"].cpu())
    p, q = torch.cat(probabilities), torch.cat(routes); pred = p.argmax(1)
    confusion = torch.bincount(labels * 2 + pred, minlength=4).reshape(2, 2)
    recall = confusion.diag().float() / confusion.sum(1).clamp_min(1)
    metrics = {
        "accuracy": float((pred == labels).float().mean()),
        "balanced_accuracy": float(recall.mean()),
        "loss": float(torch.nn.functional.nll_loss(p.clamp_min(1e-12).log(), labels)),
        "confusion": confusion.tolist(),
        "mean_route": q.mean(0).tolist(),
        "route_std": q.std(0, unbiased=False).tolist(),
        "dominant_expert_counts": torch.bincount(q.argmax(1), minlength=12).tolist(),
    }
    return metrics, p, q


def train_epoch(model, optimizer, current_x, current_y, order, batch_size, rng,
                replay_x=None, replay_y=None, replay_fraction=0.0, warmup=False):
    model.train(); total = 0.0; count = 0
    replay_n = round(batch_size * replay_fraction) if replay_x is not None else 0
    current_n = batch_size - replay_n
    if current_n <= 0:
        raise ValueError("Replay fraction leaves no current-task samples")
    for start in range(0, len(order), current_n):
        ids = order[start:start + current_n]
        xb, yb = current_x[ids], current_y[ids]
        if replay_n:
            rid = rng.choice(len(replay_y), size=replay_n, replace=len(replay_y) < replay_n)
            xb = torch.cat((xb, replay_x[rid])); yb = torch.cat((yb, replay_y[rid]))
        device = next(model.parameters()).device
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True); output = model(xb, warmup=warmup); value = loss(output, yb)
        if not torch.isfinite(value):
            raise RuntimeError("Nonfinite loss")
        value.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
        total += value.item() * len(yb); count += len(yb)
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-a", type=Path, required=True)
    parser.add_argument("--task-a-manifest", type=Path, required=True)
    parser.add_argument("--task-b", type=Path, required=True)
    parser.add_argument("--task-b-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pilot", action="store_true", help="Run one epoch per stage with the real data and optical graph")
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    if args.pilot:
        cfg.update(epochs_A=1, epochs_warmup=1, epochs_B=1)
    args.out.mkdir(parents=True, exist_ok=False); save(args.out / "status.json", {"state": "preparing"})
    try:
        torch.set_num_threads(4); torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
        data_a, manifest_a = load_dataset(args.task_a, args.task_a_manifest)
        data_b, manifest_b = load_dataset(args.task_b, args.task_b_manifest)
        ia = balanced_subset(data_a["train_labels"], cfg["train_per_class_A"], cfg["seed"])
        ib = balanced_subset(data_b["train_labels"], cfg["train_per_class_B"], cfg["seed"] + 1)
        va = balanced_subset(data_a["val_labels"], cfg["val_per_class_A"], cfg["seed"] + 2)
        vb = balanced_subset(data_b["val_labels"], cfg["val_per_class_B"], cfg["seed"] + 3)
        replay_ids = balanced_subset(data_a["train_labels"][ia], cfg["replay_capacity"] // 2, cfg["seed"] + 4)
        x_a = torch.from_numpy(data_a["train_images"][ia]); y_a = torch.from_numpy(data_a["train_labels"][ia]).long()
        x_b = torch.from_numpy(data_b["train_images"][ib]); y_b = torch.from_numpy(data_b["train_labels"][ib]).long()
        vx_a = torch.from_numpy(data_a["val_images"][va]); vy_a = torch.from_numpy(data_a["val_labels"][va]).long()
        vx_b = torch.from_numpy(data_b["val_images"][vb]); vy_b = torch.from_numpy(data_b["val_labels"][vb]).long()
        replay_x, replay_y = x_a[replay_ids], y_a[replay_ids]
        save(args.out / "config.json", cfg)
        save(args.out / "split.json", {
            "task_A": manifest_a.get("dataset"), "task_B": manifest_b.get("dataset"),
            "train_A_ids": data_a["train_ids"][ia].tolist(), "train_B_ids": data_b["train_ids"][ib].tolist(),
            "val_A_ids": data_a["val_ids"][va].tolist(), "val_B_ids": data_b["val_ids"][vb].tolist(),
            "replay_A_ids": data_a["train_ids"][ia][replay_ids].tolist(), "test_images_read": False,
        })
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        save(args.out / "metadata.json", {
            "command": sys.argv, "commit": commit, "python": sys.version, "torch": torch.__version__,
            "platform": platform.platform(), "device": args.device, "task_A_manifest": manifest_a,
            "task_B_manifest": manifest_b, "task_A_sha256": sha(args.task_a), "task_B_sha256": sha(args.task_b),
            "test_images_read": False, "scope": "full-data structural pilot" if args.pilot else "validation-selected experiment",
        })
        model = OpticalMoE(cfg).to(args.device); rng = np.random.default_rng(cfg["seed"])
        geometry = {k: list(v.shape) for k, v in model.state_dict().items()}; history = []; audit = []

        model.configure("A"); frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.Adam([
            {"params": [p for p in model.experts if p.requires_grad], "lr": cfg["lr_expert"]},
            {"params": [model.router, model.global_phase], "lr": cfg["lr_shared"]},
        ])
        best = -1.0; stage_dir = args.out / "A"; stage_dir.mkdir()
        for epoch in range(1, cfg["epochs_A"] + 1):
            value = train_epoch(model, optimizer, x_a, y_a, rng.permutation(len(y_a)), cfg["batch_size"], rng)
            ma = evaluate(model, vx_a, vy_a, cfg["batch_size"], OLD_MASK)[0]
            row = {"stage": "A", "epoch": epoch, "train_loss": value, "A_old_only": ma}; history.append(row)
            state = {"model": model.state_dict(), "config": cfg, "stage": "A", "epoch": epoch, "validation": row}
            torch.save(state, stage_dir / "last_checkpoint.pt")
            if ma["balanced_accuracy"] > best:
                best = ma["balanced_accuracy"]; torch.save(state, stage_dir / "best_checkpoint.pt")
            save(args.out / "history.json", history); save(args.out / "status.json", {"state": "training", "stage": "A", "epoch": epoch})
            print(json.dumps({"stage": "A", "epoch": epoch, "loss": value, "A_bal_acc": ma["balanced_accuracy"]}), flush=True)
        for n, p in model.named_parameters():
            if n in frozen and not torch.equal(p, frozen[n]): raise RuntimeError("Frozen changed: " + n)
        model.load_state_dict(torch.load(stage_dir / "best_checkpoint.pt", map_location=args.device, weights_only=False)["model"])
        a_before = evaluate(model, vx_a, vy_a, cfg["batch_size"], OLD_MASK)[0]
        audit.append({"stage": "A", "frozen_unchanged": list(frozen), "geometry_unchanged": geometry == {k: list(v.shape) for k, v in model.state_dict().items()}})

        model.configure("warmup"); frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr_expert"])
        stage_dir = args.out / "warmup"; stage_dir.mkdir()
        for epoch in range(1, cfg["epochs_warmup"] + 1):
            value = train_epoch(model, optimizer, x_b, y_b, rng.permutation(len(y_b)), cfg["batch_size"], rng, warmup=True)
            a_old = evaluate(model, vx_a, vy_a, cfg["batch_size"], OLD_MASK)[0]
            b_uniform_new = evaluate(model, vx_b, vy_b, cfg["batch_size"], warmup=True)[0]
            row = {"stage": "warmup", "epoch": epoch, "train_loss": value, "A_old_only": a_old, "B_uniform_new": b_uniform_new}; history.append(row)
            torch.save({"model": model.state_dict(), "config": cfg, "stage": "warmup", "epoch": epoch}, stage_dir / "last_checkpoint.pt")
            save(args.out / "history.json", history); save(args.out / "status.json", {"state": "training", "stage": "warmup", "epoch": epoch})
            print(json.dumps({"stage": "warmup", "epoch": epoch, "loss": value, "A_old_bal_acc": a_old["balanced_accuracy"], "B_uniform_new_bal_acc": b_uniform_new["balanced_accuracy"]}), flush=True)
        for n, p in model.named_parameters():
            if n in frozen and not torch.equal(p, frozen[n]): raise RuntimeError("Frozen changed: " + n)
        audit.append({"stage": "warmup", "frozen_unchanged": list(frozen), "geometry_unchanged": geometry == {k: list(v.shape) for k, v in model.state_dict().items()}})

        model.configure("B"); frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        groups = [
            {"params": [p for p in model.experts if p.requires_grad], "lr": cfg["lr_expert"]},
            {"params": [model.router, model.global_phase], "lr": cfg["lr_shared"]},
        ]
        optimizer = torch.optim.Adam(groups); best = -1.0; stage_dir = args.out / "B"; stage_dir.mkdir()
        for epoch in range(1, cfg["epochs_B"] + 1):
            value = train_epoch(model, optimizer, x_b, y_b, rng.permutation(len(y_b)), cfg.get("b_batch_size", cfg["batch_size"]), rng,
                                replay_x, replay_y, cfg["replay_fraction"])
            a_all = evaluate(model, vx_a, vy_a, cfg["batch_size"])[0]
            a_old = evaluate(model, vx_a, vy_a, cfg["batch_size"], OLD_MASK)[0]
            b_all = evaluate(model, vx_b, vy_b, cfg["batch_size"])[0]
            b_new = evaluate(model, vx_b, vy_b, cfg["batch_size"], NEW_MASK)[0]
            row = {"stage": "B", "epoch": epoch, "train_loss": value, "A_all": a_all, "A_old_only": a_old, "B_all": b_all, "B_new_only": b_new}; history.append(row)
            score = (a_all["balanced_accuracy"] + b_all["balanced_accuracy"]) / 2
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg, "stage": "B", "epoch": epoch, "validation": row, "selection_score": score}
            torch.save(state, stage_dir / "last_checkpoint.pt")
            if score > best:
                best = score; torch.save(state, stage_dir / "best_checkpoint.pt")
            save(args.out / "history.json", history); save(args.out / "status.json", {"state": "training", "stage": "B", "epoch": epoch})
            print(json.dumps({"stage": "B", "epoch": epoch, "loss": value, "A_all_bal_acc": a_all["balanced_accuracy"], "A_old_bal_acc": a_old["balanced_accuracy"], "B_all_bal_acc": b_all["balanced_accuracy"], "score": score}), flush=True)
        for n, p in model.named_parameters():
            if n in frozen and not torch.equal(p, frozen[n]): raise RuntimeError("Frozen changed: " + n)
        audit.append({"stage": "B", "frozen_unchanged": list(frozen), "geometry_unchanged": geometry == {k: list(v.shape) for k, v in model.state_dict().items()}})
        best_state = torch.load(stage_dir / "best_checkpoint.pt", map_location=args.device, weights_only=False); model.load_state_dict(best_state["model"])
        results = {"A_before_old_only": a_before, "selected_B_epoch": best_state["epoch"], "selection_score": best_state["selection_score"]}
        for task, x, y in (("A", vx_a, vy_a), ("B", vx_b, vy_b)):
            for name, mask in (("all", None), ("old_only", OLD_MASK), ("new_only", NEW_MASK)):
                metrics, probabilities, routes = evaluate(model, x, y, cfg["batch_size"], mask)
                results[f"{task}_{name}"] = metrics
                np.savez_compressed(args.out / f"{task}_{name}.npz", labels=y.numpy(), probabilities=probabilities.numpy(), routes=routes.numpy())
        results["BWT_all"] = results["A_all"]["balanced_accuracy"] - a_before["balanced_accuracy"]
        results["BWT_old_only"] = results["A_old_only"]["balanced_accuracy"] - a_before["balanced_accuracy"]
        results["headline_metric"] = "balanced_accuracy"
        results["interpretation"] = "Mask ablations alter coherent interference and are diagnostics, not additive knowledge estimates."
        save(args.out / "metrics.json", results); save(args.out / "audit.json", audit); save(args.out / "status.json", {"state": "complete"})
    except BaseException as error:
        save(args.out / "status.json", {"state": "failed", "error": repr(error)}); raise


if __name__ == "__main__":
    main()
