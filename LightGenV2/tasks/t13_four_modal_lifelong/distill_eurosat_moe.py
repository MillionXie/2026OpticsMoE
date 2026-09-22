"""Validation-only electronic-teacher distillation for the EuroSAT MoE.

The teacher is used only to produce soft training targets.  Student checkpoints
contain the optical MoE and its existing ``Linear(784, 10)`` readout; no
teacher parameters or extra inference layers are serialized with the student.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .run import (build_model, chunks, classification_metrics, device_fields,
                  evaluate, load_tasks, save, seed_all, selection_score)


class EuroSatTeacher(nn.Module):
    """Training-only MLP over the audited 128-D shared frozen feature."""
    def __init__(self, classes: int = 10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, classes),
        )

    @staticmethod
    def unpack_feature(fields):
        """Invert FeatureFields' exact 16x8 nearest-neighbour expansion."""
        if fields.ndim != 3 or tuple(fields.shape[-2:]) != (224, 224):
            raise ValueError(f"expected Bx224x224 fields, got {tuple(fields.shape)}")
        return F.adaptive_avg_pool2d(fields[:, None], (16, 8)).flatten(1)

    def forward(self, fields):
        return self.classifier(self.unpack_feature(fields))


def training_field(fields):
    """Keep frozen-feature coordinates fixed; their axes are not image pixels."""
    return fields


@torch.no_grad()
def evaluate_teacher(model, task, split, device, batch):
    fields, labels, rows = task.splits[split]
    model.eval(); probabilities = []
    for start in range(0, len(labels), batch):
        x = device_fields(fields, slice(start, start + batch), device)
        probabilities.append(model(x).softmax(1).cpu().numpy())
    p = np.concatenate(probabilities)
    return classification_metrics(task.name, labels, p, rows)


def train_teacher(task, root, device, seed, epochs, batch, lr):
    teacher_root = root / "teacher"; teacher_root.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = EuroSatTeacher(task.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=lr * 0.05)
    labels = task.splits["train"][1]
    best = -float("inf"); history = []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []; started = time.time()
        order = np.random.default_rng(seed + epoch).permutation(len(labels))
        for ix in chunks(order, batch):
            x = training_field(device_fields(task.splits["train"][0], ix, device))
            y = torch.as_tensor(labels[ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step(); losses.append(float(loss.detach()))
        scheduler.step()
        val = evaluate_teacher(model, task, "val", device, batch)
        score = selection_score(task.name, val)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "validation_score": score, "validation": val,
               "seconds": time.time() - started}
        history.append(row); save(teacher_root / "history.json", history)
        checkpoint = {"model": copy.deepcopy(model.state_dict()), "epoch": epoch,
                      "validation": val, "score": score, "training_only": True}
        torch.save(checkpoint, teacher_root / "last_checkpoint.pt")
        if score > best:
            best = score; torch.save(checkpoint, teacher_root / "best_checkpoint.pt")
        print(json.dumps({"arch": "electronic_teacher_training_only", "epoch": epoch,
                          "val": score, "loss": row["loss"],
                          "seconds": row["seconds"]}), flush=True)
    best_checkpoint = torch.load(teacher_root / "best_checkpoint.pt",
                                 map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    metrics = {split: evaluate_teacher(model, task, split, device, batch)
               for split in ("train", "val")}
    result = {"selected_epoch": best_checkpoint["epoch"], "training_only": True,
              "test_was_evaluated": False, "metrics": metrics}
    save(teacher_root / "results.json", result)
    return teacher_root / "best_checkpoint.pt"


def assert_student_inference_contract(model):
    head = model.heads["eurosat"]
    if not isinstance(head, nn.Linear) or head.in_features != 784 or head.out_features != 10:
        raise AssertionError(f"expected Linear(784,10), got {head!r}")


def train_student(task, cfg, root, device, init_checkpoint, teacher_checkpoint,
                  epochs, batch, lr, alpha, temperature, ema_decay):
    student_root = root / "student"; student_root.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"]) + 3000
    seed_all(seed)
    model = build_model("moe", cfg, int(cfg["seed"]), max_experts=16).to(device)
    model.configure_single_task("eurosat")
    initial = torch.load(init_checkpoint, map_location=device, weights_only=False)
    if initial.get("task") != "eurosat":
        raise ValueError("student initialization must be an EuroSAT checkpoint")
    model.load_state_dict(initial["model"])
    assert_student_inference_contract(model)

    teacher = EuroSatTeacher(task.classes).to(device)
    teacher_state = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    teacher.load_state_dict(teacher_state["model"]); teacher.eval(); teacher.requires_grad_(False)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr,
                                  weight_decay=float(cfg.get("moe_weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=lr * 0.05)
    ema = {key: parameter.detach().clone()
           for key, parameter in model.named_parameters() if parameter.requires_grad}
    labels = task.splits["train"][1]
    route_balance = float(cfg.get("route_balance_eurosat", cfg.get("route_balance", 0.0)))
    initial_val = evaluate(model, task, "val", device, batch)[0]
    best = selection_score(task.name, initial_val)
    initial_student = {"model": copy.deepcopy(model.state_dict()), "epoch": 0,
                       "task": "eurosat", "validation": initial_val, "score": best,
                       "inference_readout": "Linear(784,10)",
                       "teacher_included_at_inference": False,
                       "initialized_from": str(init_checkpoint)}
    torch.save(initial_student, student_root / "best_checkpoint.pt")
    history = [{"epoch": 0, "validation_score": best, "validation": initial_val,
                "initialized_from": str(init_checkpoint)}]
    for epoch in range(1, epochs + 1):
        model.train(); losses = []; hard_losses = []; soft_losses = []; started = time.time()
        order = np.random.default_rng(seed + epoch).permutation(len(labels))
        for ix in chunks(order, batch):
            x = training_field(device_fields(task.splits["train"][0], ix, device))
            y = torch.as_tensor(labels[ix], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(x)
            output = model(x, "eurosat")
            student_logits = output["logits"]
            hard = F.cross_entropy(student_logits, y)
            soft = F.kl_div(F.log_softmax(student_logits / temperature, 1),
                            F.softmax(teacher_logits / temperature, 1),
                            reduction="batchmean") * temperature ** 2
            loss = (1.0 - alpha) * hard + alpha * soft
            if route_balance and output["route_power"] is not None:
                q = output["route_power"][:, :4]
                loss = loss + route_balance * (q.mean(0) - 0.25).square().sum()
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            with torch.no_grad():
                for key, parameter in model.named_parameters():
                    if key in ema:
                        ema[key].mul_(ema_decay).add_(parameter, alpha=1.0 - ema_decay)
            losses.append(float(loss.detach())); hard_losses.append(float(hard.detach()))
            soft_losses.append(float(soft.detach()))
        scheduler.step()
        raw_state = copy.deepcopy(model.state_dict())
        raw_val = evaluate(model, task, "val", device, batch)[0]
        raw_score = selection_score(task.name, raw_val)
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                if key in ema:
                    parameter.copy_(ema[key])
        ema_val = evaluate(model, task, "val", device, batch)[0]
        ema_score = selection_score(task.name, ema_val)
        if ema_score >= raw_score:
            selected_state = copy.deepcopy(model.state_dict())
            val, score, variant = ema_val, ema_score, "ema"
        else:
            selected_state = raw_state
            val, score, variant = raw_val, raw_score, "raw"
        model.load_state_dict(raw_state)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "hard_loss": float(np.mean(hard_losses)),
               "distillation_loss": float(np.mean(soft_losses)),
               "validation_score": score, "raw_validation_score": raw_score,
               "ema_validation_score": ema_score, "selected_variant": variant,
               "validation": val, "seconds": time.time() - started}
        history.append(row); save(student_root / "history.json", history)
        checkpoint = {"model": selected_state, "epoch": epoch, "task": "eurosat",
                      "validation": val, "score": score,
                      "inference_readout": "Linear(784,10)",
                      "teacher_included_at_inference": False,
                      "training_teacher_checkpoint": str(teacher_checkpoint),
                      "initialized_from": str(init_checkpoint)}
        torch.save(checkpoint, student_root / "last_checkpoint.pt")
        if score > best:
            best = score; torch.save(checkpoint, student_root / "best_checkpoint.pt")
        print(json.dumps({"arch": "moe_teacher_distilled", "epoch": epoch,
                          "val": score, "raw_val": raw_score, "ema_val": ema_score,
                          "loss": row["loss"], "seconds": row["seconds"]}), flush=True)

    best_checkpoint = torch.load(student_root / "best_checkpoint.pt",
                                 map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"]); assert_student_inference_contract(model)
    metrics = {split: evaluate(model, task, split, device, batch)[0]
               for split in ("train", "val")}
    result = {"selected_epoch": best_checkpoint["epoch"],
              "validation_selection_score": selection_score(task.name, metrics["val"]),
              "test_was_evaluated": False,
              "inference_readout": "Linear(784,10)",
              "teacher_included_at_inference": False, "metrics": metrics}
    save(student_root / "results.json", result)
    return student_root / "best_checkpoint.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--student-epochs", type=int, default=15)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "actual_config.json", cfg)
    save(args.out / "distillation_protocol.json", {
        "teacher_is_training_only": True,
        "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
        "student_initial_checkpoint": str(args.init_checkpoint),
        "teacher_epochs": args.teacher_epochs,
        "student_epochs": args.student_epochs,
        "teacher_lr": args.teacher_lr,
        "student_lr": args.student_lr,
        "distillation_alpha": args.alpha,
        "temperature": args.temperature,
        "ema_decay": args.ema_decay,
        "checkpoint_selection_split": "validation",
        "test_was_evaluated": False,
        "inference_readout": "Linear(784,10)",
    })
    (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    task = load_tasks({"eurosat": args.eurosat},
                      require_full=bool(cfg.get("require_full", False)),
                      names=("eurosat",))["eurosat"]
    teacher_checkpoint = args.teacher_checkpoint
    if teacher_checkpoint is None:
        teacher_checkpoint = train_teacher(
            task, args.out, device, int(cfg["seed"]) + 2000,
            args.teacher_epochs, int(cfg["batch"]), args.teacher_lr)
    student_checkpoint = train_student(
        task, cfg, args.out, device, args.init_checkpoint, teacher_checkpoint,
        args.student_epochs, int(cfg["batch"]), args.student_lr,
        args.alpha, args.temperature, args.ema_decay)
    save(args.out / "status.json", {"status": "complete",
         "teacher_checkpoint": str(teacher_checkpoint),
         "student_checkpoint": str(student_checkpoint),
         "test_was_evaluated": False,
         "inference_readout": "Linear(784,10)"})


if __name__ == "__main__":
    main()
