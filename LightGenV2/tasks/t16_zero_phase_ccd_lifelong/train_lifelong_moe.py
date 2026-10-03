"""Sequential T16 MoE/D2NN with one head and explicit replay mode.

One invocation trains one dependent stage (B, C or D). Stage A is an audited
EuroSAT MoE checkpoint. Candidate stages may omit test; only the chosen stage
checkpoint should be evaluated on test before advancing the chain.
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import DirectCCDOptics
from .train_eurosat import (load_split, routing_balance_penalty, save_json,
                            sha256_file, source_paths)
from .train_other_tasks import (CLASSES, EXPECTED, clevr_pairwise_loss,
                                evaluate, load_task)


ORDER = ("eurosat", "clevr", "speech_binary", "physical_binary")
EXPECTED_EURO = {"train": 15998, "val": 5465, "test": 5429}


class IndexedMemory:
    """Fixed training records; the indices refer to the original task split."""

    def __init__(self, data, indices):
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(data.labels)[self.indices]

    def __len__(self):
        return len(self.indices)

    def get_batch(self, indices, device):
        return self.data.get_batch(self.indices[indices], device)


def select_memory_indices(data, name, count, seed):
    """Stratify EuroSAT; preserve positive/negative source pairs elsewhere."""
    if count <= 0 or count > len(data):
        raise ValueError("memory size must fit the original training split")
    rng = np.random.default_rng(seed)
    labels = np.asarray(data.labels)
    if name == "eurosat":
        classes = np.unique(labels)
        if count % len(classes):
            raise ValueError("EuroSAT memory must divide evenly among classes")
        per_class = count // len(classes)
        chosen = np.concatenate([rng.choice(np.flatnonzero(labels == cls), per_class,
                                             replace=False) for cls in classes])
        return np.sort(chosen)
    if count % 2 or len(data) % 2 or not np.array_equal(
            labels.reshape(-1, 2), np.tile([1, 0], (len(data) // 2, 1))):
        raise ValueError("binary memory requires intact positive/negative pairs")
    pairs = np.sort(rng.choice(len(data) // 2, count // 2, replace=False))
    return np.column_stack((2 * pairs, 2 * pairs + 1)).reshape(-1)


class EuroAdapter:
    def __init__(self, source, split, vision_checkpoint=None, device=None):
        trainval, holdout = source_paths(source)
        self.data = load_split(trainval, holdout, split, vision_checkpoint, device)
        self.labels = self.data.labels

    def __len__(self):
        return len(self.data)

    def get_batch(self, indices, device):
        return self.data[indices].to(device)


def load_dataset(protocols, name, split, vision_checkpoint=None, device=None):
    if name == "eurosat":
        data = EuroAdapter(protocols[name], split, vision_checkpoint, device)
        if len(data) != EXPECTED_EURO[split]:
            raise ValueError("EuroSAT count changed")
        return data
    task = {"clevr": "clevr", "speech_binary": "speech_binary",
            "physical_binary": "physical_binary",
            "physical_binary_raw": "physical_binary_raw"}[name]
    return load_task(protocols[name], task, split, vision_checkpoint, device)


def shuffled_batches(data, name, batch_size, rng, pair_physical=False):
    """Return every record once before cycling; keep CLEVR pairs adjacent."""
    if name == "clevr" or (name == "physical_binary_raw" and pair_physical):
        if batch_size % 2 or len(data) % 2:
            raise ValueError("paired task needs an even batch and complete pairs")
        pair_order = rng.permutation(len(data) // 2)
        return [np.column_stack((2 * part, 2 * part + 1)).reshape(-1)
                for part in np.array_split(pair_order,
                    range(batch_size // 2, len(pair_order), batch_size // 2))]
    order = rng.permutation(len(data))
    return [part for part in np.array_split(order,
            range(batch_size, len(order), batch_size))]


def stage_epoch_batches(datasets, names, batch_size, seed, mode="one_pass",
                        pair_physical=False, memory_cycles=50):
    """Interleave full passes; optionally reshuffle/cycle shorter tasks."""
    if mode not in ("one_pass", "balanced_cycle", "memory_spread"):
        raise ValueError("unknown replay schedule")
    pools = {name: shuffled_batches(datasets[name], name, batch_size,
                                    np.random.default_rng(seed + index * 101),
                                    pair_physical=pair_physical)
             for index, name in enumerate(names)}
    steps = max(map(len, pools.values()))
    if mode == "memory_spread":
        if len(names) < 2 or memory_cycles <= 0:
            raise ValueError("memory_spread needs old tasks and positive cycles")
        repeated = {}
        for index, name in enumerate(names[:-1]):
            repeated[name] = list(pools[name])
            for cycle in range(1, memory_cycles):
                repeated[name].extend(shuffled_batches(
                    datasets[name], name, batch_size,
                    np.random.default_rng(seed + index * 101 + cycle * 100003),
                    pair_physical=pair_physical))
        repeated[names[-1]] = pools[names[-1]]
        steps = max(map(len, repeated.values()))
        scheduled = {name: {step: batch for step, batch in zip(
            (np.arange(len(pool)) * steps // len(pool)).tolist(), pool)}
                     for name, pool in repeated.items()}
        for step in range(steps):
            yield {name: batches[step] for name, batches in scheduled.items()
                   if step in batches}
        return
    if mode == "balanced_cycle":
        repeated = {}
        for step in range(steps):
            row = {}
            for index, name in enumerate(names):
                cycle, offset = divmod(step, len(pools[name]))
                if cycle == 0:
                    row[name] = pools[name][offset]
                else:
                    key = (name, cycle)
                    if key not in repeated:
                        repeated[key] = shuffled_batches(
                            datasets[name], name, batch_size,
                            np.random.default_rng(seed + index * 101 + cycle * 100003),
                            pair_physical=pair_physical)
                    row[name] = repeated[key][offset]
            yield row
        return
    scheduled = {name: {step: batch for step, batch in zip(
                    (np.arange(len(pool)) * steps // len(pool)).tolist(), pool)}
                 for name, pool in pools.items()}
    for step in range(steps):
        yield {name: batches[step] for name, batches in scheduled.items()
               if step in batches}


def task_loss(logits, target, name, pairwise_weight, physical_pairwise_weight=0.0):
    loss = F.cross_entropy(logits, target)
    if name == "clevr" and pairwise_weight:
        loss = loss + pairwise_weight * clevr_pairwise_loss(logits, target)
    if name == "physical_binary_raw" and physical_pairwise_weight:
        loss = loss + physical_pairwise_weight * clevr_pairwise_loss(logits, target)
    return loss


def load_previous(model, path, stage, protocols, device, order=ORDER,
                  architecture="moe", vision_sha=None, replay_mode=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    prior = checkpoint.get("config", {})
    if stage == 2:
        trainval, holdout = source_paths(protocols["eurosat"])
        expected = {"trainval": sha256_file(trainval), "holdout": sha256_file(holdout)}
        if (prior.get("task") != "eurosat_paired_rgb_sar" or
                prior.get("source_sha256") != expected or
                prior.get("architecture") != architecture):
            raise ValueError("stage A checkpoint has a different task or data")
    else:
        if (prior.get("stage") != stage - 1 or
                tuple(prior.get("order", ())) != tuple(order) or
                prior.get("architecture") != architecture):
            raise ValueError("previous checkpoint is not the dependent stage")
        expected_hashes = {name: sha256_file(protocols[name])
                           for name in order[:stage - 1]}
        if prior.get("protocol_sha256") != expected_hashes:
            raise ValueError("previous checkpoint used different task protocols")
        if replay_mode == "memory" and prior.get("replay_mode") != "memory":
            raise ValueError("memory replay must continue its own dependent chain")
    if prior.get("activation_order") != "center_out":
        raise ValueError("previous checkpoint has a different optical geometry")
    if prior.get("vision_checkpoint_sha256") != vision_sha:
        raise ValueError("previous checkpoint has a different frozen vision front")
    if ("shared_head.weight" not in checkpoint["model"] or
            tuple(checkpoint["model"]["shared_head.weight"].shape) != (10, 784)):
        raise ValueError("expected exactly one ten-output Linear readout")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def validate_head(model):
    layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    if layers != [model.shared_head] or model.shared_head.bias is not None:
        raise ValueError("inference must have only one biasless Linear(784,10)")


def score_all(model, datasets, names, device, batch_size):
    return {name: evaluate(model, datasets[name], device, batch_size,
                           10 if name == "eurosat" else CLASSES[name])
            for name in names}


def make_optimizer(model, lr, head_lr_scale):
    head_parameters = list(model.shared_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    optical_parameters = [parameter for parameter in model.parameters()
                          if parameter.requires_grad and id(parameter) not in head_ids]
    parameters = optical_parameters + head_parameters
    optimizer = torch.optim.Adam([
        {"params": optical_parameters, "lr": lr},
        {"params": head_parameters, "lr": lr * head_lr_scale},
    ])
    return optimizer, parameters


def initialize_new_experts(model, path, stage, clevr_protocol, device, vision_sha=None):
    """Warm only B's four new phase masks from a standalone original-CLEVR MoE."""
    if path is None:
        return
    if stage != 2 or model.architecture != "moe":
        raise ValueError("CLEVR expert warm start applies only to MoE stage B")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    if (config.get("task") != "clevr" or config.get("architecture") != "moe" or
            config.get("activation_order") != "center_out" or
            config.get("source_protocol_sha256") != sha256_file(clevr_protocol) or
            config.get("vision_checkpoint_sha256") != vision_sha):
        raise ValueError("warm checkpoint does not match original CLEVR input")
    source = DirectCCDOptics("moe", activation_order="center_out").to(device)
    source.load_state_dict(checkpoint["model"])
    new_indices = model.active_indices[4:8].tolist()
    first_indices = model.active_indices[:4].tolist()
    with torch.no_grad():
        for destination, original in zip(new_indices, first_indices):
            model.first_phase[destination].copy_(source.first_phase[original])


def run_train(args, protocols, device):
    if args.stage not in (2, 3, 4) or args.previous_checkpoint is None:
        raise ValueError("training requires stage 2/3/4 and its previous checkpoint")
    order = ("eurosat", "clevr", "speech_binary", args.physical_task)
    names = order[:args.stage]
    train_names = names if args.replay_mode != "none" else (names[-1],)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {
        "stage": args.stage, "order": order, "architecture": args.architecture,
        "activation_order": "center_out", "readout": "one shared Linear(784,10), bias=False",
        "replay_mode": args.replay_mode,
        "replay_schedule": args.replay_schedule,
        "memory_per_old_task": args.memory_per_old_task if args.replay_mode == "memory" else None,
        "memory_cycles_per_epoch": args.memory_cycles if args.replay_mode == "memory" else None,
        "full_replay": (
            "fixed unique training records per old task, spread for a bounded number of cycles"
            if args.replay_mode == "memory" else
            "each task traversed completely; shorter tasks reshuffled and repeated"
            if args.replay_mode == "full" and args.replay_schedule == "balanced_cycle"
            else "all records of every learned task exactly once per epoch; interleaved"
            if args.replay_mode == "full" else "current task only; no replay"),
        "current_task": names[-1], "learned_tasks": names,
        "protocol_sha256": {n: sha256_file(protocols[n]) for n in names},
        "vision_checkpoint_sha256": (sha256_file(args.vision_checkpoint)
                                      if args.vision_checkpoint else None),
        "previous_checkpoint": str(args.previous_checkpoint),
        "previous_checkpoint_sha256": sha256_file(args.previous_checkpoint),
        "epochs": args.epochs, "min_epochs": args.min_epochs,
        "patience": args.patience, "batch": args.batch,
        "eval_batch": args.eval_batch, "lr": args.lr, "seed": args.seed,
        "clevr_pairwise_weight": args.clevr_pairwise_weight,
        "physical_pairwise_weight": args.physical_pairwise_weight,
        "route_balance_weight": args.route_balance_weight,
        "old_task_loss_weight": args.old_task_loss_weight,
        "current_task_loss_weight": args.current_task_loss_weight,
        "head_lr_scale": args.head_lr_scale,
        "new_expert_init_checkpoint": (str(args.new_expert_init_checkpoint)
                                       if args.new_expert_init_checkpoint else None),
        "new_expert_init_sha256": (sha256_file(args.new_expert_init_checkpoint)
                                   if args.new_expert_init_checkpoint else None),
        "test_policy": "omitted" if args.skip_test else "once after validation selection",
        "selection": "mean full-validation macro recall of all learned tasks",
        "model_git_commit": commit,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device)},
    }
    if args.resume:
        previous_config = json.loads((args.out / "config.json").read_text())
        if previous_config != json.loads(json.dumps(config)):
            raise ValueError("resume config differs from original run")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        save_json(args.out / "config.json", config)
        (args.out / "command.txt").write_text(" ".join(sys.argv) + "\n")
    save_json(args.out / "status.json", {"status": "running"})

    train = {name: load_dataset(protocols, name, "train", args.vision_checkpoint,
                               device) for name in names}
    if args.replay_mode == "memory":
        manifest = {"seed": args.seed, "split": "train", "per_old_task": args.memory_per_old_task,
                    "source_protocol_sha256": config["protocol_sha256"], "tasks": {}}
        for task_index, name in enumerate(names[:-1]):
            indices = select_memory_indices(train[name], name, args.memory_per_old_task,
                                            args.seed + task_index * 101)
            manifest["tasks"][name] = {"indices": indices.tolist(),
                                        "label_counts": np.bincount(
                                            np.asarray(train[name].labels)[indices],
                                            minlength=10 if name == "eurosat" else 2).tolist()}
            train[name] = IndexedMemory(train[name], indices)
        if args.resume:
            if json.loads((args.out / "memory_manifest.json").read_text()) != manifest:
                raise ValueError("resume memory differs from original run")
        else:
            save_json(args.out / "memory_manifest.json", manifest)
    val = {name: load_dataset(protocols, name, "val", args.vision_checkpoint,
                             device) for name in names}
    model = DirectCCDOptics(args.architecture, activation_order="center_out").to(device)
    validate_head(model)
    load_previous(model, args.previous_checkpoint, args.stage, protocols, device,
                  order=order, architecture=args.architecture,
                  vision_sha=config["vision_checkpoint_sha256"],
                  replay_mode=args.replay_mode)
    model.configure_stage(args.stage - 1)
    if not args.resume:
        initialize_new_experts(model, args.new_expert_init_checkpoint, args.stage,
                               protocols["clevr"], device,
                               config["vision_checkpoint_sha256"])
    old_indices = (model.active_indices[:4 * (args.stage - 1)].tolist()
                   if args.architecture == "moe" else [])
    old_phases = {index: model.first_phase[index].detach().cpu().clone()
                  for index in old_indices}
    optimizer, parameters = make_optimizer(model, args.lr, args.head_lr_scale)
    best_score, best_epoch, wait, start_epoch = -1.0, 0, 0, 1
    history = []
    if args.resume:
        last = torch.load(args.out / "last_checkpoint.pt", map_location=device,
                          weights_only=False)
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        history = json.loads((args.out / "history.json").read_text())
        best_score, best_epoch, wait = (float(last["best_score"]),
                                        int(last["best_epoch"]), int(last["wait"]))
        start_epoch = int(last["epoch"]) + 1
    for epoch in range(start_epoch, args.epochs + 1):
        began = time.time()
        model.train()
        loss_sum = 0.0
        steps = 0
        for batch_map in stage_epoch_batches(train, train_names, args.batch,
                                             args.seed + args.stage * 1000 + epoch,
                                             mode=args.replay_schedule,
                                             pair_physical=bool(args.physical_pairwise_weight),
                                             memory_cycles=args.memory_cycles):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            # Backpropagate each task separately to bound 1026x1026 FFT memory.
            for name in batch_map:
                indices = batch_map[name]
                field = train[name].get_batch(indices, device)
                target = torch.as_tensor(train[name].labels[indices], device=device)
                output = model(field)
                loss = task_loss(output["logits"], target, name,
                                 args.clevr_pairwise_weight,
                                 args.physical_pairwise_weight)
                if args.route_balance_weight:
                    active = model.active_indices[:int(model.active_count)]
                    loss = loss + args.route_balance_weight * routing_balance_penalty(
                        output["route_power"], active)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite {name} loss at epoch {epoch}")
                task_weight = (args.current_task_loss_weight if name == names[-1]
                               else args.old_task_loss_weight)
                (task_weight * loss / len(train_names)).backward()
                step_loss += task_weight * float(loss.detach()) / len(train_names)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            loss_sum += step_loss
            steps += 1
        validation = score_all(model, val, names, device, args.eval_batch)
        score = float(np.mean([validation[name]["balanced_accuracy"] for name in names]))
        if score > best_score + 1e-5:
            best_score, best_epoch, wait = score, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "config": config, "validation": validation},
                       args.out / "best_checkpoint.pt")
        else:
            wait += 1
        row = {"epoch": epoch, "train_loss": loss_sum / steps,
               "validation_mean": score, "validation": validation,
               "steps": steps, "seconds": time.time() - began,
               "best_epoch": best_epoch}
        history.append(row)
        save_json(args.out / "history.json", history)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_score": best_score,
                    "best_epoch": best_epoch, "wait": wait},
                   args.out / "last_checkpoint.pt")
        save_json(args.out / "status.json", {"status": "running", "epoch": epoch,
                                              "best_epoch": best_epoch,
                                              "best_val_mean": best_score})
        print(json.dumps({"stage": args.stage, "epoch": epoch,
                          "val_mean": score, "best_val_mean": best_score,
                          "seconds": row["seconds"]}), flush=True)
        if epoch >= args.min_epochs and wait >= args.patience:
            break
    selected = torch.load(args.out / "best_checkpoint.pt", map_location=device,
                          weights_only=False)
    model.load_state_dict(selected["model"])
    unchanged = all(torch.equal(model.first_phase[index].detach().cpu(), old)
                    for index, old in old_phases.items())
    if not unchanged:
        raise AssertionError("an old expert phase changed")
    result = {"stage": args.stage, "learned_tasks": names,
              "selected_epoch": best_epoch, "validation": selected["validation"],
              "validation_mean": best_score,
              "old_experts_unchanged": unchanged if args.architecture == "moe" else None,
              "shared_head": "one trainable Linear(784,10)"}
    if not args.skip_test:
        test = {name: load_dataset(protocols, name, "test", args.vision_checkpoint,
                                   device) for name in names}
        result["test"] = score_all(model, test, names, device, args.eval_batch)
    save_json(args.out / "result.json", result)
    save_json(args.out / "status.json", {"status": "complete",
                                          "epoch": history[-1]["epoch"],
                                          "best_epoch": best_epoch})
    print(json.dumps({"result": result}), flush=True)


def main(default_architecture="moe"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("moe", "d2nn"),
                        default=default_architecture)
    parser.add_argument("--physical-task", choices=("physical_binary", "physical_binary_raw"),
                        default="physical_binary_raw")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--previous-checkpoint", type=Path, required=True)
    parser.add_argument("--eurosat", type=Path, required=True)
    parser.add_argument("--clevr", type=Path, required=True)
    parser.add_argument("--speech", type=Path, required=True)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clevr-pairwise-weight", type=float, default=4.0)
    parser.add_argument("--physical-pairwise-weight", type=float, default=0.0)
    parser.add_argument("--route-balance-weight", type=float, default=0.0)
    parser.add_argument("--old-task-loss-weight", type=float, default=1.0)
    parser.add_argument("--current-task-loss-weight", type=float, default=1.0)
    parser.add_argument("--replay-mode", choices=("full", "none", "memory"), default="full")
    parser.add_argument("--memory-per-old-task", type=int, default=300)
    parser.add_argument("--memory-cycles", type=int, default=50)
    parser.add_argument("--replay-schedule", choices=("one_pass", "balanced_cycle",
                                                   "memory_spread"),
                        default="one_pass")
    parser.add_argument("--head-lr-scale", type=float, default=1.0)
    parser.add_argument("--new-expert-init-checkpoint", type=Path)
    parser.add_argument("--vision-checkpoint", type=Path)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (0 < args.min_epochs <= args.epochs and args.patience > 0 and
            args.batch > 0 and args.batch % 2 == 0 and args.eval_batch > 0 and
            args.lr > 0 and args.clevr_pairwise_weight >= 0 and
            args.physical_pairwise_weight >= 0 and
            args.route_balance_weight >= 0 and args.old_task_loss_weight > 0 and
            args.current_task_loss_weight > 0 and
            args.head_lr_scale > 0):
        raise ValueError("invalid stage training budget")
    if args.architecture == "d2nn" and (args.route_balance_weight or
                                          args.new_expert_init_checkpoint):
        raise ValueError("D2NN has no router or expert warm start")
    if args.architecture == "moe" and args.replay_mode != "full":
        raise ValueError("MoE sequential protocol requires replay")
    if args.replay_mode == "none" and args.replay_schedule != "one_pass":
        raise ValueError("no-replay schedule cannot cycle task data")
    if args.replay_mode == "memory" and (args.architecture != "d2nn" or
                                         args.replay_schedule != "memory_spread" or
                                         args.memory_per_old_task <= 0 or
                                         args.memory_per_old_task % 2 or
                                         args.memory_cycles <= 0):
        raise ValueError("D2NN memory replay requires memory_spread and a positive budget")
    if args.replay_mode != "memory" and args.replay_schedule == "memory_spread":
        raise ValueError("memory_spread requires fixed memory replay")
    if ("runs", "simulation") not in list(zip(args.out.parts, args.out.parts[1:])):
        raise ValueError("stage run must live under runs/simulation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    protocols = dict(zip(("eurosat", "clevr", "speech_binary", args.physical_task),
                         (args.eurosat, args.clevr, args.speech, args.physical)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        run_train(args, protocols, device)
    except Exception as error:
        if args.out.is_dir():
            save_json(args.out / "status.json", {"status": "failed",
                                                "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
