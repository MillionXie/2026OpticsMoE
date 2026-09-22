"""Single-task diagnostics and sequential cross-modal optical learning."""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from .data import load_task, verify_manifest
from .model import CrossModalOptics


TASK_ORDER = ("eurosat", "clevr", "speech", "physical")

FULL_DATA_REQUIREMENTS = {}


def build_model(architecture, cfg, seed, phase_dropout=None, max_experts=16):
    configured_dropout = cfg.get(
        f"{architecture}_phase_dropout", cfg.get("phase_dropout", 0.0))
    return CrossModalOptics(
        architecture, seed,
        configured_dropout if phase_dropout is None else phase_dropout,
        readout_grid=int(cfg.get("readout_grid", 16)),
        head_width=int(cfg.get("head_width", 64)),
        head_bottleneck=int(cfg.get("head_bottleneck", 0)),
        optical_layers=int(cfg.get("optical_layers", 2)),
        max_experts=max_experts,
        oeo_activation=cfg.get("oeo_activation", "intensity_softsign"),
        routing_temperature=float(cfg.get("routing_temperature", 1.0)),
    )


def task_config(cfg, key, task, default=None):
    """Resolve a task-specific value before the shared fallback."""
    return cfg.get(f"{key}_{task}", cfg.get(key, default))


def validate_full_protocol(name, protocol):
    """Reject any sampled or incomplete package at the formal training boundary."""
    if not protocol.get("all_original_samples", False):
        raise ValueError(f"{name}: formal training requires all_original_samples=true")
    for key, expected in FULL_DATA_REQUIREMENTS.get(name, {}).items():
        actual = protocol.get(key)
        if actual != expected:
            raise ValueError(f"{name}: {key}={actual!r}, expected {expected}")


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def module_sha(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TaskData:
    name: str
    classes: int
    root: Path
    splits: dict
    manifest_sha: str


def load_tasks(paths, require_full=False, names=TASK_ORDER):
    tasks = {}
    for name in names:
        root = Path(paths[name])
        manifest_sha = verify_manifest(root)
        protocol = json.loads((root / "protocol.json").read_text())
        assert protocol["task"] == name
        classes = int(protocol["classes"])
        expected_classes = {"eurosat": 10, "clevr": 2, "speech": 8, "physical": 10}
        assert classes == expected_classes[name]
        if require_full:
            validate_full_protocol(name, protocol)
        splits = {s: load_task(root, s) for s in ("train", "val", "test")}
        tasks[name] = TaskData(name, classes, root, splits, manifest_sha)
    return tasks


def classification_metrics(name, labels, probabilities, rows):
    pred = probabilities.argmax(1)
    result = {
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(recall_score(labels, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()),
        "confusion_matrix": confusion_matrix(labels, pred, labels=list(range(probabilities.shape[1]))).tolist(),
        "support": np.bincount(labels, minlength=probabilities.shape[1]).tolist(),
    }
    return result


def selection_score(name, metrics):
    return metrics["balanced_accuracy"]


def device_fields(fields, index, device):
    """Materialize a batch, using device-side expansion when the adapter supports it."""
    if hasattr(fields, "get_batch"):
        return fields.get_batch(index, device)
    return fields[index].to(device=device, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, task, split, device, batch, active_task=None):
    fields, labels, rows = task.splits[split]
    model.eval()
    probs, detector_probs, routes, capture = [], [], [], []
    if active_task is not None and model.architecture == "moe":
        model.active_count.fill_(4 * (active_task + 1))
    for start in range(0, len(labels), batch):
        out = model(device_fields(fields, slice(start, start + batch), device), task.name)
        probs.append(out["probabilities"].cpu().numpy())
        detector_probs.append(out["detector_probabilities"].cpu().numpy())
        capture.append(out["capture"].cpu().numpy())
        if out["route_power"] is not None:
            routes.append(out["route_power"].cpu().numpy())
    p = np.concatenate(probs)
    detector_p = np.concatenate(detector_probs)
    q = np.concatenate(routes) if routes else None
    result = classification_metrics(task.name, labels, p, rows)
    detector_metrics = classification_metrics(task.name, labels, detector_p, rows)
    result["detector_accuracy"] = detector_metrics["accuracy"]
    result["detector_balanced_accuracy"] = detector_metrics["balanced_accuracy"]
    result["detector_nll"] = detector_metrics["nll"]
    result["zero_readout_fraction"] = float((np.concatenate(capture) <= 1e-12).mean())
    result["route_mean"] = q.mean(0).tolist() if q is not None else None
    result["route_top_frequency"] = (np.bincount(q.argmax(1), minlength=model.max_experts) / len(q)).tolist() if q is not None else None
    return result, p, q


@torch.no_grad()
def evaluate_indices(model, task, indices, device, batch, active_task=None):
    """Evaluate a declared subset without changing the dataset split."""
    fields, labels, rows = task.splits["train"]
    indices = np.asarray(indices, dtype=np.int64)
    model.eval()
    probs = []
    if active_task is not None and model.architecture == "moe":
        model.active_count.fill_(4 * (active_task + 1))
    for start in range(0, len(indices), batch):
        ix = indices[start:start + batch]
        out = model(device_fields(fields, ix, device), task.name)
        probs.append(out["probabilities"].cpu().numpy())
    p = np.concatenate(probs)
    return classification_metrics(task.name, labels[indices], p, [rows[i] for i in indices])


def sample_weights(task, indices):
    return torch.ones(len(indices))


def smoothed_nll(log_probabilities, labels, smoothing):
    nll = F.nll_loss(log_probabilities, labels, reduction="none")
    if smoothing:
        nll = (1.0 - smoothing) * nll - smoothing * log_probabilities.mean(1)
    return nll


def task_loss(model, task, indices, device, warmup=False, balance=0.0,
              route_entropy=0.0, augment=False, label_smoothing=0.0,
              detector_aux_weight=0.0):
    fields, labels, _ = task.splits["train"]
    batch_fields = device_fields(fields, indices, device)
    # No task-specific augmentation is applied in the audited four-task protocol.
    out = model(batch_fields, task.name, warmup=warmup)
    y = torch.as_tensor(labels[indices], device=device)
    w = sample_weights(task, indices).to(device)
    log_probabilities = out["probabilities"].clamp_min(1e-12).log()
    loss = (smoothed_nll(log_probabilities, y, label_smoothing) * w).mean()
    if detector_aux_weight:
        detector_log = out["detector_probabilities"].clamp_min(1e-12).log()
        detector_loss = (smoothed_nll(detector_log, y, label_smoothing) * w).mean()
        loss = loss + detector_aux_weight * detector_loss
    if balance and out["route_power"] is not None and not warmup:
        n = int(model.active_count)
        loss = loss + balance * (out["route_power"][:, :n].mean(0) - 1.0/n).square().sum()
    if route_entropy and out["route_power"] is not None and not warmup:
        n = int(model.active_count)
        q = out["route_power"][:, :n].clamp_min(1e-12)
        loss = loss + route_entropy * (-(q * q.log()).sum(1).mean())
    return loss


def configured_task_loss(model, task, indices, device, cfg, **kwargs):
    key = f"{model.architecture}_detector_aux_weight"
    detector_aux_weight = float(task_config(cfg, key, task.name,
                                            cfg.get("detector_aux_weight", 0.0)))
    return task_loss(model, task, indices, device,
                     augment=False,
                     label_smoothing=float(cfg.get("label_smoothing", 0.0)),
                     detector_aux_weight=detector_aux_weight,
                     **kwargs)


def replay_indices(task, budget, seed):
    labels = task.splits["train"][1]
    rng = np.random.default_rng(seed)
    groups = []
    keys = [int(y) for y in labels]
    unique = sorted(set(keys), key=str)
    for key in unique:
        ids = np.flatnonzero(np.array([k == key for k in keys]))
        rng.shuffle(ids)
        groups.append(ids)
    chosen = []
    while len(chosen) < min(budget, len(labels)) and any(len(g) for g in groups):
        for i, group in enumerate(groups):
            if len(chosen) >= budget:
                break
            if len(group):
                chosen.append(int(group[0])); groups[i] = group[1:]
    return np.array(chosen, dtype=np.int64)


def chunks(indices, size):
    return [indices[i:i+size] for i in range(0, len(indices), size)]


def combine_current_replay(current_loss, replay_losses, replay_weight):
    """Keep current-task weight stable as the number of old tasks grows."""
    if not replay_losses:
        return current_loss
    replay_loss = torch.stack(replay_losses).mean()
    return (current_loss + replay_weight * replay_loss) / (1.0 + replay_weight)


def add_continual_metrics(result, all_history):
    learned={row["task"]:selection_score(row["task"],row["val"][row["task"]]) for row in all_history}
    final={name:selection_score(name,result["validation"][name]) for name in TASK_ORDER}
    result["continual"]={"score_when_learned":learned,"final_validation_score":final,
                         "backward_transfer":{name:final[name]-learned[name] for name in TASK_ORDER[:-1]}}
    result["continual"]["mean_backward_transfer"]=float(np.mean(list(result["continual"]["backward_transfer"].values())))
    return result


def stage_evaluation(model, tasks, learned_names, device, cfg, active_task=None):
    """Evaluate a selected stage checkpoint without using test data for selection."""
    return {
        split: {name: evaluate(model, tasks[name], split, device, cfg["eval_batch"], active_task)[0]
                for name in learned_names}
        for split in ("val", "test")
    }


@torch.no_grad()
def extract_ccd_features(model, task, split, device, batch):
    fields, labels, rows = task.splits[split]
    model.eval(); features=[]
    for start in range(0,len(labels),batch):
        out=model(device_fields(fields, slice(start, start + batch), device),task.name)
        features.append(out["ccd_features"].cpu())
    return torch.cat(features),np.asarray(labels),rows


def fit_mlp_head(model, task, cfg, device, seed, splits=("train", "val", "test")):
    """Fit and validation-select an electronic head on frozen CCD features."""
    # Test fields are not propagated until the validation-selected head is fixed.
    cached={split:extract_ccd_features(model,task,split,device,cfg["eval_batch"])
            for split in splits if split != "test"}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head=model.make_head(task.classes).to(device)
    train_x,train_y,_=cached["train"]
    train_x=train_x.to(device)
    train_y_t=torch.as_tensor(np.array(train_y, copy=True), device=device)
    weights=sample_weights(task,np.arange(len(train_y))).to(device)
    epochs=int(cfg.get("probe_epochs",50));batch=int(cfg.get("probe_batch",256))
    optimizer=torch.optim.Adam(head.parameters(),lr=cfg.get("probe_lr",cfg["lr"]))
    best=-float("inf");best_state=None;best_epoch=None
    for epoch in range(1,epochs+1):
        head.train();order=np.random.default_rng(seed+epoch).permutation(len(train_y))
        for ix in chunks(order,batch):
            ix=torch.as_tensor(ix,device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=head(train_x[ix]);loss=(F.cross_entropy(logits,train_y_t[ix],reduction="none")*weights[ix]).mean()
            loss.backward();optimizer.step()
        head.eval();val_x,val_y,val_rows=cached["val"]
        with torch.no_grad():p=head(val_x.to(device)).softmax(1).cpu().numpy()
        score=selection_score(task.name,classification_metrics(task.name,val_y,p,val_rows))
        if score>best:
            best=score;best_epoch=epoch;best_state=copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state);head.eval();metrics={}
    if "test" in splits:
        cached["test"] = extract_ccd_features(model,task,"test",device,cfg["eval_batch"])
    for split,(x,y,rows) in cached.items():
        with torch.no_grad():p=head(x.to(device)).softmax(1).cpu().numpy()
        metrics[split]=classification_metrics(task.name,y,p,rows)
    return head, {"selected_epoch":best_epoch,"metrics":metrics}


def fit_mlp_probe(model, task, cfg, device, seed):
    """Report an MLP transfer probe without changing the supplied model."""
    return fit_mlp_head(model, task, cfg, device, seed)[1]


def calibrate_head(model, task, cfg, device, seed):
    """Refit the same one-layer readout and accept it only on validation gain.

    The optical parameters stay frozen.  This routine never adds another
    electronic layer and never consults the test split when choosing whether
    to keep the refitted Linear head.
    """
    original = copy.deepcopy(model.heads[task.name].state_dict())
    before = evaluate(model, task, "val", device, cfg["eval_batch"])[0]
    before_score = selection_score(task.name, before)
    head, result = fit_mlp_head(model, task, cfg, device, seed, ("train", "val"))
    model.heads[task.name].load_state_dict(head.state_dict())
    after = evaluate(model, task, "val", device, cfg["eval_batch"])[0]
    after_score = selection_score(task.name, after)
    accepted = after_score >= before_score
    if not accepted:
        model.heads[task.name].load_state_dict(original)
    result.update({
        "accepted": accepted,
        "validation_score_before": before_score,
        "validation_score_after": after_score,
        "selection_rule": "accept_refit_only_if_validation_score_does_not_decrease",
    })
    return result


def save_continual_matrix(root, all_history):
    payload = {"task_order": list(TASK_ORDER), "metric": {
        "eurosat": "balanced_accuracy", "clevr": "balanced_accuracy",
        "speech": "balanced_accuracy", "physical": "balanced_accuracy"}}
    for split in ("validation", "test"):
        history_split = "val" if split == "validation" else split
        payload[split] = [
            {name: selection_score(name, row[history_split][name]) for name in TASK_ORDER[:i + 1]}
            for i, row in enumerate(all_history)
        ]
    final=payload["test"][-1]
    old_tasks=TASK_ORDER[:max(0,len(all_history)-1)]
    backward={name:final[name]-payload["test"][i][name] for i,name in enumerate(old_tasks)}
    forgetting={name:max(row[name] for row in payload["test"][i:] if name in row)-final[name]
                for i,name in enumerate(old_tasks)}
    payload["continual"]={"backward_transfer":backward,
                           "mean_backward_transfer":float(np.mean(list(backward.values()))) if backward else None,
                           "forgetting":forgetting,
                           "mean_forgetting":float(np.mean(list(forgetting.values()))) if forgetting else None}
    save(root / "continual_matrix.json", payload)
    return payload


def train_single_task_d2nn(tasks, cfg, out, device, selected_task=None):
    """Train a plain independent D2NN with the same one-layer readout."""
    root = out / "single_task_d2nn"; root.mkdir()
    summary = {}
    epochs = int(cfg.get("single_task_epochs", cfg["task_epochs"]))
    names = (selected_task,) if selected_task else TASK_ORDER
    for name in names:
        task_index = TASK_ORDER.index(name)
        task_root = root / name; task_root.mkdir()
        seed_all(cfg["seed"] + task_index)
        model = build_model(
            "d2nn", cfg, cfg["seed"] + task_index,
            max_experts=int(cfg.get("single_task_d2nn_max_experts", 4))).to(device)
        model.configure_task(task_index, warmup=False)
        lr = float(cfg.get("d2nn_lr", cfg["lr"]))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr * .1)
        best, history = -float("inf"), []
        for epoch in range(1, epochs + 1):
            model.train(); started=time.time(); losses=[]
            order=np.random.default_rng(cfg["seed"]+task_index*1000+epoch).permutation(len(tasks[name].splits["train"][1]))
            for ix in chunks(order, cfg["batch"]):
                optimizer.zero_grad(set_to_none=True)
                loss=configured_task_loss(model,tasks[name],ix,device,cfg)
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            val=evaluate(model,tasks[name],"val",device,cfg["eval_batch"])[0]
            score=selection_score(name,val)
            # Baseline selection uses only the declared final readout. The
            # training-only detector must not tune or select the D2NN.
            phase_score=score
            row={"epoch":epoch,"loss":float(np.mean(losses)),"validation_score":score,
                 "optical_selection_score":phase_score,
                 "validation":val,"seconds":time.time()-started}
            history.append(row);save(task_root/"history.json",history)
            cp={"model":model.state_dict(),"epoch":epoch,"task":name,"validation":val,
                "score":phase_score,"mlp_score":score}
            torch.save(cp,task_root/"last_checkpoint.pt")
            if phase_score>best:best=phase_score;torch.save(cp,task_root/"best_checkpoint.pt")
            print(json.dumps({"arch":"single_task_d2nn","task":name,"epoch":epoch,
                              "val":score,"detector_val":val["detector_balanced_accuracy"],
                              "loss":row["loss"],"seconds":row["seconds"]}),flush=True)
        cp=torch.load(task_root/"best_checkpoint.pt",map_location=device,weights_only=False);model.load_state_dict(cp["model"])
        metrics={split:evaluate(model,tasks[name],split,device,cfg["eval_batch"])[0]
                 for split in ("train","val","test")}
        score = selection_score(name, metrics["val"])
        summary[name]={"selected_epoch":cp["epoch"],"head_calibration":None,
                       "electronic_readout":"single_linear_layer",
                       "validation_selection_score":score,
                       "admission_threshold":float(cfg.get("d2nn_admission_threshold", .65)),
                       "admission_passed":score >= float(cfg.get("d2nn_admission_threshold", .65)),
                       "metrics":metrics}
        save(task_root/"results.json",summary[name])
    save(root/"results.json",summary)
    if selected_task:
        return {"single_task": summary}
    probes={}
    score_matrix={}
    for source_index,source in enumerate(TASK_ORDER):
        model=build_model("d2nn",cfg,cfg["seed"]+source_index,max_experts=4).to(device)
        cp=torch.load(root/source/"best_checkpoint.pt",map_location=device,weights_only=False)
        model.load_state_dict(cp["model"])
        model.requires_grad_(False)
        probes[source]={};score_matrix[source]={}
        for target_index,target in enumerate(TASK_ORDER):
            probe=fit_mlp_probe(model,tasks[target],cfg,device,
                                cfg["seed"]+source_index*100+target_index)
            probes[source][target]=probe
            score_matrix[source][target]=selection_score(target,probe["metrics"]["test"])
            print(json.dumps({"arch":"frozen_optics_mlp_probe","source":source,
                              "target":target,"test":score_matrix[source][target]}),flush=True)
    save(root/"frozen_optics_probe.json",{"task_order":list(TASK_ORDER),
         "score_matrix":score_matrix,"details":probes})
    return {"single_task":summary,"frozen_optics_probe":score_matrix}


def train_single_task_moe(tasks, cfg, out, device, selected_task=None, init_checkpoint=None):
    """Train a single-task MoE, optionally in the fixed 16-slot geometry."""
    root = out / "single_task_moe"; root.mkdir()
    summary = {}
    epochs = int(cfg.get("single_task_moe_epochs",
                         cfg.get("single_task_epochs", cfg.get("task_epochs", 20))))
    names = (selected_task,) if selected_task else TASK_ORDER
    for name in names:
        task_index = TASK_ORDER.index(name)
        task_root = root / name; task_root.mkdir()
        seed_all(cfg["seed"] + task_index)
        max_experts = int(cfg.get("single_task_moe_max_experts", 4))
        model = build_model("moe", cfg, cfg["seed"] + task_index,
                            max_experts=max_experts).to(device)
        model.configure_single_task(name)
        initial_epoch = 0
        if init_checkpoint is not None:
            initial = torch.load(init_checkpoint, map_location=device, weights_only=False)
            if initial.get("task", name) != name:
                raise ValueError(f"initial checkpoint task {initial.get('task')} != {name}")
            model.load_state_dict(initial["model"])
            initial_epoch = int(initial["epoch"])
            if initial_epoch >= epochs:
                raise ValueError(f"initial epoch {initial_epoch} must be below target {epochs}")
        lr = float(cfg.get("moe_lr", cfg["lr"]))
        weight_decay = float(cfg.get("moe_weight_decay", 0.0))
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr,
            weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs - initial_epoch, eta_min=lr * .1)
        best, history = -float("inf"), []
        ema_decay = float(cfg.get("moe_ema_decay", 0.0))
        ema = ({key: parameter.detach().clone()
                for key, parameter in model.named_parameters() if parameter.requires_grad}
               if ema_decay else None)
        if initial_epoch:
            val=evaluate(model,tasks[name],"val",device,cfg["eval_batch"])[0]
            best=selection_score(name,val)
            cp={"model":model.state_dict(),"epoch":initial_epoch,"task":name,
                "validation":val,"score":best,"mlp_score":best,
                "initialized_from":str(init_checkpoint)}
            torch.save(cp,task_root/"best_checkpoint.pt")
            history.append({"epoch":initial_epoch,"validation_score":best,
                            "optical_selection_score":best,"validation":val,
                            "initialized_from":str(init_checkpoint)})
        warmup_epochs = int(cfg.get("single_task_moe_warmup_epochs", 0))
        route_balance = float(task_config(cfg, "route_balance", name, 0.0))
        route_entropy = float(task_config(cfg, "route_entropy", name, 0.0))
        for epoch in range(initial_epoch + 1, epochs + 1):
            model.train(); started=time.time(); losses=[]
            warmup = initial_epoch == 0 and epoch <= warmup_epochs
            order=np.random.default_rng(cfg["seed"]+task_index*1000+epoch).permutation(
                len(tasks[name].splits["train"][1]))
            for ix in chunks(order, cfg["batch"]):
                optimizer.zero_grad(set_to_none=True)
                loss=configured_task_loss(model,tasks[name],ix,device,cfg,
                                          warmup=warmup,
                                          balance=0.0 if warmup else route_balance,
                                          route_entropy=0.0 if warmup else route_entropy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],1.0)
                optimizer.step(); losses.append(float(loss.detach()))
                if ema is not None:
                    with torch.no_grad():
                        for key, parameter in model.named_parameters():
                            if key in ema:
                                ema[key].mul_(ema_decay).add_(parameter, alpha=1.0-ema_decay)
            scheduler.step()
            val=evaluate(model,tasks[name],"val",device,cfg["eval_batch"])[0]
            score=selection_score(name,val)
            selected_state = copy.deepcopy(model.state_dict())
            selected_variant = "raw"
            ema_score = None
            if ema is not None:
                raw = {key: parameter.detach().clone()
                       for key, parameter in model.named_parameters() if key in ema}
                with torch.no_grad():
                    for key, parameter in model.named_parameters():
                        if key in ema:
                            parameter.copy_(ema[key])
                ema_val=evaluate(model,tasks[name],"val",device,cfg["eval_batch"])[0]
                ema_score=selection_score(name,ema_val)
                if ema_score > score:
                    val, score = ema_val, ema_score
                    selected_state = copy.deepcopy(model.state_dict())
                    selected_variant = "ema"
                with torch.no_grad():
                    for key, parameter in model.named_parameters():
                        if key in raw:
                            parameter.copy_(raw[key])
            # The declared model output is the single Linear readout.  The
            # auxiliary optical detector may shape training, but it cannot
            # select the reported checkpoint.
            phase_score=score
            row={"epoch":epoch,"loss":float(np.mean(losses)),"validation_score":score,
                 "optical_selection_score":phase_score,
                 "selected_variant":selected_variant,
                 "ema_validation_score":ema_score,
                 "uniform_route_warmup":warmup,
                 "validation":val,"seconds":time.time()-started}
            history.append(row);save(task_root/"history.json",history)
            cp={"model":selected_state,"epoch":epoch,"task":name,
                "validation":val,"score":phase_score,"mlp_score":score}
            torch.save(cp,task_root/"last_checkpoint.pt")
            if phase_score>best:
                best=phase_score;torch.save(cp,task_root/"best_checkpoint.pt")
            print(json.dumps({"arch":"single_task_moe","task":name,"epoch":epoch,
                              "val":score,"detector_val":val["detector_balanced_accuracy"],
                              "loss":row["loss"],
                              "route":val["route_mean"],"seconds":row["seconds"]}),flush=True)
        cp=torch.load(task_root/"best_checkpoint.pt",map_location=device,weights_only=False)
        model.load_state_dict(cp["model"])
        calibration = None
        if cfg.get("moe_head_refit", True):
            # The optical checkpoint is already fixed. Refit the same single
            # Linear CCD readout; this changes training schedule, not capacity.
            calibration=calibrate_head(model,tasks[name],cfg,device,
                                       cfg["seed"]+1000+task_index)
            torch.save({**cp,"model":model.state_dict(),"head_calibration":calibration},
                       task_root/"refit_checkpoint.pt")
        evaluation_splits = (("train", "val") if cfg.get("validation_only", False)
                             else ("train", "val", "test"))
        metrics={split:evaluate(model,tasks[name],split,device,cfg["eval_batch"])[0]
                 for split in evaluation_splits}
        score = selection_score(name, metrics["val"])
        summary[name]={"selected_epoch":cp["epoch"],"active_experts":4,
                       "max_experts":max_experts,
                       "head_calibration":calibration,
                       "electronic_readout":"single_linear_layer",
                       "validation_selection_score":score,
                       "admission_threshold":float(cfg.get("moe_admission_threshold", .70)),
                       "admission_passed":score >= float(cfg.get("moe_admission_threshold", .70)),
                       "metrics":metrics}
        save(task_root/"results.json",summary[name])
    save(root/"results.json",summary)
    return summary


def overfit_diagnostics(tasks, cfg, out, device):
    """Verify that each task can be memorized before any full-data run."""
    root=out/"overfit_diagnostics";root.mkdir()
    budget=int(cfg.get("overfit_samples",64));steps=int(cfg.get("overfit_steps",200))
    result={}
    for task_index,name in enumerate(TASK_ORDER):
        seed_all(cfg["seed"]+task_index)
        model=build_model("d2nn",cfg,cfg["seed"]+task_index,phase_dropout=0,max_experts=4).to(device)
        model.configure_task(task_index,warmup=False)
        indices=replay_indices(tasks[name],budget,cfg["seed"]+task_index)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg.get("overfit_lr",cfg["lr"]))
        rng=np.random.default_rng(cfg["seed"]+task_index)
        losses=[]
        for step in range(steps):
            ix=rng.choice(indices,size=min(cfg["batch"],len(indices)),replace=True)
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=task_loss(model,tasks[name],ix,device);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step()
            losses.append(float(loss.detach()))
        metrics=evaluate_indices(model,tasks[name],indices,device,cfg["eval_batch"])
        result[name]={"samples":len(indices),"steps":steps,"final_loss":losses[-1],
                      "selection_score":selection_score(name,metrics),"metrics":metrics}
        print(json.dumps({"arch":"overfit_d2nn","task":name,**result[name]}),flush=True)
    save(root/"results.json",result)
    return result


def restore_completed_stages(model, root, use_replay=False, moe=False):
    """Restore the last fully evaluated stage; incomplete stages are rerun."""
    history = []
    replay = {}
    for task_index, name in enumerate(TASK_ORDER):
        task_root = root / f"stage_{task_index + 1}_{name}"
        result_path = task_root / "stage_result.json"
        checkpoint_path = task_root / ("refit_checkpoint.pt" if moe else "best_checkpoint.pt")
        if moe and not checkpoint_path.exists():
            checkpoint_path = task_root / "best_checkpoint.pt"
        if not result_path.exists() or not checkpoint_path.exists():
            break
        stage = json.loads(result_path.read_text())
        checkpoint = torch.load(checkpoint_path, map_location=next(model.parameters()).device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model"])
        history.append({"task": name, "selected_epoch": stage["selected_epoch"],
                        "val": stage["val"], "test": stage["test"]})
        if use_replay:
            replay[name] = None
    return len(history), history, replay


def train_sequential_d2nn(tasks, cfg, out, device, use_replay, resume=False):
    """Sequential D2NN with optional replay and a stage-by-task score matrix."""
    tag = "replay" if use_replay else "no_replay"
    root = out / f"sequential_d2nn_{tag}"; root.mkdir(exist_ok=resume)
    seed_all(cfg["seed"])
    model = build_model("d2nn", cfg, cfg["seed"],
                        max_experts=int(cfg.get("d2nn_max_experts", 16))).to(device)
    replay = {}; all_history = []
    start_index = 0
    if resume:
        start_index, all_history, restored_replay = restore_completed_stages(
            model, root, use_replay=use_replay)
        if use_replay:
            replay = {name: replay_indices(tasks[name], cfg["replay_per_task"],
                                           cfg["seed"] + task_index)
                      for task_index, name in enumerate(TASK_ORDER[:start_index])}
    for task_index, name in enumerate(TASK_ORDER):
        if task_index < start_index:
            continue
        task_root = root / f"stage_{task_index+1}_{name}"; task_root.mkdir(exist_ok=resume)
        task = tasks[name]
        old_head_hash = {n:module_sha(model.heads[n]) for n in TASK_ORDER[:task_index]}
        shared_before = {"first_phase":state_sha(model.first_phase),
                         "global_phase":state_sha(model.global_phase),
                         **{f"additional_phase_{i}":state_sha(p)
                            for i,p in enumerate(model.additional_phases)}}
        model.configure_task(task_index, warmup=False)
        lr=float(cfg.get("d2nn_lr",cfg["lr"]))
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg["task_epochs"],eta_min=lr*.1)
        best=-float("inf"); history=[]
        for epoch in range(1,cfg["task_epochs"]+1):
            model.train();started=time.time();losses=[]
            order=np.random.default_rng(cfg["seed"]+task_index*1000+epoch).permutation(len(task.splits["train"][1]))
            current_chunks=chunks(order,cfg["batch"])
            replay_chunks={old:chunks(np.random.default_rng(cfg["seed"]+task_index*1000+epoch+j).permutation(ix),cfg["replay_batch"])
                           for j,(old,ix) in enumerate(replay.items())} if use_replay else {}
            for step,current in enumerate(current_chunks):
                optimizer.zero_grad(set_to_none=True)
                current_loss=configured_task_loss(model,task,current,device,cfg)
                replay_losses=[]
                for old in TASK_ORDER[:task_index] if use_replay else ():
                    pool=replay_chunks[old]; ix=pool[step%len(pool)]
                    replay_losses.append(configured_task_loss(model,tasks[old],ix,device,cfg))
                loss=combine_current_replay(current_loss,replay_losses,cfg["replay_weight"])
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step()
            val={n:evaluate(model,tasks[n],"val",device,cfg["eval_batch"])[0] for n in TASK_ORDER[:task_index+1]}
            score=float(np.mean([selection_score(n,val[n]) for n in val]))
            phase_score=score
            row={"epoch":epoch,"loss":float(np.mean(losses)),"validation_mean_score":score,
                 "optical_selection_score":phase_score,"validation":val,
                 "seconds":time.time()-started}
            history.append(row);save(task_root/"history.json",history)
            cp={"model":model.state_dict(),"epoch":epoch,"task_index":task_index,
                "validation":val,"score":phase_score,"mlp_score":score}
            torch.save(cp,task_root/"last_checkpoint.pt")
            if phase_score>best:best=phase_score;torch.save(cp,task_root/"best_checkpoint.pt")
            print(json.dumps({"arch":f"sequential_d2nn_{tag}","task":name,"epoch":epoch,"val":score,"loss":row["loss"],"seconds":row["seconds"]}),flush=True)
        cp=torch.load(task_root/"best_checkpoint.pt",map_location=device,weights_only=False);model.load_state_dict(cp["model"])
        after_heads={n:module_sha(model.heads[n]) for n in TASK_ORDER[:task_index]}
        assert old_head_hash==after_heads, "frozen old task head changed"
        stage_eval=stage_evaluation(model,tasks,TASK_ORDER[:task_index+1],device,cfg)
        shared_after={"first_phase":state_sha(model.first_phase),"global_phase":state_sha(model.global_phase),
                      **{f"additional_phase_{i}":state_sha(p)
                         for i,p in enumerate(model.additional_phases)}}
        save(task_root/"stage_result.json",{"selected_epoch":cp["epoch"],**stage_eval,
             "head_calibration":None,
             "electronic_readout":"single_linear_layer",
             "old_heads_unchanged":old_head_hash==after_heads,
             "shared_phases_changed":{k:shared_before[k]!=shared_after[k] for k in shared_before}})
        all_history.append({"task":name,"selected_epoch":cp["epoch"],**stage_eval})
        if use_replay:
            replay[name]=replay_indices(task,cfg["replay_per_task"],cfg["seed"]+task_index)
    save(root/"sequence.json",all_history)
    save_continual_matrix(root,all_history)
    result=finalize(model,tasks,root,device,cfg,all_history[-1]["selected_epoch"],f"sequential_{tag}")
    add_continual_metrics(result,all_history);save(root/"results.json",result)
    return result


def train_lifelong_moe(tasks, cfg, out, device, resume=False):
    root = out / "lifelong_moe"; root.mkdir(exist_ok=resume)
    seed_all(cfg["seed"])
    model = build_model("moe", cfg, cfg["seed"]).to(device)
    replay = {}; all_history = []
    start_index = 0
    if resume:
        start_index, all_history, _ = restore_completed_stages(
            model, root, use_replay=True, moe=True)
        replay = {name: replay_indices(tasks[name], cfg["replay_per_task"],
                                       cfg["seed"] + task_index)
                  for task_index, name in enumerate(TASK_ORDER[:start_index])}
    for task_index, name in enumerate(TASK_ORDER):
        if task_index < start_index:
            continue
        task_root = root / f"stage_{task_index+1}_{name}"; task_root.mkdir(exist_ok=resume)
        task = tasks[name]
        old_hash = [state_sha(p) for p in model.first_phase[:4*task_index]]
        old_head_hash = {n: module_sha(model.heads[n]) for n in TASK_ORDER[:task_index]}
        if task_index:
            model.configure_task(task_index, warmup=True)
            lr=float(cfg.get("moe_lr",cfg["lr"]))
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
            for epoch in range(1, cfg["warmup_epochs"] + 1):
                model.train(); losses=[]
                order=np.random.default_rng(cfg["seed"]+task_index*100+epoch).permutation(len(task.splits["train"][1]))
                for ix in chunks(order, cfg["batch"]):
                    optimizer.zero_grad(set_to_none=True); loss=configured_task_loss(model,task,ix,device,cfg,warmup=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();losses.append(float(loss.detach()))
                print(json.dumps({"arch":"moe","task":name,"stage":"warmup","epoch":epoch,"loss":float(np.mean(losses))}),flush=True)
        model.configure_task(task_index, warmup=False)
        lr=float(cfg.get("moe_lr",cfg["lr"]))
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg["task_epochs"],eta_min=lr*.1)
        best=-float("inf"); history=[]
        for epoch in range(1,cfg["task_epochs"]+1):
            model.train();started=time.time();losses=[]
            order=np.random.default_rng(cfg["seed"]+task_index*1000+epoch).permutation(len(task.splits["train"][1]))
            current_chunks=chunks(order,cfg["batch"])
            replay_chunks={old:chunks(np.random.default_rng(cfg["seed"]+task_index*1000+epoch+j).permutation(ix),cfg["replay_batch"])
                           for j,(old,ix) in enumerate(replay.items())}
            for step,current in enumerate(current_chunks):
                optimizer.zero_grad(set_to_none=True)
                route_balance=float(task_config(cfg,"route_balance",name,0.0))
                route_entropy=float(task_config(cfg,"route_entropy",name,0.0))
                current_loss=configured_task_loss(model,task,current,device,cfg,
                                                  balance=route_balance,
                                                  route_entropy=route_entropy)
                replay_losses=[]
                for old in TASK_ORDER[:task_index]:
                    pool=replay_chunks[old]; ix=pool[step%len(pool)]
                    old_balance=float(task_config(cfg,"route_balance",old,0.0))
                    old_entropy=float(task_config(cfg,"route_entropy",old,0.0))
                    replay_losses.append(configured_task_loss(
                        model,tasks[old],ix,device,cfg,balance=old_balance,
                        route_entropy=old_entropy))
                loss=combine_current_replay(current_loss,replay_losses,cfg["replay_weight"])
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step()
            val={n:evaluate(model,tasks[n],"val",device,cfg["eval_batch"],task_index)[0] for n in TASK_ORDER[:task_index+1]}
            score=float(np.mean([selection_score(n,val[n]) for n in val]))
            # Select lifelong checkpoints with the declared one-layer
            # electronic output only; detector scores are diagnostics.
            phase_score=score
            row={"epoch":epoch,"loss":float(np.mean(losses)),"validation_mean_score":score,
                 "optical_selection_score":phase_score,"validation":val,
                 "seconds":time.time()-started}
            history.append(row);save(task_root/"history.json",history)
            cp={"model":model.state_dict(),"epoch":epoch,"task_index":task_index,
                "validation":val,"score":phase_score,"mlp_score":score}
            torch.save(cp,task_root/"last_checkpoint.pt")
            if phase_score>best:best=phase_score;torch.save(cp,task_root/"best_checkpoint.pt")
            print(json.dumps({"arch":"moe","task":name,"epoch":epoch,"val":score,"loss":row["loss"],"route":val[name]["route_mean"],"seconds":row["seconds"]}),flush=True)
        cp=torch.load(task_root/"best_checkpoint.pt",map_location=device,weights_only=False);model.load_state_dict(cp["model"])
        calibration = None
        if cfg.get("moe_head_refit", True):
            calibration=calibrate_head(model,task,cfg,device,cfg["seed"]+1000+task_index)
            torch.save({**cp,"model":model.state_dict(),"head_calibration":calibration},
                       task_root/"refit_checkpoint.pt")
        after=[state_sha(p) for p in model.first_phase[:4*task_index]]
        after_heads={n:module_sha(model.heads[n]) for n in TASK_ORDER[:task_index]}
        assert old_hash==after, "frozen old expert changed"
        assert old_head_hash==after_heads, "frozen old task head changed"
        stage_eval=stage_evaluation(model,tasks,TASK_ORDER[:task_index+1],device,cfg,task_index)
        save(task_root/"stage_result.json",{"selected_epoch":cp["epoch"],**stage_eval,
             "head_calibration":calibration,
             "electronic_readout":"single_linear_layer",
             "old_experts_unchanged":old_hash==after,"old_heads_unchanged":old_head_hash==after_heads})
        all_history.append({"task":name,"selected_epoch":cp["epoch"],**stage_eval})
        replay[name]=replay_indices(task,cfg["replay_per_task"],cfg["seed"]+task_index)
        model.configure_task(task_index, warmup=False)
    save(root/"sequence.json",all_history)
    save_continual_matrix(root,all_history)
    result=finalize(model,tasks,root,device,cfg,all_history[-1]["selected_epoch"],"sequential_lifelong")
    add_continual_metrics(result,all_history);save(root/"results.json",result)
    return result


def finalize(model,tasks,root,device,cfg,epoch,training):
    results={"training":training,"selected_epoch":epoch,"validation":{},"test":{}}
    task_index=3 if model.architecture=="moe" else None
    for name in TASK_ORDER:
        for split in ("val","test"):
            metrics,p,q=evaluate(model,tasks[name],split,device,cfg["eval_batch"],task_index)
            results["validation" if split=="val" else "test"][name]=metrics
            np.savez_compressed(root/f"{split}_{name}_predictions.npz",probabilities=p,labels=tasks[name].splits[split][1],route=q if q is not None else np.empty((len(p),0)))
    results["mean_test_selection_metric"]=float(np.mean([selection_score(n,results["test"][n]) for n in TASK_ORDER]))
    save(root/"results.json",results)
    return results


def smoke(tasks,cfg,out,device):
    records={}
    for arch in ("moe","d2nn"):
        model=build_model(arch,cfg,cfg["seed"],phase_dropout=0).to(device)
        task_records={}
        for task_index,name in enumerate(TASK_ORDER):
            model.configure_task(task_index);model.train();model.zero_grad(set_to_none=True)
            loss=task_loss(model,tasks[name],np.arange(1),device,balance=.1);loss.backward()
            grads={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
            assert grads and all(np.isfinite(list(grads.values())))
            task_records[name]={"loss":float(loss),"active_experts":4*(task_index+1) if arch=="moe" else None,"gradients":grads}
        records[arch]={"parameters":sum(p.numel() for p in model.parameters()),"optical_parameters":sum(p.numel() for n,p in model.named_parameters() if not n.startswith("heads.")),"electronic_head_parameters":sum(p.numel() for p in model.heads.parameters()),"shape":[model.height,model.width],"tasks":task_records}
    save(out/"smoke.json",{"status":"pass","records":records})


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--eurosat",type=Path);p.add_argument("--clevr",type=Path);p.add_argument("--speech",type=Path);p.add_argument("--physical",type=Path);p.add_argument("--out",type=Path,required=True);p.add_argument("--phase",choices=["smoke","overfit","train"],default="train");p.add_argument("--only",choices=["all","single_task","single_task_moe","sequential_d2nn","sequential_d2nn_replay","moe"],default="all");p.add_argument("--single-task-name",choices=TASK_ORDER);p.add_argument("--init-checkpoint",type=Path,help="continue one single-task MoE run up to the configured target epoch");p.add_argument("--resume",action="store_true",help="restore completed sequential stages and rerun the first incomplete stage");a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=a.resume);save(a.out/"status.json",{"status":"running","pid":os.getpid(),"resume":a.resume})
    try:
        if a.single_task_name and not (a.phase == "train" and a.only in ("single_task", "single_task_moe")):
            raise ValueError("--single-task-name requires formal single_task or single_task_moe training")
        if a.init_checkpoint and not (a.phase == "train" and a.only == "single_task_moe" and a.single_task_name):
            raise ValueError("--init-checkpoint requires single_task_moe and --single-task-name")
        if a.resume and not (a.phase == "train" and a.only in ("sequential_d2nn", "sequential_d2nn_replay", "moe")):
            raise ValueError("--resume requires one sequential training mode")
        names = (a.single_task_name,) if a.single_task_name else TASK_ORDER
        paths={"eurosat":a.eurosat,"clevr":a.clevr,"speech":a.speech,"physical":a.physical}
        missing=[name for name in names if paths[name] is None]
        if missing:
            raise ValueError("missing dataset paths: " + ", ".join(missing))
        seed_all(cfg["seed"]);torch.set_num_threads(4);device=torch.device(cfg.get("device","cuda"));tasks=load_tasks(paths,require_full=bool(cfg.get("require_full", False)),names=names)
        repo_root=Path(__file__).resolve().parents[3]
        git_commit=subprocess.check_output(["git","-C",str(repo_root),"rev-parse","HEAD"],text=True).strip()
        meta={"command":sys.argv,"config":cfg,"git":git_commit,"python":platform.python_version(),"torch":torch.__version__,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,"data":{n:{"root":str(t.root),"manifest_sha256":t.manifest_sha,"sizes":{s:len(t.splits[s][1]) for s in t.splits}} for n,t in tasks.items()},"contract":"single-task D2NN learnability ceiling; sequential D2NN without/with replay; sequential MoE replay 4->8->12->16; stage-by-task matrices"}
        save(a.out/"metadata.json",meta);save(a.out/"actual_config.json",cfg);(a.out/"command.txt").write_text(" ".join(sys.argv)+"\n")
        if a.phase=="smoke":smoke(tasks,cfg,a.out,device);result={"smoke":"pass"}
        elif a.phase=="overfit":result={"overfit":overfit_diagnostics(tasks,cfg,a.out,device)}
        else:
            result={}
            if a.only in ("all","single_task"):result["single_task"]=train_single_task_d2nn(tasks,cfg,a.out,device,a.single_task_name)
            if a.only in ("all","single_task_moe"):result["single_task_moe"]=train_single_task_moe(tasks,cfg,a.out,device,a.single_task_name,a.init_checkpoint)
            if a.only in ("all","sequential_d2nn"):result["sequential_d2nn"]=train_sequential_d2nn(tasks,cfg,a.out,device,False,a.resume)
            if a.only in ("all","sequential_d2nn_replay"):result["sequential_d2nn_replay"]=train_sequential_d2nn(tasks,cfg,a.out,device,True,a.resume)
            if a.only in ("all","moe"):result["moe"]=train_lifelong_moe(tasks,cfg,a.out,device,a.resume)
            save(a.out/"comparison.json",result)
        save(a.out/"status.json",{"status":"complete"})
    except Exception:
        save(a.out/"status.json",{"status":"failed","error":traceback.format_exc()});raise


if __name__=="__main__":main()
