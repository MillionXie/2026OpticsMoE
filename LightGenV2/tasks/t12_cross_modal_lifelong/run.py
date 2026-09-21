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

from .data import load_common, verify_manifest
from .model import CrossModalOptics


TASK_ORDER = ("kather2016", "clevr", "sonyc", "video")

FULL_DATA_REQUIREMENTS = {
    "kather2016": {"source_images": 5000},
    "clevr": {"source_train_images": 70000, "source_val_images": 15000},
    "sonyc": {"source_recordings": 18510},
    "video": {"source_quadruplets": 5000},
}


def validate_full_protocol(name, protocol):
    """Reject any sampled or incomplete package at the formal training boundary."""
    if not protocol.get("all_original_samples", False):
        raise ValueError(f"{name}: formal training requires all_original_samples=true")
    for key, expected in FULL_DATA_REQUIREMENTS[name].items():
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
        expected_classes = {"kather2016": 8, "clevr": 2, "sonyc": 2, "video": 2}
        assert classes == expected_classes[name]
        if require_full:
            validate_full_protocol(name, protocol)
        splits = {s: load_common(root, s) for s in ("train", "val", "test")}
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
    if name == "sonyc":
        events = sorted({r["event"] for r in rows})
        aps, aucs, per_event = [], [], {}
        for event in events:
            mask = np.array([r["event"] == event for r in rows])
            y, score = labels[mask], probabilities[mask, 1]
            ap = float(average_precision_score(y, score)) if (y == 1).any() else None
            auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None
            if ap is not None:
                aps.append(ap)
            if auc is not None:
                aucs.append(auc)
            per_event[event] = {"n": int(mask.sum()), "positive": int(y.sum()), "ap": ap, "auroc": auc}
        result.update(macro_ap=float(np.mean(aps)) if aps else None,
                      macro_auroc=float(np.mean(aucs)) if aucs else None,
                      macro_ap_supported_classes=len(aps), per_event=per_event)
    return result


def selection_score(name, metrics):
    if name == "kather2016":
        return metrics["macro_f1"]
    if name == "sonyc":
        return metrics["macro_ap"]
    return metrics["balanced_accuracy"]


@torch.no_grad()
def evaluate(model, task, split, device, batch, active_task=None):
    fields, labels, rows = task.splits[split]
    model.eval()
    probs, routes, capture = [], [], []
    if active_task is not None and model.architecture == "moe":
        model.active_count.fill_(4 * (active_task + 1))
    for start in range(0, len(labels), batch):
        out = model(fields[start:start+batch].to(device=device, dtype=torch.float32), task.name)
        probs.append(out["probabilities"].cpu().numpy())
        capture.append(out["capture"].cpu().numpy())
        if out["route_power"] is not None:
            routes.append(out["route_power"].cpu().numpy())
    p = np.concatenate(probs)
    q = np.concatenate(routes) if routes else None
    result = classification_metrics(task.name, labels, p, rows)
    result["zero_readout_fraction"] = float((np.concatenate(capture) <= 1e-12).mean())
    result["route_mean"] = q.mean(0).tolist() if q is not None else None
    result["route_top_frequency"] = (np.bincount(q.argmax(1), minlength=16) / len(q)).tolist() if q is not None else None
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
        out = model(fields[ix].to(device=device, dtype=torch.float32), task.name)
        probs.append(out["probabilities"].cpu().numpy())
    p = np.concatenate(probs)
    return classification_metrics(task.name, labels[indices], p, [rows[i] for i in indices])


def sample_weights(task, indices):
    if task.name != "sonyc":
        return torch.ones(len(indices))
    labels, rows = task.splits["train"][1], task.splits["train"][2]
    weights = np.ones(len(indices), np.float32)
    for event in sorted({rows[i]["event"] for i in indices}):
        event_all = np.array([r["event"] == event for r in rows])
        for label in (0, 1):
            count = int((event_all & (labels == label)).sum())
            mask = np.array([rows[i]["event"] == event and labels[i] == label for i in indices])
            if count:
                weights[mask] = 1.0 / count
    weights /= weights.mean()
    return torch.from_numpy(weights)


def task_loss(model, task, indices, device, warmup=False, balance=0.0):
    fields, labels, _ = task.splits["train"]
    out = model(fields[indices].to(device=device, dtype=torch.float32), task.name, warmup=warmup)
    y = torch.as_tensor(labels[indices], device=device)
    w = sample_weights(task, indices).to(device)
    loss = (F.nll_loss(out["probabilities"].clamp_min(1e-12).log(), y, reduction="none") * w).mean()
    if balance and out["route_power"] is not None and not warmup:
        n = int(model.active_count)
        loss = loss + balance * (out["route_power"][:, :n].mean(0) - 1.0/n).square().sum()
    return loss


def replay_indices(task, budget, seed):
    labels = task.splits["train"][1]
    rng = np.random.default_rng(seed)
    groups = []
    if task.name == "sonyc":
        rows = task.splits["train"][2]
        keys = [(r["event"], int(y)) for r, y in zip(rows, labels)]
    else:
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
    learned={row["task"]:selection_score(row["task"],row["validation"][row["task"]]) for row in all_history}
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
        out=model(fields[start:start+batch].to(device=device,dtype=torch.float32),task.name)
        features.append(out["ccd_features"].cpu())
    return torch.cat(features),np.asarray(labels),rows


def fit_mlp_probe(model, task, cfg, device, seed):
    """Fit only the electronic head on all target training samples."""
    cached={split:extract_ccd_features(model,task,split,device,cfg["eval_batch"])
            for split in ("train","val","test")}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64),nn.GELU(),
                           nn.Linear(64,task.classes)).to(device)
    train_x,train_y,_=cached["train"]
    train_x=train_x.to(device);train_y_t=torch.as_tensor(train_y,device=device)
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
    for split,(x,y,rows) in cached.items():
        with torch.no_grad():p=head(x.to(device)).softmax(1).cpu().numpy()
        metrics[split]=classification_metrics(task.name,y,p,rows)
    return {"selected_epoch":best_epoch,"metrics":metrics}


def save_continual_matrix(root, all_history):
    payload = {"task_order": list(TASK_ORDER), "metric": {
        "kather2016": "macro_f1", "clevr": "balanced_accuracy",
        "sonyc": "macro_ap", "video": "balanced_accuracy"}}
    for split in ("validation", "test"):
        payload[split] = [
            {name: selection_score(name, row[split][name]) for name in TASK_ORDER[:i + 1]}
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
    """Train an independent optical D2NN for each task as a learnability ceiling."""
    root = out / "single_task_d2nn"; root.mkdir()
    summary = {}
    epochs = int(cfg.get("single_task_epochs", cfg["task_epochs"]))
    names = (selected_task,) if selected_task else TASK_ORDER
    for name in names:
        task_index = TASK_ORDER.index(name)
        task_root = root / name; task_root.mkdir()
        seed_all(cfg["seed"] + task_index)
        model = CrossModalOptics("d2nn", cfg["seed"] + task_index, cfg["phase_dropout"]).to(device)
        model.configure_task(task_index, warmup=False)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=cfg["lr"] * .1)
        best, history = -float("inf"), []
        for epoch in range(1, epochs + 1):
            model.train(); started=time.time(); losses=[]
            order=np.random.default_rng(cfg["seed"]+task_index*1000+epoch).permutation(len(tasks[name].splits["train"][1]))
            for ix in chunks(order, cfg["batch"]):
                optimizer.zero_grad(set_to_none=True)
                loss=task_loss(model,tasks[name],ix,device)
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            val=evaluate(model,tasks[name],"val",device,cfg["eval_batch"])[0]
            score=selection_score(name,val)
            row={"epoch":epoch,"loss":float(np.mean(losses)),"validation_score":score,
                 "validation":val,"seconds":time.time()-started}
            history.append(row);save(task_root/"history.json",history)
            cp={"model":model.state_dict(),"epoch":epoch,"task":name,"validation":val,"score":score}
            torch.save(cp,task_root/"last_checkpoint.pt")
            if score>best:best=score;torch.save(cp,task_root/"best_checkpoint.pt")
            print(json.dumps({"arch":"single_task_d2nn","task":name,"epoch":epoch,
                              "val":score,"loss":row["loss"],"seconds":row["seconds"]}),flush=True)
        cp=torch.load(task_root/"best_checkpoint.pt",map_location=device,weights_only=False);model.load_state_dict(cp["model"])
        metrics={split:evaluate(model,tasks[name],split,device,cfg["eval_batch"])[0]
                 for split in ("train","val","test")}
        summary[name]={"selected_epoch":cp["epoch"],"metrics":metrics}
        save(task_root/"results.json",summary[name])
    save(root/"results.json",summary)
    if selected_task:
        return {"single_task": summary}
    probes={}
    score_matrix={}
    for source_index,source in enumerate(TASK_ORDER):
        model=CrossModalOptics("d2nn",cfg["seed"]+source_index,cfg["phase_dropout"]).to(device)
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


def overfit_diagnostics(tasks, cfg, out, device):
    """Verify that each task can be memorized before any full-data run."""
    root=out/"overfit_diagnostics";root.mkdir()
    budget=int(cfg.get("overfit_samples",64));steps=int(cfg.get("overfit_steps",200))
    result={}
    for task_index,name in enumerate(TASK_ORDER):
        seed_all(cfg["seed"]+task_index)
        model=CrossModalOptics("d2nn",cfg["seed"]+task_index,0).to(device)
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


def train_sequential_d2nn(tasks, cfg, out, device, use_replay):
    """Sequential D2NN with optional replay and a stage-by-task score matrix."""
    tag = "replay" if use_replay else "no_replay"
    root = out / f"sequential_d2nn_{tag}"; root.mkdir()
    seed_all(cfg["seed"])
    model = CrossModalOptics("d2nn", cfg["seed"], cfg["phase_dropout"]).to(device)
    replay = {}; all_history = []
    for task_index, name in enumerate(TASK_ORDER):
        task_root = root / f"stage_{task_index+1}_{name}"; task_root.mkdir()
        task = tasks[name]
        old_head_hash = {n:module_sha(model.heads[n]) for n in TASK_ORDER[:task_index]}
        shared_before = {"first_phase":state_sha(model.first_phase),
                         "global_phase":state_sha(model.global_phase)}
        model.configure_task(task_index, warmup=False)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg["lr"])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg["task_epochs"],eta_min=cfg["lr"]*.1)
        best=-float("inf"); history=[]
        for epoch in range(1,cfg["task_epochs"]+1):
            model.train();started=time.time();losses=[]
            order=np.random.default_rng(cfg["seed"]+task_index*1000+epoch).permutation(len(task.splits["train"][1]))
            current_chunks=chunks(order,cfg["batch"])
            replay_chunks={old:chunks(np.random.default_rng(cfg["seed"]+task_index*1000+epoch+j).permutation(ix),cfg["replay_batch"])
                           for j,(old,ix) in enumerate(replay.items())} if use_replay else {}
            for step,current in enumerate(current_chunks):
                optimizer.zero_grad(set_to_none=True)
                current_loss=task_loss(model,task,current,device)
                replay_losses=[]
                for old in TASK_ORDER[:task_index] if use_replay else ():
                    pool=replay_chunks[old]; ix=pool[step%len(pool)]
                    replay_losses.append(task_loss(model,tasks[old],ix,device))
                loss=combine_current_replay(current_loss,replay_losses,cfg["replay_weight"])
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step()
            val={n:evaluate(model,tasks[n],"val",device,cfg["eval_batch"])[0] for n in TASK_ORDER[:task_index+1]}
            score=float(np.mean([selection_score(n,val[n]) for n in val]))
            row={"epoch":epoch,"loss":float(np.mean(losses)),"validation_mean_score":score,"validation":val,"seconds":time.time()-started}
            history.append(row);save(task_root/"history.json",history)
            cp={"model":model.state_dict(),"epoch":epoch,"task_index":task_index,"validation":val,"score":score}
            torch.save(cp,task_root/"last_checkpoint.pt")
            if score>best:best=score;torch.save(cp,task_root/"best_checkpoint.pt")
            print(json.dumps({"arch":f"sequential_d2nn_{tag}","task":name,"epoch":epoch,"val":score,"loss":row["loss"],"seconds":row["seconds"]}),flush=True)
        cp=torch.load(task_root/"best_checkpoint.pt",map_location=device,weights_only=False);model.load_state_dict(cp["model"])
        after_heads={n:module_sha(model.heads[n]) for n in TASK_ORDER[:task_index]}
        assert old_head_hash==after_heads, "frozen old task head changed"
        stage_eval=stage_evaluation(model,tasks,TASK_ORDER[:task_index+1],device,cfg)
        shared_after={"first_phase":state_sha(model.first_phase),"global_phase":state_sha(model.global_phase)}
        save(task_root/"stage_result.json",{"selected_epoch":cp["epoch"],**stage_eval,
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


def train_lifelong_moe(tasks, cfg, out, device):
    root = out / "lifelong_moe"; root.mkdir()
    seed_all(cfg["seed"])
    model = CrossModalOptics("moe", cfg["seed"], cfg["phase_dropout"]).to(device)
    replay = {}; all_history = []
    for task_index, name in enumerate(TASK_ORDER):
        task_root = root / f"stage_{task_index+1}_{name}"; task_root.mkdir()
        task = tasks[name]
        old_hash = [state_sha(p) for p in model.first_phase[:4*task_index]]
        old_head_hash = {n: module_sha(model.heads[n]) for n in TASK_ORDER[:task_index]}
        if task_index:
            model.configure_task(task_index, warmup=True)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
            for epoch in range(1, cfg["warmup_epochs"] + 1):
                model.train(); losses=[]
                order=np.random.default_rng(cfg["seed"]+task_index*100+epoch).permutation(len(task.splits["train"][1]))
                for ix in chunks(order, cfg["batch"]):
                    optimizer.zero_grad(set_to_none=True); loss=task_loss(model,task,ix,device,warmup=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();losses.append(float(loss.detach()))
                print(json.dumps({"arch":"moe","task":name,"stage":"warmup","epoch":epoch,"loss":float(np.mean(losses))}),flush=True)
        model.configure_task(task_index, warmup=False)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg["lr"])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg["task_epochs"],eta_min=cfg["lr"]*.1)
        best=-float("inf"); history=[]
        for epoch in range(1,cfg["task_epochs"]+1):
            model.train();started=time.time();losses=[]
            order=np.random.default_rng(cfg["seed"]+task_index*1000+epoch).permutation(len(task.splits["train"][1]))
            current_chunks=chunks(order,cfg["batch"])
            replay_chunks={old:chunks(np.random.default_rng(cfg["seed"]+task_index*1000+epoch+j).permutation(ix),cfg["replay_batch"])
                           for j,(old,ix) in enumerate(replay.items())}
            for step,current in enumerate(current_chunks):
                optimizer.zero_grad(set_to_none=True)
                current_loss=task_loss(model,task,current,device,balance=cfg["route_balance"])
                replay_losses=[]
                for old in TASK_ORDER[:task_index]:
                    pool=replay_chunks[old]; ix=pool[step%len(pool)]
                    replay_losses.append(task_loss(model,tasks[old],ix,device,balance=cfg["route_balance"]))
                loss=combine_current_replay(current_loss,replay_losses,cfg["replay_weight"])
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();losses.append(float(loss.detach()))
            scheduler.step()
            val={n:evaluate(model,tasks[n],"val",device,cfg["eval_batch"],task_index)[0] for n in TASK_ORDER[:task_index+1]}
            score=float(np.mean([selection_score(n,val[n]) for n in val]))
            row={"epoch":epoch,"loss":float(np.mean(losses)),"validation_mean_score":score,"validation":val,"seconds":time.time()-started}
            history.append(row);save(task_root/"history.json",history)
            cp={"model":model.state_dict(),"epoch":epoch,"task_index":task_index,"validation":val,"score":score}
            torch.save(cp,task_root/"last_checkpoint.pt")
            if score>best:best=score;torch.save(cp,task_root/"best_checkpoint.pt")
            print(json.dumps({"arch":"moe","task":name,"epoch":epoch,"val":score,"loss":row["loss"],"route":val[name]["route_mean"],"seconds":row["seconds"]}),flush=True)
        cp=torch.load(task_root/"best_checkpoint.pt",map_location=device,weights_only=False);model.load_state_dict(cp["model"])
        after=[state_sha(p) for p in model.first_phase[:4*task_index]]
        after_heads={n:module_sha(model.heads[n]) for n in TASK_ORDER[:task_index]}
        assert old_hash==after, "frozen old expert changed"
        assert old_head_hash==after_heads, "frozen old task head changed"
        stage_eval=stage_evaluation(model,tasks,TASK_ORDER[:task_index+1],device,cfg,task_index)
        save(task_root/"stage_result.json",{"selected_epoch":cp["epoch"],**stage_eval,
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
        model=CrossModalOptics(arch,cfg["seed"],0).to(device)
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
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--kather2016",type=Path);p.add_argument("--clevr",type=Path);p.add_argument("--sonyc",type=Path);p.add_argument("--video",type=Path);p.add_argument("--out",type=Path,required=True);p.add_argument("--phase",choices=["smoke","overfit","train"],default="train");p.add_argument("--only",choices=["all","single_task","sequential_d2nn","sequential_d2nn_replay","moe"],default="all");p.add_argument("--single-task-name",choices=TASK_ORDER);a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False);save(a.out/"status.json",{"status":"running","pid":os.getpid()})
    try:
        if a.single_task_name and not (a.phase == "train" and a.only == "single_task"):
            raise ValueError("--single-task-name is only valid with --phase train --only single_task")
        names = (a.single_task_name,) if a.single_task_name else TASK_ORDER
        paths={"kather2016":a.kather2016,"clevr":a.clevr,"sonyc":a.sonyc,"video":a.video}
        missing=[name for name in names if paths[name] is None]
        if missing:
            raise ValueError("missing dataset paths: " + ", ".join(missing))
        seed_all(cfg["seed"]);torch.set_num_threads(4);device=torch.device(cfg.get("device","cuda"));tasks=load_tasks(paths,require_full=a.phase=="train",names=names)
        repo_root=Path(__file__).resolve().parents[3]
        git_commit=subprocess.check_output(["git","-C",str(repo_root),"rev-parse","HEAD"],text=True).strip()
        meta={"command":sys.argv,"config":cfg,"git":git_commit,"python":platform.python_version(),"torch":torch.__version__,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,"data":{n:{"root":str(t.root),"manifest_sha256":t.manifest_sha,"sizes":{s:len(t.splits[s][1]) for s in t.splits}} for n,t in tasks.items()},"contract":"single-task D2NN learnability ceiling; sequential D2NN without/with replay; sequential MoE replay 4->8->12->16; stage-by-task matrices"}
        save(a.out/"metadata.json",meta);save(a.out/"actual_config.json",cfg);(a.out/"command.txt").write_text(" ".join(sys.argv)+"\n")
        if a.phase=="smoke":smoke(tasks,cfg,a.out,device);result={"smoke":"pass"}
        elif a.phase=="overfit":result={"overfit":overfit_diagnostics(tasks,cfg,a.out,device)}
        else:
            result={}
            if a.only in ("all","single_task"):result["single_task"]=train_single_task_d2nn(tasks,cfg,a.out,device,a.single_task_name)
            if a.only in ("all","sequential_d2nn"):result["sequential_d2nn"]=train_sequential_d2nn(tasks,cfg,a.out,device,False)
            if a.only in ("all","sequential_d2nn_replay"):result["sequential_d2nn_replay"]=train_sequential_d2nn(tasks,cfg,a.out,device,True)
            if a.only in ("all","moe"):result["moe"]=train_lifelong_moe(tasks,cfg,a.out,device)
            save(a.out/"comparison.json",result)
        save(a.out/"status.json",{"status":"complete"})
    except Exception:
        save(a.out/"status.json",{"status":"failed","error":traceback.format_exc()});raise


if __name__=="__main__":main()
