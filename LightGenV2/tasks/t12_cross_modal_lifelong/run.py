"""Joint D2NN baseline and sequential cross-modal optical MoE training."""
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, recall_score, roc_auc_score

from .data import load_common, verify_manifest
from .model import CrossModalOptics


TASK_ORDER = ("sen12ms", "clevr", "sonyc")


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


def load_tasks(paths):
    tasks = {}
    for name in TASK_ORDER:
        root = Path(paths[name])
        manifest_sha = verify_manifest(root)
        protocol = json.loads((root / "protocol.json").read_text())
        assert protocol["task"] == name
        classes = int(protocol["classes"])
        assert classes == (10 if name == "sen12ms" else 2)
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
    if name == "sen12ms":
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
    result["route_top_frequency"] = (np.bincount(q.argmax(1), minlength=12) / len(q)).tolist() if q is not None else None
    return result, p, q


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


def train_joint_d2nn(tasks, cfg, out, device):
    root = out / "joint_d2nn"; root.mkdir()
    seed_all(cfg["seed"])
    model = CrossModalOptics("d2nn", cfg["seed"], cfg["phase_dropout"]).to(device)
    for head in model.heads.values():
        head.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["joint_epochs"], eta_min=cfg["lr"] * 0.1)
    best, history = -float("inf"), []
    for epoch in range(1, cfg["joint_epochs"] + 1):
        model.train(); started = time.time(); losses = []
        queues = {}
        for j, name in enumerate(TASK_ORDER):
            queues[name] = list(np.array_split(np.random.default_rng(cfg["seed"] + epoch*10 + j).permutation(len(tasks[name].splits["train"][1])),
                                               max(1, int(np.ceil(len(tasks[name].splits["train"][1]) / cfg["batch"])))))
        while any(queues.values()):
            for name in TASK_ORDER:
                if not queues[name]:
                    continue
                ix = queues[name].pop(0)
                optimizer.zero_grad(set_to_none=True)
                loss = task_loss(model, tasks[name], ix, device)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append(float(loss.detach()))
        scheduler.step()
        val = {n: evaluate(model, tasks[n], "val", device, cfg["eval_batch"])[0] for n in TASK_ORDER}
        score = float(np.mean([selection_score(n, val[n]) for n in TASK_ORDER]))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_mean_score": score,
               "validation": val, "seconds": time.time() - started}
        history.append(row); save(root / "history.json", history)
        cp = {"model": model.state_dict(), "epoch": epoch, "validation": val, "score": score}
        torch.save(cp, root / "last_checkpoint.pt")
        if score > best:
            best = score; torch.save(cp, root / "best_checkpoint.pt")
        print(json.dumps({"arch":"d2nn","epoch":epoch,"val":score,"loss":row["loss"],"seconds":row["seconds"]}), flush=True)
    cp = torch.load(root / "best_checkpoint.pt", map_location=device, weights_only=False); model.load_state_dict(cp["model"])
    return finalize(model, tasks, root, device, cfg, cp["epoch"], "offline_joint")


def chunks(indices, size):
    return [indices[i:i+size] for i in range(0, len(indices), size)]


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
                terms=[task_loss(model,task,current,device,balance=cfg["route_balance"])]
                for old in TASK_ORDER[:task_index]:
                    pool=replay_chunks[old]; ix=pool[step%len(pool)]
                    terms.append(task_loss(model,tasks[old],ix,device,balance=cfg["route_balance"])*cfg["replay_weight"])
                loss=torch.stack(terms).mean();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();losses.append(float(loss.detach()))
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
        stage_eval={n:evaluate(model,tasks[n],"val",device,cfg["eval_batch"],task_index)[0] for n in TASK_ORDER[:task_index+1]}
        save(task_root/"stage_result.json",{"selected_epoch":cp["epoch"],"validation":stage_eval,"old_experts_unchanged":old_hash==after,"old_heads_unchanged":old_head_hash==after_heads})
        all_history.append({"task":name,"selected_epoch":cp["epoch"],"validation":stage_eval})
        replay[name]=replay_indices(task,cfg["replay_per_task"],cfg["seed"]+task_index)
        model.configure_task(task_index, warmup=False)
    save(root/"sequence.json",all_history)
    return finalize(model,tasks,root,device,cfg,all_history[-1]["selected_epoch"],"sequential_lifelong")


def finalize(model,tasks,root,device,cfg,epoch,training):
    results={"training":training,"selected_epoch":epoch,"validation":{},"test":{}}
    task_index=2 if model.architecture=="moe" else None
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
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--sen12ms",type=Path,required=True);p.add_argument("--clevr",type=Path,required=True);p.add_argument("--sonyc",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--phase",choices=["smoke","train"],default="train");p.add_argument("--only",choices=["both","d2nn","moe"],default="both");a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False);save(a.out/"status.json",{"status":"running","pid":os.getpid()})
    try:
        seed_all(cfg["seed"]);torch.set_num_threads(4);device=torch.device(cfg.get("device","cuda"));tasks=load_tasks({"sen12ms":a.sen12ms,"clevr":a.clevr,"sonyc":a.sonyc})
        meta={"command":sys.argv,"config":cfg,"git":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"python":platform.python_version(),"torch":torch.__version__,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,"data":{n:{"root":str(t.root),"manifest_sha256":t.manifest_sha,"sizes":{s:len(t.splits[s][1]) for s in t.splits}} for n,t in tasks.items()},"contract":"four modalities (SAR, optical/RGB, text, audio) across three paired tasks; shared physical phases and CCD; D2NN offline joint, MoE sequential 4->8->12"}
        save(a.out/"metadata.json",meta);save(a.out/"actual_config.json",cfg);(a.out/"command.txt").write_text(" ".join(sys.argv)+"\n")
        if a.phase=="smoke":smoke(tasks,cfg,a.out,device);result={"smoke":"pass"}
        else:
            result={}
            if a.only in ("both","d2nn"):result["d2nn"]=train_joint_d2nn(tasks,cfg,a.out,device)
            if a.only in ("both","moe"):result["moe"]=train_lifelong_moe(tasks,cfg,a.out,device)
            save(a.out/"comparison.json",result)
        save(a.out/"status.json",{"status":"complete"})
    except Exception:
        save(a.out/"status.json",{"status":"failed","error":traceback.format_exc()});raise


if __name__=="__main__":main()
