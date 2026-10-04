"""Formal CRC9 MoE continual learning and immutable-D2NN controls."""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from .crc9_data import DOMAIN_NAMES, balanced_subset, load_domain, sha256
from .crc9_model import CRC9Optics


def save(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n"); temporary.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def metrics(labels, probabilities):
    prediction = probabilities.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(recall_score(labels, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "nll": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=list(range(9))).tolist(),
        "support": np.bincount(labels, minlength=9).tolist(),
    }


def load_tasks(paths, cfg):
    tasks = []
    for index, (name, path) in enumerate(zip(DOMAIN_NAMES, paths)):
        manifest_path = Path(path).with_name(Path(path).stem + "_manifest.json")
        data, manifest = load_domain(path, manifest_path)
        train_ids = balanced_subset(data["train_labels"], cfg["train_per_class"], cfg["seed"] + index)
        val_ids = balanced_subset(data["val_labels"], cfg["val_per_class"], cfg["seed"] + 20 + index)
        tasks.append({
            "name": name, "manifest": manifest, "path": str(path),
            "train_x": torch.from_numpy(data["train_images"][train_ids]),
            "train_y": torch.from_numpy(data["train_labels"][train_ids]).long(),
            "train_ids": data["train_ids"][train_ids],
            "val_x": torch.from_numpy(data["val_images"][val_ids]),
            "val_y": torch.from_numpy(data["val_labels"][val_ids]).long(),
            "val_ids": data["val_ids"][val_ids],
            "test_x": torch.from_numpy(data["test_images"]),
            "test_y": torch.from_numpy(data["test_labels"]).long(),
            "test_ids": data["test_ids"],
        })
    return tasks


def augment(images, rng):
    result = images
    if rng.random() < .5:
        result = result.flip(2)
    if rng.random() < .5:
        result = result.flip(1)
    return result


@torch.no_grad()
def evaluate(model, task, split, device, batch):
    model.eval(); images, labels = task[f"{split}_x"], task[f"{split}_y"]
    probabilities, routes = [], []
    for start in range(0, len(labels), batch):
        output = model(images[start:start+batch].to(device))
        probabilities.append(output["probabilities"].cpu().numpy())
        if output["route_power"] is not None:
            routes.append(output["route_power"].cpu().numpy())
    probability = np.concatenate(probabilities)
    route = np.concatenate(routes) if routes else None
    result = metrics(labels.numpy(), probability)
    if route is not None:
        result["route_mean"] = route.mean(0).tolist()
        result["route_top_frequency"] = (np.bincount(route.argmax(1), minlength=16) / len(route)).tolist()
        result["effective_experts"] = float(np.mean(1. / np.square(route).sum(1)))
    return result, probability, route


def batches(length, size, rng):
    order = rng.permutation(length)
    return [order[start:start+size] for start in range(0, length, size)]


def configure_optimizer(model, cfg):
    head = [parameter for parameter in model.readout.parameters() if parameter.requires_grad]
    optical = [parameter for name, parameter in model.named_parameters()
               if parameter.requires_grad and not name.startswith("readout.")]
    groups = []
    if optical:
        groups.append({"params": optical, "lr": cfg["optical_lr"]})
    if head:
        groups.append({"params": head, "lr": cfg["head_lr"], "weight_decay": cfg["head_weight_decay"]})
    return torch.optim.AdamW(groups)


def cosine_scheduler(optimizer, epochs):
    """Decay every parameter group's own initial LR from 1.0 to 0.1."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: .1 + .9 * (1. + np.cos(np.pi * epoch / epochs)) / 2.)


def train_batches(model, task_batches, optimizer, device, rng):
    model.train(); losses = []; correct = count = 0
    for task, indices, warmup in task_batches:
        x = augment(task["train_x"][indices], rng).to(device)
        y = task["train_y"][indices].to(device)
        optimizer.zero_grad(set_to_none=True); output = model(x, warmup=warmup)
        value = F.cross_entropy(output["logits"], y, label_smoothing=.02)
        value.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step(); losses.append(float(value.detach()))
        correct += int((output["logits"].argmax(1) == y).sum()); count += len(y)
    return {"nll": float(np.mean(losses)), "accuracy_online": correct / count, "samples": count}


def train_d2nn_source(tasks, cfg, source_index, out, device):
    source = tasks[source_index]; seed_all(cfg["seed"])
    model = CRC9Optics("d2nn", cfg["seed"], cfg["phase_dropout"]).to(device)
    model.configure_d2nn_source(); optimizer = configure_optimizer(model, cfg)
    scheduler = cosine_scheduler(optimizer, cfg["source_epochs"])
    history = []; best = (-1., -1., float("inf")); rng = np.random.default_rng(cfg["seed"] + source_index)
    for epoch in range(1, cfg["source_epochs"] + 1):
        train = train_batches(model, [(source, ids, False) for ids in batches(len(source["train_y"]), cfg["batch"], rng)],
                              optimizer, device, rng)
        validation = evaluate(model, source, "val", device, cfg["eval_batch"])[0]
        key = (validation["balanced_accuracy"], validation["macro_f1"], -validation["nll"])
        row = {"epoch": epoch, "train": train, "validation": validation}; history.append(row)
        state = {"model": model.state_dict(), "epoch": epoch, "validation": validation, "config": cfg,
                 "source_index": source_index}
        torch.save(state, out / "last_checkpoint.pt")
        if key > best:
            best = key; torch.save(state, out / "best_checkpoint.pt")
        save(out / "history.json", history); save(out / "status.json", {"state":"training","epoch":epoch})
        print(json.dumps({"mode":"d2nn_source","source":source["name"],"epoch":epoch,"train":train,
                          "val_bal_acc":validation["balanced_accuracy"],"val_macro_f1":validation["macro_f1"]}), flush=True)
        scheduler.step()
    selected = torch.load(out / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"]); results = {"source": source["name"], "selected_epoch": selected["epoch"],
        "selection_rule":"own-domain validation balanced accuracy, then macro-F1, then lower NLL", "domains": {}}
    for task in tasks:
        for split in ("val", "test"):
            value, probability, _ = evaluate(model, task, split, device, cfg["eval_batch"])
            results["domains"].setdefault(task["name"], {})[split] = value
            np.savez_compressed(out / f"direct_{task['name']}_{split}.npz", ids=task[f"{split}_ids"],
                                labels=task[f"{split}_y"].numpy(), probabilities=probability)
    save(out / "results.json", results); save(out / "status.json", {"state":"complete","selected_epoch":selected["epoch"]})


@torch.no_grad()
def extract_features(model, task, split, device, batch):
    model.eval(); x, y = task[f"{split}_x"], task[f"{split}_y"]
    features = []
    for start in range(0, len(y), batch):
        features.append(model.optical_features(x[start:start+batch].to(device))[0].cpu())
    return torch.cat(features), y.clone()


def adapt_d2nn_heads(tasks, cfg, source_checkpoint, out, device):
    payload = torch.load(source_checkpoint, map_location=device, weights_only=False)
    source_index = int(payload["source_index"]); source_name = tasks[source_index]["name"]
    base = CRC9Optics("d2nn", cfg["seed"], 0.).to(device); base.load_state_dict(payload["model"])
    base.configure_d2nn_head_adaptation()
    optical_before = {name:tensor_sha(p) for name,p in base.named_parameters() if not name.startswith("readout.")}
    result = {"source": source_name, "source_checkpoint": str(source_checkpoint), "targets": {},
              "selection_rule":"target validation balanced accuracy, then macro-F1, then lower NLL"}
    for target_index, task in enumerate(tasks):
        model = copy.deepcopy(base).to(device); model.configure_d2nn_head_adaptation()
        train_x, train_y = extract_features(model, task, "train", device, cfg["eval_batch"])
        val_x, val_y = extract_features(model, task, "val", device, cfg["eval_batch"])
        test_x, test_y = extract_features(model, task, "test", device, cfg["eval_batch"])
        optimizer = torch.optim.AdamW(model.readout.parameters(), lr=cfg["adapt_head_lr"],
                                      weight_decay=cfg["head_weight_decay"])
        rng = np.random.default_rng(cfg["seed"] + 100 + source_index * 10 + target_index)
        best = (-1., -1., float("inf")); best_state = None; history = []
        for epoch in range(1, cfg["adapt_epochs"] + 1):
            model.readout.train(); losses = []
            for ids in batches(len(train_y), cfg["head_batch"], rng):
                logits = model.readout(train_x[ids].to(device)); y = train_y[ids].to(device)
                optimizer.zero_grad(set_to_none=True); value = F.cross_entropy(logits, y, label_smoothing=.02)
                value.backward(); optimizer.step(); losses.append(float(value.detach()))
            model.readout.eval()
            with torch.no_grad(): val_prob = model.readout(val_x.to(device)).softmax(1).cpu().numpy()
            val = metrics(val_y.numpy(), val_prob); key = (val["balanced_accuracy"], val["macro_f1"], -val["nll"])
            history.append({"epoch":epoch,"train_nll":float(np.mean(losses)),"validation":val})
            if key > best:
                best = key; best_state = {name:value.detach().cpu().clone() for name,value in model.readout.state_dict().items()}; best_epoch = epoch
        model.readout.load_state_dict(best_state); model.readout.eval()
        with torch.no_grad(): test_prob = model.readout(test_x.to(device)).softmax(1).cpu().numpy()
        target = {"selected_epoch":best_epoch,"validation":history[best_epoch-1]["validation"],
                  "test":metrics(test_y.numpy(),test_prob),"history":history}
        result["targets"][task["name"]] = target
        np.savez_compressed(out / f"adapt_{source_name}_to_{task['name']}_test.npz", ids=task["test_ids"],
                            labels=test_y.numpy(), probabilities=test_prob)
    optical_after = {name:tensor_sha(p) for name,p in base.named_parameters() if not name.startswith("readout.")}
    if optical_before != optical_after:
        raise RuntimeError("D2NN optical tensors changed during head adaptation")
    result["optical_tensors_unchanged"] = True; save(out / "results.json", result); save(out / "status.json", {"state":"complete"})


def balanced_epoch(tasks, batch, rng):
    if batch % len(tasks):
        raise ValueError("batch must be divisible by seen task count")
    each = batch // len(tasks); queues = [batches(len(task["train_y"]), each, rng) for task in tasks]
    steps = max(map(len, queues)); combined = []
    for step in range(steps):
        pieces_x, pieces_y, pieces_task = [], [], []
        for task_index, (task, queue) in enumerate(zip(tasks, queues)):
            ids = queue[step % len(queue)]; pieces_x.append(task["train_x"][ids]); pieces_y.append(task["train_y"][ids])
            pieces_task.append(torch.full((len(ids),), task_index, dtype=torch.long))
        # Old tasks precede the newly introduced task.  The boundary is used
        # only by the optional replay distillation term.
        combined.append((torch.cat(pieces_x), torch.cat(pieces_y), torch.cat(pieces_task),
                         sum(len(value) for value in pieces_y[:-1])))
    return combined


def train_moe_full_replay(tasks, cfg, out, device):
    seed_all(cfg["seed"]); model = CRC9Optics("moe", cfg["seed"], cfg["phase_dropout"]).to(device)
    rng = np.random.default_rng(cfg["seed"]); sequence = []
    for stage_index, task in enumerate(tasks):
        stage = out / f"stage_{stage_index+1}_{task['name']}"; stage.mkdir()
        old_experts = {str(i):tensor_sha(model.first_phase[i]) for i in range(4*stage_index)}
        shared_before = tensor_sha(model.global_phase)
        teacher = None
        if stage_index:
            if cfg.get("distillation_weight", 0.) > 0:
                teacher = copy.deepcopy(model).eval().requires_grad_(False)
            model.configure_moe_stage(stage_index, warmup=True); optimizer = configure_optimizer(model, cfg)
            for epoch in range(1, cfg["warmup_epochs"] + 1):
                train = train_batches(model, [(task, ids, True) for ids in batches(len(task["train_y"]), cfg["batch"], rng)],
                                      optimizer, device, rng)
                print(json.dumps({"mode":"moe","stage":task["name"],"warmup":epoch,"train":train}),flush=True)
        model.configure_moe_stage(stage_index, warmup=False); optimizer = configure_optimizer(model, cfg)
        scheduler = cosine_scheduler(optimizer, cfg["stage_epochs"])
        history=[]; best=(-1.,-1.,float("inf"))
        for epoch in range(1,cfg["stage_epochs"]+1):
            model.train(); losses=[]; correct=count=0
            distillation=[]
            routing=[]; balancing=[]
            for xb,yb,task_id,old_count in balanced_epoch(tasks[:stage_index+1],cfg["batch_by_stage"][stage_index],rng):
                xb=augment(xb,rng).to(device);yb=yb.to(device);task_id=task_id.to(device);optimizer.zero_grad(set_to_none=True)
                output=model(xb);value=F.cross_entropy(output["logits"],yb,label_smoothing=.02)
                route=output["route_power"][:,:4*(stage_index+1)].reshape(len(yb),stage_index+1,4)
                group_mass=route.sum(-1).gather(1,task_id[:,None]).squeeze(1).clamp_min(1e-8)
                route_loss=-group_mass.log().mean()
                member=route[torch.arange(len(yb),device=device),task_id]/group_mass[:,None]
                balance_loss=(member-.25).square().mean()
                value=value+cfg.get("router_group_weight",0.)*route_loss
                value=value+cfg.get("router_balance_weight",0.)*balance_loss
                routing.append(float(route_loss.detach()));balancing.append(float(balance_loss.detach()))
                if teacher is not None and old_count:
                    temperature=float(cfg.get("distillation_temperature",2.))
                    with torch.no_grad():
                        target=F.softmax(teacher(xb[:old_count])["logits"]/temperature,dim=1)
                    kd=F.kl_div(F.log_softmax(output["logits"][:old_count]/temperature,dim=1),target,
                                reduction="batchmean")*temperature**2
                    value=value+cfg["distillation_weight"]*kd;distillation.append(float(kd.detach()))
                value.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
                losses.append(float(value.detach()));correct+=int((output["logits"].argmax(1)==yb).sum());count+=len(yb)
            validation={old["name"]:evaluate(model,old,"val",device,cfg["eval_batch"])[0] for old in tasks[:stage_index+1]}
            score=float(np.mean([value["balanced_accuracy"] for value in validation.values()]))
            macro=float(np.mean([value["macro_f1"] for value in validation.values()]));nll=float(np.mean([value["nll"] for value in validation.values()]))
            row={"epoch":epoch,"train":{"objective":float(np.mean(losses)),"accuracy_online":correct/count,
                 "old_task_distillation":float(np.mean(distillation)) if distillation else None,
                 "router_group_nll":float(np.mean(routing)),"router_within_group_variance":float(np.mean(balancing))},
                 "validation":validation,"selection_score":score};history.append(row)
            state={"model":model.state_dict(),"epoch":epoch,"stage_index":stage_index,"validation":validation,"selection_score":score,"config":cfg}
            torch.save(state,stage/"last_checkpoint.pt")
            key=(score,macro,-nll)
            if key>best:best=key;torch.save(state,stage/"best_checkpoint.pt")
            save(stage/"history.json",history);save(out/"status.json",{"state":"training","stage":task["name"],"epoch":epoch})
            print(json.dumps({"mode":"moe","stage":task["name"],"epoch":epoch,"train_acc":correct/count,
                              "val_bal_acc":{k:v["balanced_accuracy"] for k,v in validation.items()},"mean":score}),flush=True)
            scheduler.step()
        selected=torch.load(stage/"best_checkpoint.pt",map_location=device,weights_only=False);model.load_state_dict(selected["model"])
        old_after={str(i):tensor_sha(model.first_phase[i]) for i in range(4*stage_index)}
        if old_experts!=old_after:raise RuntimeError("old MoE expert changed")
        if stage_index and shared_before!=tensor_sha(model.global_phase):raise RuntimeError("fabricated shared phase changed")
        sequence.append({"task":task["name"],"selected_epoch":selected["epoch"],"validation":selected["validation"],
                         "old_experts_unchanged":True,"shared_phase_unchanged":stage_index==0 or shared_before==tensor_sha(model.global_phase)})
        save(stage/"stage_result.json",sequence[-1])
    result={"sequence":sequence,"final":{},"selection_rule":"mean validation balanced accuracy across all seen domains",
            "replay":"all selected training examples from every previously seen domain in task-balanced batches"}
    for task in tasks:
        result["final"][task["name"]]={}
        for split in ("val","test"):
            value,probability,route=evaluate(model,task,split,device,cfg["eval_batch"]);result["final"][task["name"]][split]=value
            np.savez_compressed(out/f"{task['name']}_{split}.npz",ids=task[f"{split}_ids"],labels=task[f"{split}_y"].numpy(),
                                probabilities=probability,routes=route)
    save(out/"results.json",result);save(out/"status.json",{"state":"complete"})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--domains",type=Path,nargs=4,required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--mode",choices=("moe_full_replay","d2nn_source","d2nn_head_adapt"),required=True)
    parser.add_argument("--source-index",type=int,choices=range(4));parser.add_argument("--source-checkpoint",type=Path)
    parser.add_argument("--device",default="cuda:0");args=parser.parse_args();cfg=json.loads(args.config.read_text())
    args.out.mkdir(parents=True,exist_ok=False);save(args.out/"status.json",{"state":"preparing","pid":os.getpid()})
    try:
        torch.set_num_threads(4);tasks=load_tasks(args.domains,cfg)
        save(args.out/"metadata.json",{"command":sys.argv,"git":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             "python":platform.python_version(),"torch":torch.__version__,"device":args.device,"gpu":torch.cuda.get_device_name(torch.device(args.device)),
             "data":{task["name"]:{"path":task["path"],"sha256":sha256(task["path"])} for task in tasks},"test_used_for_selection":False})
        save(args.out/"split.json",{task["name"]:{"train_ids":task["train_ids"].tolist(),"val_ids":task["val_ids"].tolist(),
             "test_ids":task["test_ids"].tolist()} for task in tasks});save(args.out/"config.json",cfg)
        device=torch.device(args.device)
        if args.mode=="moe_full_replay":train_moe_full_replay(tasks,cfg,args.out,device)
        elif args.mode=="d2nn_source":
            if args.source_index is None:raise ValueError("--source-index is required")
            train_d2nn_source(tasks,cfg,args.source_index,args.out,device)
        else:
            if args.source_checkpoint is None:raise ValueError("--source-checkpoint is required")
            adapt_d2nn_heads(tasks,cfg,args.source_checkpoint,args.out,device)
    except BaseException as error:
        save(args.out/"status.json",{"state":"failed","error":traceback.format_exc()});raise


if __name__=="__main__":main()
