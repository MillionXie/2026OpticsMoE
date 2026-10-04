"""Validation-selected shared-MLP polish for a completed full-replay MoE."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .crc9_model import CRC9Optics
from .crc9_train import (batches, extract_features, load_tasks, metrics, save,
                         seed_all, tensor_sha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domains", type=Path, nargs=4, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--architecture", choices=("moe", "d2nn"), default="moe")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text()); seed_all(cfg["seed"]); device = torch.device(args.device)
    tasks = load_tasks(args.domains, cfg); payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CRC9Optics(args.architecture, cfg["seed"], 0.).to(device); model.load_state_dict(payload["model"])
    if args.architecture == "moe": model.active_count.fill_(16)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("readout."))
    optical_before = {name:tensor_sha(parameter) for name,parameter in model.named_parameters()
                      if not name.startswith("readout.")}
    cached = {}
    for task in tasks:
        cached[task["name"]] = {split:extract_features(model, task, split, device, cfg["eval_batch"])
                                for split in ("train", "val", "test")}
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=cfg["adapt_head_lr"],
                                  weight_decay=cfg["head_weight_decay"])
    rng = np.random.default_rng(cfg["seed"] + 900); history=[]; best=(-1.,-1.,float("inf"));best_state=None
    for epoch in range(1,cfg["adapt_epochs"]+1):
        model.readout.train(); queues={task["name"]:batches(len(task["train_y"]),cfg["head_batch"]//4,rng) for task in tasks}
        losses=[]
        for step in range(max(map(len,queues.values()))):
            x=[];y=[]
            for task in tasks:
                name=task["name"];ids=queues[name][step%len(queues[name])]
                x.append(cached[name]["train"][0][ids]);y.append(cached[name]["train"][1][ids])
            x=torch.cat(x).to(device);y=torch.cat(y).to(device);order=torch.randperm(len(y),device=device);x,y=x[order],y[order]
            optimizer.zero_grad(set_to_none=True);value=F.cross_entropy(model.readout(x),y,label_smoothing=.02)
            value.backward();optimizer.step();losses.append(float(value.detach()))
        validation={};model.readout.eval()
        with torch.no_grad():
            for task in tasks:
                x,y=cached[task["name"]]["val"];p=model.readout(x.to(device)).softmax(1).cpu().numpy()
                validation[task["name"]]=metrics(y.numpy(),p)
        score=float(np.mean([v["balanced_accuracy"] for v in validation.values()]))
        macro=float(np.mean([v["macro_f1"] for v in validation.values()]));nll=float(np.mean([v["nll"] for v in validation.values()]))
        history.append({"epoch":epoch,"train_nll":float(np.mean(losses)),"validation":validation,"selection_score":score})
        key=(score,macro,-nll)
        if key>best:
            best=key;best_epoch=epoch;best_state={name:value.detach().cpu().clone() for name,value in model.readout.state_dict().items()}
        save(args.out/"history.json",history);save(args.out/"status.json",{"state":"training","epoch":epoch,"best_epoch":best_epoch})
    model.readout.load_state_dict(best_state);model.readout.eval();result={"architecture":args.architecture,"selected_epoch":best_epoch,"validation":{},"test":{},
        "selection_rule":"mean validation balanced accuracy across all four domains","shared_head":True}
    with torch.no_grad():
        for task in tasks:
            for split in ("val","test"):
                x,y=cached[task["name"]][split];p=model.readout(x.to(device)).softmax(1).cpu().numpy()
                result["validation" if split == "val" else "test"][task["name"]]=metrics(y.numpy(),p)
                np.savez_compressed(args.out/f"{task['name']}_{split}.npz",ids=task[f"{split}_ids"],labels=y.numpy(),probabilities=p)
    optical_after={name:tensor_sha(parameter) for name,parameter in model.named_parameters() if not name.startswith("readout.")}
    if optical_before!=optical_after:raise RuntimeError("MoE optical tensors changed during MLP polish")
    result["optical_tensors_unchanged"]=True;torch.save({"model":model.state_dict(),"source_checkpoint":str(args.checkpoint),
        "epoch":best_epoch,"validation":result["validation"]},args.out/"best_checkpoint.pt")
    save(args.out/"results.json",result);save(args.out/"status.json",{"state":"complete","selected_epoch":best_epoch})


if __name__=="__main__":main()
