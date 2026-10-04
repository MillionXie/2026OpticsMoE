"""Continual full-replay adaptation of one shared D2NN MLP readout.

The optical phase planes and both parameter-free OEO blocks remain immutable.
Only the common readout is updated as B, C and D arrive.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .crc9_model import CRC9Optics
from .crc9_train import (batches, evaluate, extract_features, load_tasks,
                         metrics, save, seed_all, tensor_sha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--domains", type=Path, nargs=4, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text()); seed_all(cfg["seed"]); device = torch.device(args.device)
    tasks = load_tasks(args.domains, cfg)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CRC9Optics("d2nn", cfg["seed"], 0.).to(device); model.load_state_dict(payload["model"])
    model.configure_d2nn_head_adaptation()
    optical_before = {name:tensor_sha(parameter) for name,parameter in model.named_parameters()
                      if not name.startswith("readout.")}
    cached = {task["name"]:{split:extract_features(model, task, split, device, cfg["eval_batch"])
                            for split in ("train", "val", "test")} for task in tasks}
    first_validation = evaluate(model, tasks[0], "val", device, cfg["eval_batch"])[0]
    sequence = [{"task":tasks[0]["name"], "selected_epoch":int(payload["epoch"]),
                 "validation":{tasks[0]["name"]:first_validation}, "source_optics":True}]
    rng = np.random.default_rng(cfg["seed"] + 1200)
    for stage_index in range(1, 4):
        optimizer = torch.optim.AdamW(model.readout.parameters(), lr=cfg["adapt_head_lr"],
                                      weight_decay=cfg["head_weight_decay"])
        history=[];best=(-1.,-1.,float("inf"));best_state=None
        seen=tasks[:stage_index+1]; each=max(1,cfg["head_batch"]//len(seen))
        for epoch in range(1,cfg["adapt_epochs"]+1):
            model.readout.train(); queues={t["name"]:batches(len(t["train_y"]),each,rng) for t in seen};losses=[]
            for step in range(max(map(len,queues.values()))):
                xs=[];ys=[]
                for task in seen:
                    ids=queues[task["name"]][step%len(queues[task["name"]])]
                    xs.append(cached[task["name"]]["train"][0][ids]);ys.append(cached[task["name"]]["train"][1][ids])
                x=torch.cat(xs).to(device);y=torch.cat(ys).to(device);order=torch.randperm(len(y),device=device);x,y=x[order],y[order]
                optimizer.zero_grad(set_to_none=True);value=F.cross_entropy(model.readout(x),y,label_smoothing=.02)
                value.backward();optimizer.step();losses.append(float(value.detach()))
            validation={};model.readout.eval()
            with torch.no_grad():
                for task in seen:
                    x,y=cached[task["name"]]["val"];p=model.readout(x.to(device)).softmax(1).cpu().numpy()
                    validation[task["name"]]=metrics(y.numpy(),p)
            score=float(np.mean([v["balanced_accuracy"] for v in validation.values()]))
            macro=float(np.mean([v["macro_f1"] for v in validation.values()]));nll=float(np.mean([v["nll"] for v in validation.values()]))
            history.append({"epoch":epoch,"train_nll":float(np.mean(losses)),"validation":validation,"selection_score":score})
            key=(score,macro,-nll)
            if key>best:
                best=key;best_epoch=epoch;best_state=copy.deepcopy(model.readout.state_dict())
            save(args.out/f"stage_{stage_index+1}_{tasks[stage_index]['name']}_history.json",history)
            save(args.out/"status.json",{"state":"training","stage":tasks[stage_index]["name"],"epoch":epoch})
        model.readout.load_state_dict(best_state)
        sequence.append({"task":tasks[stage_index]["name"],"selected_epoch":best_epoch,
                         "validation":history[best_epoch-1]["validation"],"source_optics":False})
    model.readout.eval();result={"sequence":sequence,"final":{},"optics":"fixed source-A D2NN",
        "adaptation":"one shared MLP, sequential A->B->C->D with full task-balanced replay",
        "selection_rule":"mean validation balanced accuracy across all seen domains"}
    with torch.no_grad():
        for task in tasks:
            result["final"][task["name"]]={}
            for split in ("val","test"):
                x,y=cached[task["name"]][split];p=model.readout(x.to(device)).softmax(1).cpu().numpy()
                result["final"][task["name"]][split]=metrics(y.numpy(),p)
                np.savez_compressed(args.out/f"{task['name']}_{split}.npz",ids=task[f"{split}_ids"],labels=y.numpy(),probabilities=p)
    optical_after={name:tensor_sha(parameter) for name,parameter in model.named_parameters() if not name.startswith("readout.")}
    if optical_before!=optical_after:raise RuntimeError("D2NN optical tensors changed during continual head adaptation")
    result["optical_tensors_unchanged"]=True
    torch.save({"model":model.state_dict(),"source_checkpoint":str(args.checkpoint),"sequence":sequence},args.out/"best_checkpoint.pt")
    save(args.out/"results.json",result);save(args.out/"status.json",{"state":"complete"})


if __name__ == "__main__": main()
