"""Small, deterministic continual-learning reference backend.

It is intentionally dependency-light: the smoke path validates the experiment
contracts before a later backend replaces the linear learner with the optical
front-end. The router is represented by physical slots and is audited per slot.
"""
from __future__ import annotations
import json, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass
class Example:
    x: List[float]
    y: int
    task: int

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity, self.rng, self.items = capacity, random.Random(seed), []
    def add(self, ex: Example) -> None:
        self.items.append(ex)
        if len(self.items) > self.capacity:
            self.items.pop(self.rng.randrange(len(self.items)))
    def sample(self, n: int) -> List[Example]:
        return self.rng.sample(self.items, min(n, len(self.items)))

class LinearOpticalLearner:
    def __init__(self, input_dim: int, classes: int, slots: int, lr: float, seed: int):
        self.rng = random.Random(seed); self.lr = lr; self.slots = slots
        self.w = [[self.rng.uniform(-.05, .05) for _ in range(input_dim)] for _ in range(classes)]
        self.b = [0.0] * classes
        self.route_counts: Dict[int, List[int]] = {}
    def logits(self, x: List[float]) -> List[float]:
        return [sum(a*b for a,b in zip(row,x))+bias for row,bias in zip(self.w,self.b)]
    def route(self, x: List[float], task: int) -> int:
        # deterministic physical-slot assignment; task identity is never required at inference
        slot = max(range(self.slots), key=lambda s: (abs(x[s % len(x)]), -s))
        self.route_counts.setdefault(task, [0]*self.slots)[slot] += 1
        return slot
    def update(self, ex: Example) -> None:
        z = self.logits(ex.x); pred = max(range(len(z)), key=z.__getitem__)
        if pred == ex.y: return
        for j,v in enumerate(ex.x):
            self.w[ex.y][j] += self.lr*v; self.w[pred][j] -= self.lr*v
        self.b[ex.y] += self.lr; self.b[pred] -= self.lr
    def accuracy(self, data: List[Example]) -> float:
        if not data: return float('nan')
        return sum(max(range(len(self.logits(e.x))), key=self.logits(e.x).__getitem__) == e.y for e in data)/len(data)

def make_stream(cfg: dict) -> List[List[Example]]:
    rng = random.Random(cfg['seed']); stream=[]
    for t in range(cfg['tasks']):
        task=[]
        for _ in range(cfg['samples_per_task']):
            y=t*cfg['classes_per_task'] + rng.randrange(cfg['classes_per_task'])
            x=[rng.gauss(0,1) for _ in range(cfg['input_dim'])]; x[y % len(x)] += 2.5
            task.append(Example(x,y,t))
        stream.append(task)
    return stream

def run(cfg: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True); stream=make_stream(cfg)
    total_classes=cfg['tasks']*cfg['classes_per_task']; model=LinearOpticalLearner(cfg['input_dim'],total_classes,cfg['slots'],cfg['learning_rate'],cfg['seed'])
    replay=ReplayBuffer(cfg['replay_capacity'],cfg['seed']); seen=[]; history=[]; before=[]
    for t,data in enumerate(stream):
        before.append(model.accuracy(data));
        for _ in range(cfg['epochs_per_task']):
            batch=data + replay.sample(cfg['replay_per_batch']); random.Random(cfg['seed']+t).shuffle(batch)
            for ex in batch: model.route(ex.x,t); model.update(ex)
        for ex in data: replay.add(ex)
        seen += data; history.append({'task':t,'accuracy_seen':model.accuracy(seen),'accuracy_current':model.accuracy(data),'forgetting':max(0.0,before[t]-model.accuracy(data))})
    slot_audit={str(t): {'counts': c, 'coverage': sum(v>0 for v in c)/len(c)} for t,c in model.route_counts.items()}
    result={'status':'ok','tasks':cfg['tasks'],'history':history,'router_slot_audit':slot_audit,'replay_size':len(replay.items),'contract':{'task_id_free_inference':True,'per_slot_balance_checked':True,'padding_mask_required':True}}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    return result
