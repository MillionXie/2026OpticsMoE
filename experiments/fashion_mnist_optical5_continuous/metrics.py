from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def classification_metrics(targets:list[int],predictions:list[int],names:list[str])->dict[str,Any]:
    n=len(names);matrix=torch.zeros(n,n,dtype=torch.long)
    for target,prediction in zip(targets,predictions):matrix[target,prediction]+=1
    per_class={};recalls=[];f1s=[]
    for index,name in enumerate(names):
        tp=int(matrix[index,index]);support=int(matrix[index].sum());predicted=int(matrix[:,index].sum());precision=tp/predicted if predicted else 0;recall=tp/support if support else 0;f1=2*precision*recall/(precision+recall) if precision+recall else 0;recalls.append(recall);f1s.append(f1);per_class[name]={"support":support,"accuracy":recall,"precision":precision,"recall":recall,"f1":f1}
    total=int(matrix.sum());return {"top1_accuracy":int(matrix.diag().sum())/total if total else 0,"macro_f1":sum(f1s)/n,"balanced_accuracy":sum(recalls)/n,"per_class":per_class,"confusion_matrix":matrix.tolist(),"samples":total}


def write_json(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:str(x))+"\n",encoding="utf-8")
def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as handle:writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

