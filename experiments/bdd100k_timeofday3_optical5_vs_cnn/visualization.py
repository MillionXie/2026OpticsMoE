from __future__ import annotations

import math
from pathlib import Path
from typing import Any,Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_optical_diagnostics(model:Any,images:torch.Tensor,root:Path,epoch:int)->None:
    was=model.training;model.eval()
    with torch.no_grad():_,diagnostics=model(images[:8],return_diagnostics=True)
    epoch_name=f"epoch_{epoch:04d}";phases=[layer.wrapped_phase().detach().cpu().numpy() for layer in model.layers]
    fig,axes=plt.subplots(1,5,figsize=(15,3),squeeze=False);image=None
    for index,(ax,phase) in enumerate(zip(axes[0],phases),1):image=ax.imshow(phase,cmap="twilight",vmin=0,vmax=2*math.pi);ax.set_title(f"Layer {index}");ax.axis("off")
    if image is not None:fig.colorbar(image,ax=axes.ravel().tolist(),fraction=.02,pad=.02)
    path=root/"phase_masks"/f"{epoch_name}.png";path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)
    entries=[("input_intensity",diagnostics["input_intensity"][0])]+[(f"after_layer_{i}_intensity",value[0]) for i,value in enumerate(diagnostics["after_layers"],1)]+[("detector_readout_input",diagnostics["detector_input"][0])]
    sample=root/"light_fields"/epoch_name/"sample_000";sample.mkdir(parents=True,exist_ok=True)
    for index,(name,value) in enumerate(entries):_save_intensity(value,sample/f"{index:02d}_{name}.png",name.replace("_"," ").title())
    values=diagnostics["detector_input"].detach().cpu();cols=min(4,len(values));rows=math.ceil(len(values)/cols);fig,axes=plt.subplots(rows,cols,figsize=(3*cols,3*rows),squeeze=False)
    for i,(ax,value) in enumerate(zip(axes.ravel(),values)):ax.imshow(_log(value),cmap="inferno");ax.set_title(f"Sample {i}");ax.axis("off")
    for ax in axes.ravel()[len(values):]:ax.axis("off")
    path=root/"detector_outputs"/f"{epoch_name}.png";path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig);model.train(was)


def save_training_curves(history:list[dict[str,Any]],path:Path)->None:
    if not history:return
    epochs=[r["epoch"] for r in history];fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(epochs,[r["train_loss"] for r in history],label="train");axes[0].plot(epochs,[r["validation_loss"] for r in history],label="validation");axes[0].set_title("Loss")
    axes[1].plot(epochs,[r["train_top1_accuracy"] for r in history],label="train");axes[1].plot(epochs,[r["validation_top1_accuracy"] for r in history],label="validation");axes[1].set_title("Top-1")
    axes[2].plot(epochs,[r["validation_macro_f1"] for r in history],label="macro-F1");axes[2].plot(epochs,[r["validation_balanced_accuracy"] for r in history],label="balanced");axes[2].set_title("Validation")
    for ax in axes:ax.legend();ax.grid(alpha=.25);ax.set_xlabel("Epoch")
    path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def save_confusion_matrix(matrix:list[list[int]],names:Sequence[str],path:Path)->None:
    values=np.asarray(matrix);fig,ax=plt.subplots(figsize=(6,5));image=ax.imshow(values,cmap="Blues");fig.colorbar(image,ax=ax);ax.set_xticks(range(len(names)),names,rotation=30,ha="right");ax.set_yticks(range(len(names)),names);ax.set_xlabel("Predicted");ax.set_ylabel("True")
    for i in range(len(names)):
        for j in range(len(names)):ax.text(j,i,str(values[i,j]),ha="center",va="center")
    path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def _log(value:torch.Tensor)->np.ndarray:
    value=value.detach().cpu().float();return torch.log10(value/value.max().clamp_min(1e-8)+1e-8).numpy()
def _save_intensity(value:torch.Tensor,path:Path,title:str)->None:
    fig,ax=plt.subplots(figsize=(4.5,4));image=ax.imshow(_log(value),cmap="inferno");ax.set_title(title);ax.axis("off");fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

