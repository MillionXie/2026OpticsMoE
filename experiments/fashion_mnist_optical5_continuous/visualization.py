from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


def save_diagnostics(model:Any,images:torch.Tensor,root:Path,epoch:int)->None:
    was=model.training;model.eval()
    with torch.no_grad():_,diagnostics=model(images[:4],return_diagnostics=True)
    phases=[layer.phase_mask.detach().cpu().float() for layer in model.layers];epoch_name=f"epoch_{epoch:04d}"
    for kind,transform,cmap,vmin,vmax in (("wrapped",lambda x:torch.remainder(x,2*math.pi),"twilight",0,2*math.pi),("raw",lambda x:x,"coolwarm",None,None),("cosine",torch.cos,"coolwarm",-1,1)):
        fig,axes=plt.subplots(1,5,figsize=(15,3),squeeze=False);image=None
        for index,(ax,phase) in enumerate(zip(axes[0],phases),1):image=ax.imshow(transform(phase),cmap=cmap,vmin=vmin,vmax=vmax);ax.set_title(f"Layer {index}");ax.axis("off")
        if image is not None:fig.colorbar(image,ax=axes.ravel().tolist(),fraction=.02,pad=.02)
        path=root/"phase_masks"/f"{epoch_name}_{kind}.png";path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160,bbox_inches="tight");plt.close(fig)
    sample=root/"light_fields"/epoch_name/"sample_000";sample.mkdir(parents=True,exist_ok=True);entries=[("input",diagnostics["input_intensity"][0])]+[(f"after_{i}",value[0]) for i,value in enumerate(diagnostics["after_layers"],1)]+[("final_detector",diagnostics["detector_input"][0])]
    for name,value in entries:_save_field(value,sample/f"{name}.png",name)
    values=diagnostics["detector_input"].detach().cpu();distribution=diagnostics["region_distribution"].detach().cpu();fig,axes=plt.subplots(len(values),2,figsize=(11,3*len(values)),squeeze=False)
    for index,value in enumerate(values):
        ax=axes[index,0];ax.imshow(_log(value),cmap="inferno")
        for box in model.class_detector.boxes:ax.add_patch(Rectangle((box["x0"],box["y0"]),box["width"],box["height"],fill=False,edgecolor="cyan",linewidth=1));ax.text(box["x0"],box["y0"],str(box["class_index"]),color="white",fontsize=6)
        ax.axis("off");axes[index,1].bar(range(10),distribution[index].numpy());axes[index,1].set_xticks(range(10));axes[index,1].set_ylim(0,1);axes[index,1].set_title("Detector-region distribution")
    path=root/"detector_outputs"/f"{epoch_name}.png";path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig);model.train(was)


def save_curves(history:list[dict[str,Any]],path:Path)->None:
    if not history:return
    epochs=[row["epoch"] for row in history];fig,axes=plt.subplots(1,3,figsize=(14,4));axes[0].plot(epochs,[row["loss_total"] for row in history]);axes[0].set_title("Total loss");axes[1].plot(epochs,[row["train_top1"] for row in history],label="train");axes[1].plot(epochs,[row["validation_top1"] for row in history],label="validation");axes[1].legend();axes[1].set_title("Top-1");axes[2].plot(epochs,[row["validation_macro_f1"] for row in history]);axes[2].set_title("Validation macro-F1")
    for ax in axes:ax.grid(alpha=.25);ax.set_xlabel("Epoch")
    path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def save_confusion(matrix:list[list[int]],names:list[str],path:Path)->None:
    values=np.asarray(matrix);fig,ax=plt.subplots(figsize=(9,8));image=ax.imshow(values,cmap="Blues");fig.colorbar(image,ax=ax);ax.set_xticks(range(10),names,rotation=45,ha="right");ax.set_yticks(range(10),names);ax.set_xlabel("Predicted");ax.set_ylabel("True");path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def _log(value:torch.Tensor)->np.ndarray:value=value.detach().cpu().float();return torch.log10(value/value.max().clamp_min(1e-8)+1e-8).numpy()
def _save_field(value:torch.Tensor,path:Path,title:str)->None:
    fig,ax=plt.subplots(figsize=(4,4));image=ax.imshow(_log(value),cmap="inferno");ax.set_title(title);ax.axis("off");fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

