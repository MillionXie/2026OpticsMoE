from __future__ import annotations

import math
from pathlib import Path
from typing import Any,Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


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
    values=diagnostics["detector_input"].detach().cpu();distributions=diagnostics["region_distribution"].detach().cpu();detector_fractions=diagnostics["detector_fraction"].detach().cpu();sample_count=min(4,len(values));colors=["cyan","lime","magenta"]
    fig,axes=plt.subplots(sample_count,2,figsize=(9,3.2*sample_count),squeeze=False)
    for i in range(sample_count):
        ax=axes[i,0];ax.imshow(_log(values[i]),cmap="inferno")
        for color,box in zip(colors,model.class_detector.boxes):
            ax.add_patch(Rectangle((box["x0"],box["y0"]),box["width"],box["height"],fill=False,edgecolor=color,linewidth=1.5));ax.text(box["x0"],max(0,box["y0"]-3),box["class_name"],color=color,fontsize=7)
        ax.set_title(f"Sample {i}: detector energy={detector_fractions[i]:.3f}");ax.axis("off")
        bar=axes[i,1];bar.bar(model.class_detector.class_names,distributions[i].numpy(),color=colors);bar.set_ylim(0,1);bar.set_ylabel("Energy share inside 3 regions");bar.set_title(f"Region prediction: {model.class_detector.class_names[int(distributions[i].argmax())]}")
    path=root/"detector_outputs"/f"{epoch_name}.png";path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
    _save_region_layout(model,root/"detector_regions"/"layout.png",colors);model.train(was)


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


def _save_region_layout(model:Any,path:Path,colors:list[str])->None:
    fig,ax=plt.subplots(figsize=(6,5));ax.imshow(np.zeros((model.field_size,model.field_size)),cmap="gray",vmin=0,vmax=1)
    for color,box in zip(colors,model.class_detector.boxes):
        ax.add_patch(Rectangle((box["x0"],box["y0"]),box["width"],box["height"],facecolor=color,edgecolor="white",alpha=.55,linewidth=1.5));ax.text((box["x0"]+box["x1"])/2,(box["y0"]+box["y1"])/2,box["class_name"],ha="center",va="center",fontsize=9,color="black",weight="bold")
    ax.set_title("Fixed class-region detector layout");ax.set_xlim(0,model.field_size);ax.set_ylim(model.field_size,0);ax.set_xlabel("detector x");ax.set_ylabel("detector y");path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
