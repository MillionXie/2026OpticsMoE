from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def save_stack_diagnostics(surrogate: torch.nn.Module, phase_root: Path, field_root: Path, epoch: int) -> None:
    raw_phases=[conversion.phase_mask.detach().cpu().float() for conversion in surrogate.conversions]
    phase_root.mkdir(parents=True,exist_ok=True)
    _save_phase_overview(
        [torch.remainder(phase,2*math.pi).numpy() for phase in raw_phases],
        phase_root/f"epoch_{epoch:04d}.png","twilight",0,2*math.pi,
    )
    raw_limit=max(float(phase.abs().max()) for phase in raw_phases) or 1.0
    _save_phase_overview(
        [phase.numpy() for phase in raw_phases],
        phase_root/f"epoch_{epoch:04d}_raw.png","coolwarm",-raw_limit,raw_limit,
    )
    _save_phase_overview(
        [torch.cos(phase).numpy() for phase in raw_phases],
        phase_root/f"epoch_{epoch:04d}_cosine.png","coolwarm",-1,1,
    )
    sample_dir=field_root/f"epoch_{epoch:04d}"/"sample_000"; sample_dir.mkdir(parents=True,exist_ok=True)
    for index,field in enumerate(surrogate.last_fields,start=1):
        value=field[0].detach().cpu().float(); display=torch.log10(value/value.max().clamp_min(1e-8)+1e-8).numpy()
        fig,ax=plt.subplots(figsize=(4,4)); image=ax.imshow(display,cmap="inferno"); ax.set_title(f"Detected intensity {index}"); ax.axis("off"); fig.colorbar(image,ax=ax)
        fig.tight_layout(); fig.savefig(sample_dir/f"conversion_{index}.png",dpi=140); plt.close(fig)


def _save_phase_overview(phases:list[np.ndarray],path:Path,cmap:str,vmin:float,vmax:float)->None:
    fig,axes=plt.subplots(1,4,figsize=(12,3),squeeze=False); image=None
    for index,(ax,phase) in enumerate(zip(axes[0],phases),start=1):
        image=ax.imshow(phase,cmap=cmap,vmin=vmin,vmax=vmax); ax.set_title(f"Conversion {index}"); ax.axis("off")
    if image is not None:fig.colorbar(image,ax=axes.ravel().tolist(),fraction=.02,pad=.02)
    fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)


def save_confusion(matrix: list[list[int]], names: Sequence[str], path: Path) -> None:
    values=np.asarray(matrix); fig,ax=plt.subplots(figsize=(9,8)); image=ax.imshow(values,cmap="Blues"); fig.colorbar(image,ax=ax)
    ax.set_xticks(range(len(names)),names,rotation=45,ha="right"); ax.set_yticks(range(len(names)),names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("KADID-10k Quality-3 optical fullstack4 token64 residual")
    fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)


def save_training_curves(rows: list[dict], path: Path) -> None:
    if not rows:return
    epochs=[r["epoch"] for r in rows]; fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(epochs,[r["loss_total"] for r in rows]); axes[0].set_title("Total loss")
    axes[1].plot(epochs,[r["train_top1_accuracy"] for r in rows],label="train"); axes[1].plot(epochs,[r["validation_top1_accuracy"] for r in rows],label="validation"); axes[1].legend(); axes[1].set_title("Top-1")
    axes[2].plot(epochs,[r["validation_macro_f1"] for r in rows],label="macro-F1"); axes[2].plot(epochs,[r["validation_balanced_accuracy"] for r in rows],label="balanced"); axes[2].legend(); axes[2].set_title("Validation")
    for ax in axes: ax.grid(alpha=.25); ax.set_xlabel("Epoch")
    fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160); plt.close(fig)
