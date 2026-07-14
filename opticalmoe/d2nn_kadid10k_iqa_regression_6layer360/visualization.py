import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=.12); plt.close(fig)


def _array(value, intensity=False):
    value = value.detach().cpu()
    while value.ndim > 2: value = value[0]
    if torch.is_complex(value): value = value.abs().square() if intensity else value.abs()
    return value.float().numpy()


def _bounds(detector):
    result=[]
    for mask in detector.masks.detach().cpu():
        points=mask.nonzero(); y0,x0=points.min(0).values.tolist(); y1,x1=(points.max(0).values+1).tolist(); result.append((y0,y1,x0,x1))
    return result


def save_training_curves(rows, path):
    epochs=[row["epoch"] for row in rows]; fig,axes=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
    axes[0].plot(epochs,[row["train_loss"] for row in rows],label="train");axes[0].plot(epochs,[row["validation_loss"] for row in rows],label="validation");axes[0].set_title("Loss");axes[0].legend()
    axes[1].plot(epochs,[row["train_rmse"] for row in rows],label="train");axes[1].plot(epochs,[row["validation_rmse"] for row in rows],label="validation");axes[1].set_title("Normalized RMSE");axes[1].legend()
    axes[2].plot(epochs,[row["train_srocc"] for row in rows],label="train");axes[2].plot(epochs,[row["validation_srocc"] for row in rows],label="validation");axes[2].set_title("SROCC");axes[2].set_ylim(-1,1);axes[2].legend()
    for axis in axes: axis.set_xlabel("epoch");axis.grid(alpha=.25)
    _save(fig,path)


def save_scatter(targets, predictions, path, title="Predicted vs target quality"):
    targets=torch.as_tensor(targets).cpu();predictions=torch.as_tensor(predictions).cpu();fig,ax=plt.subplots(figsize=(5.3,5),constrained_layout=True)
    ax.scatter(targets,predictions,s=12,alpha=.55);low=float(min(targets.min(),predictions.min()));high=float(max(targets.max(),predictions.max()));ax.plot([low,high],[low,high],"k--",linewidth=1)
    ax.set_xlabel("target score");ax.set_ylabel("predicted score");ax.set_title(title);ax.grid(alpha=.25);_save(fig,path)


@torch.no_grad()
def save_epoch_artifacts(model, batch, run_dir, tag, enabled=True):
    if not enabled:return
    model.eval();images,targets,_=batch;predictions,items=model(images,return_intermediates=True,capture_layer_fields=True);root=Path(run_dir)/"figures"/"epoch_artifacts"/tag
    phases=model.phase_stack_wrapped().cpu();fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for index,(axis,phase) in enumerate(zip(axes.flat,phases),1):
        image=axis.imshow(phase,cmap="twilight",vmin=0,vmax=2*math.pi);axis.set_title(f"Phase layer {index}");axis.set_xlabel("x pixel");axis.set_ylabel("y pixel");fig.colorbar(image,ax=axis).set_label("phase (rad)")
    _save(fig,root/"phase_masks_overview.png")
    for index,field in enumerate([items["input_canvas"],*items["after_each_layer"]]):
        fig,ax=plt.subplots(figsize=(5.2,4.6),constrained_layout=True);image=ax.imshow(_array(field[0],intensity=index>0),cmap="gray" if index==0 else "inferno");ax.set_xlabel("x pixel");ax.set_ylabel("y pixel");ax.set_title("Input amplitude" if index==0 else f"Intensity after layer {index}");fig.colorbar(image,ax=ax).set_label("amplitude" if index==0 else "intensity");_save(fig,root/("input_amplitude.png" if index==0 else f"after_layer_{index}_intensity.png"))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True);image=axes[0].imshow(_array(items["detector_intensity"][0]),cmap="inferno")
    for anchor,(y0,y1,x0,x1) in zip(model.quality_anchors.tolist(),_bounds(model.detector)):
        axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="cyan",linewidth=1.3));axes[0].text(x0+1,y0+9,f"{anchor:.2f}",color="cyan",fontsize=6)
    axes[0].set_title("Detector intensity and quality anchors");axes[0].set_xlabel("x pixel");axes[0].set_ylabel("y pixel");fig.colorbar(image,ax=axes[0]).set_label("intensity")
    axes[1].bar(range(len(model.quality_anchors)),items["region_probabilities"][0].cpu());axes[1].set_xticks(range(len(model.quality_anchors)),[f"{v:.2f}" for v in model.quality_anchors],rotation=45);axes[1].set_xlabel("quality anchor (0=worst, 1=best)");axes[1].set_ylabel("normalized detector energy");axes[1].set_title(f"target={float(targets[0]):.3f}, prediction={float(predictions[0]):.3f}")
    _save(fig,root/"detector_quality_readout.png")
