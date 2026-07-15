import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from utils import write_rows


def _image(value,kind="intensity",normalize=True):
    value=value.detach().cpu()
    while value.ndim>2:value=value[0]
    if torch.is_complex(value):value=value.abs().square() if kind=="intensity" else value.abs()
    value=value.float()
    if normalize and value.max()>0:value=value/value.max()
    return value.numpy()


def _save_map(value,path,title,cmap="inferno",label="normalized intensity",vmin=None,vmax=None,normalize=True,kind="intensity"):
    fig,ax=plt.subplots(figsize=(5.3,4.6),constrained_layout=True)
    im=ax.imshow(_image(value,kind=kind,normalize=normalize),cmap=cmap,vmin=vmin,vmax=vmax,origin="upper")
    ax.set_xlabel("x pixel");ax.set_ylabel("y pixel");ax.set_title(title)
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label(label)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)


def _bounds(detector):
    result=[]
    for mask in detector.masks.cpu():
        points=mask.nonzero();a=points.min(0).values.tolist();b=(points.max(0).values+1).tolist();result.append((*a,*b))
    return result


def save_detector(intensity,energies,detector,path,class_names,title):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    im=axes[0].imshow(_image(intensity),cmap="inferno",origin="upper")
    for index,(y0,x0,y1,x1) in enumerate(_bounds(detector)):
        axes[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor="cyan",linewidth=2))
        axes[0].text(x0+3,y0+13,class_names[index],color="cyan",fontsize=8,weight="bold")
    axes[0].set_xlabel("x pixel");axes[0].set_ylabel("y pixel");axes[0].set_title(title)
    cb=fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04);cb.set_label("normalized detector intensity")
    values=energies.detach().cpu().float();axes[1].bar(range(len(values)),values)
    axes[1].set_xticks(range(len(values)),class_names,rotation=20);axes[1].set_xlabel("class detector");axes[1].set_ylabel("normalized energy");axes[1].set_title("Detector-region energy")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)


def save_prompt_routing(amplitude,weights,selected,path):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    im=axes[0].imshow(_image(amplitude,normalize=False),cmap="viridis",origin="upper")
    axes[0].set_xlabel("x pixel");axes[0].set_ylabel("y pixel");axes[0].set_title("Physical 3x3 prompt-amplitude regions")
    cb=fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04);cb.set_label("routing amplitude")
    values=weights.detach().cpu().float();colors=["tab:orange" if bool(v) else "tab:blue" for v in selected.detach().cpu()]
    axes[1].bar(range(9),values,color=colors);axes[1].set_xticks(range(9),[f"E{i}" for i in range(9)]);axes[1].set_xlabel("expert");axes[1].set_ylabel("sparse routing amplitude");axes[1].set_ylim(0,max(.4,float(values.max())*1.15));axes[1].set_title("Top-k routing weights (orange = active)");axes[1].grid(axis="y",alpha=.25)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)


def save_expert_entrance(field,energy_ratios,selected,layout,path):
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    im=axes[0].imshow(_image(field),cmap="inferno",origin="upper")
    for index,aperture in enumerate(layout.expert_apertures):
        active=bool(selected[index]);color="cyan" if active else "white"
        axes[0].add_patch(Rectangle((aperture.x0,aperture.y0),aperture.x1-aperture.x0,aperture.y1-aperture.y0,fill=False,edgecolor=color,linewidth=2 if active else .7,alpha=1 if active else .6))
        axes[0].text(aperture.x0+4,aperture.y0+14,f"E{index}",color=color,fontsize=8,weight="bold")
    axes[0].set_xlabel("x pixel");axes[0].set_ylabel("y pixel");axes[0].set_title("Expert entrance intensity (cyan = selected)")
    cb=fig.colorbar(im,ax=axes[0],fraction=.046,pad=.04);cb.set_label("normalized intensity")
    values=energy_ratios.detach().cpu().float();colors=["tab:orange" if bool(v) else "tab:blue" for v in selected.detach().cpu()]
    axes[1].bar(range(9),values,color=colors);axes[1].set_xticks(range(9),[f"E{i}" for i in range(9)]);axes[1].set_xlabel("expert");axes[1].set_ylabel("fraction of total optical energy");axes[1].set_ylim(0,max(.4,float(values.max())*1.15));axes[1].set_title("Measured energy at expert entrance");axes[1].grid(axis="y",alpha=.25)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)


@torch.no_grad()
def save_epoch_artifacts(model,batch,run_dir,epoch_name,class_names,enabled=True):
    if not enabled:return
    model.eval();images,labels=batch;logits,items=model(images,return_intermediates=True);preds=logits.argmax(1)
    root=Path(run_dir)/"figures"/epoch_name;sample=root/"sample_000"
    _save_map(items["input_canvas"][0],sample/"00_input_canvas.png","CIFAR amplitude: 100x100 + zero pad to 120, centered on 480","gray","amplitude",normalize=False,vmin=0,vmax=1)
    _save_map(items["at_prompt"][0],sample/"01_input_for_global_fanout.png","Input amplitude to global fan-out convolution","gray","amplitude",normalize=False,vmin=0,vmax=1)
    _save_map(items["prompt_amplitude"][0],sample/"02_prompt_amplitude.png","Prompt amplitude: one uniform routing weight per 150x150 region","viridis","amplitude",normalize=False,vmin=0,vmax=1)
    _save_map(items["prompt_phase"][0],sample/"03_prompt_phase.png","Prompt phase: one continuous global lens + carrier map","twilight","phase (rad)",normalize=False,vmin=0,vmax=2*math.pi)
    save_expert_entrance(items["expert_entrance"][0],items["expert_entrance_energy_ratios"][0],items["routing_selected_mask"][0],model.layout,sample/"04_expert_entrance_and_energy.png")
    if items.get("optoelectronic_interlayers_enabled",False):
        conversion_root=sample/"optoelectronic_interlayers"
        for index,(detected,normalized,amplitude) in enumerate(zip(items["interlayer_detector_intensities"],items["interlayer_layer_normalized"],items["interlayer_reloaded_amplitudes"]),start=1):
            _save_map(detected[0],conversion_root/f"layer_{index:02d}_square_detector_intensity.png",f"Expert plane {index}: intensity after 20 cm",label="square-law intensity",normalize=False)
            _save_map(normalized[0],conversion_root/f"layer_{index:02d}_layernorm.png",f"Expert plane {index}: independent per-expert affine LayerNorm",cmap="coolwarm",label="normalized + affine real value",normalize=False)
            _save_map(amplitude[0],conversion_root/f"layer_{index:02d}_relu_amplitude_reload.png",f"Expert plane {index}: ReLU amplitude reloaded",cmap="viridis",label="reloaded amplitude",normalize=False)
    else:
        for index,field in enumerate(items["after_each_expert_layer"],start=1):
            _save_map(field[0],sample/f"{4+index:02d}_after_expert_layer_{index}.png",f"After expert layer {index}")
    distance="20 cm + square detection/LayerNorm/ReLU reload" if items.get("optoelectronic_interlayers_enabled",False) else "5 cm coherent propagation"
    _save_map(items["at_global_fc"][0],sample/"10_at_global_fc.png",f"At global FC after {distance}",kind="amplitude" if items.get("optoelectronic_interlayers_enabled",False) else "intensity",label="reloaded amplitude" if items.get("optoelectronic_interlayers_enabled",False) else "normalized intensity")
    _save_map(items["global_fc_phase"],sample/"11_global_fc_phase.png","Global FC phase: active 450","twilight","phase (rad)",normalize=False,vmin=0,vmax=2*math.pi)
    save_detector(items["detector_intensity"][0],items["detector_energies"][0],model.detector,sample/"12_detector_and_bars.png",class_names,f"Detector | true={class_names[int(labels[0])]} pred={class_names[int(preds[0])]}")
    phase_root=root/"expert_phase_mosaics"
    for index,mosaic in enumerate(model.expert_phase_mosaics(),start=1):
        _save_map(mosaic,phase_root/f"expert_layer_{index:02d}_mosaic.png",f"Nine-expert phase mosaic, layer {index}","twilight","phase (rad)",normalize=False,vmin=0,vmax=2*math.pi)


def save_training_curves(rows,path):
    if not rows:return
    epochs=[r["epoch"] for r in rows];fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    axes[0].plot(epochs,[r["train_loss"] for r in rows],label="train");axes[0].plot(epochs,[r["test_loss"] for r in rows],label="test");axes[0].set_xlabel("epoch");axes[0].set_ylabel("loss");axes[0].set_title("Objective loss");axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].plot(epochs,[r["train_acc"] for r in rows],label="train");axes[1].plot(epochs,[r["test_acc"] for r in rows],label="test");axes[1].set_xlabel("epoch");axes[1].set_ylabel("accuracy");axes[1].set_ylim(0,1);axes[1].set_title("Accuracy");axes[1].legend();axes[1].grid(alpha=.25)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)


def confusion_matrix(preds,targets,num_classes):
    matrix=torch.zeros(num_classes,num_classes,dtype=torch.long)
    for target,pred in zip(targets.cpu(),preds.cpu()):matrix[int(target),int(pred)]+=1
    return matrix


def save_confusion(matrix,path,class_names):
    fig,ax=plt.subplots(figsize=(5.5,4.5),constrained_layout=True);im=ax.imshow(matrix,cmap="Blues")
    ax.set_xticks(range(len(class_names)),class_names,rotation=20);ax.set_yticks(range(len(class_names)),class_names);ax.set_xlabel("predicted");ax.set_ylabel("true");ax.set_title("Confusion matrix")
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label("sample count")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=150,bbox_inches="tight");plt.close(fig)
    rows=[]
    for i in range(len(class_names)):rows.append({"true_class":class_names[i],**{class_names[j]:int(matrix[i,j]) for j in range(len(class_names))}})
    write_rows(path.with_suffix(".csv"),rows)
