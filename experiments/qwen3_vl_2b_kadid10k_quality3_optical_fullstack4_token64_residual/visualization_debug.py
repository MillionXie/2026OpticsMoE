from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


INDEX_FIELDS=["epoch","phase","sample_index","true_name","teacher_pred_name","student_pred_name",
              "correct","vision_hidden_mse","vision_hidden_cosine_mean_token","answer_hidden_mse",
              "answer_hidden_cosine","input_image_path","debug_dir"]


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value=tensor.detach().float().cpu(); count=value.numel(); negatives=int((value<0).sum())
    return {"shape":list(value.shape),"min":float(value.min()) if count else 0.0,
            "max":float(value.max()) if count else 0.0,"mean":float(value.mean()) if count else 0.0,
            "std":float(value.std(unbiased=False)) if count else 0.0,"num_negative":negatives,
            "negative_ratio":negatives/count if count else 0.0}


def field_metrics(field: torch.Tensor) -> dict[str, Any]:
    value=field.detach().float().cpu(); stats=tensor_stats(value); count=value.numel()
    stats.update({"sparsity_ratio_abs_lt_1e-6":float((value.abs()<1e-6).float().mean()) if count else 0.0,
                  "energy_sum":float(value.clamp_min(0).sum()),
                  "energy_mean":float(value.clamp_min(0).mean()) if count else 0.0})
    return stats


def hidden_similarity_metrics(student: torch.Tensor, teacher: torch.Tensor) -> dict[str, Any]:
    student=student.detach().float().cpu();teacher=teacher.detach().float().cpu()
    if student.shape!=teacher.shape:raise ValueError(f"Hidden shapes differ: {tuple(student.shape)} vs {tuple(teacher.shape)}")
    diff=student-teacher;mse=float(diff.square().mean());mae=float(diff.abs().mean())
    flat_student=student.flatten();flat_teacher=teacher.flatten()
    result={"mse":mse,"mae":mae,"rmse":math.sqrt(mse),
            "pearson":_pearson(flat_student,flat_teacher),"pearson_flat":_pearson(flat_student,flat_teacher),
            "spearman_flat":_spearman(flat_student,flat_teacher),
            "relative_l2_error":float(diff.norm()/teacher.norm().clamp_min(1e-12)),
            "student_norm":float(student.norm()),"teacher_norm":float(teacher.norm()),"diff_norm":float(diff.norm())}
    if student.ndim>=2:
        rows_student=student.reshape(-1,student.shape[-1]);rows_teacher=teacher.reshape(-1,teacher.shape[-1])
        cosine=F.cosine_similarity(rows_student,rows_teacher,dim=-1)
        row_diff=rows_student-rows_teacher
        result.update({"cosine_mean_token":float(cosine.mean()),"cosine_min_token":float(cosine.min()),
                       "cosine_max_token":float(cosine.max()),"cosine_answer_or_cls":None,
                       "student_norm_mean":float(rows_student.norm(dim=-1).mean()),
                       "teacher_norm_mean":float(rows_teacher.norm(dim=-1).mean()),
                       "diff_norm_mean":float(row_diff.norm(dim=-1).mean())})
    else:
        result["cosine"]=float(F.cosine_similarity(student[None],teacher[None],dim=-1)[0])
    return result


def save_tensor_heatmap(tensor: torch.Tensor, path: Path, title: str, value_type: str,
                        percentile_clip: float=99.0, center_zero: bool=False,
                        max_tokens: int=64) -> None:
    raw=tensor.detach().float().cpu();display=raw
    if display.ndim==1:display=display.unsqueeze(0)
    while display.ndim>2:display=display[0]
    display=display[:max_tokens]
    if display.shape[-1]>256:
        display=F.adaptive_avg_pool2d(display[None,None],(display.shape[-2],256))[0,0]
    array=display.numpy();finite=array[np.isfinite(array)]
    if finite.size==0:finite=np.asarray([0.0],dtype=np.float32)
    if center_zero:
        clip=float(np.percentile(np.abs(finite),percentile_clip));clip=max(clip,1e-12);vmin,vmax=-clip,clip;cmap="coolwarm"
    else:
        low=0.0 if value_type in {"intensity","absolute_difference"} else float(np.percentile(finite,100-percentile_clip))
        high=float(np.percentile(finite,percentile_clip));
        if high<=low:high=low+1e-12
        vmin,vmax=low,high;cmap="inferno" if value_type=="intensity" else "viridis"
    stats=tensor_stats(raw);figure,axis=plt.subplots(figsize=(10,4))
    image=axis.imshow(array,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax,interpolation="nearest")
    axis.set_title(f"{title}\nshape={stats['shape']} min={stats['min']:.4g} max={stats['max']:.4g} mean={stats['mean']:.4g} std={stats['std']:.4g}")
    axis.set_xlabel("feature / optical column");axis.set_ylabel("token / optical row")
    colorbar=figure.colorbar(image,ax=axis);colorbar.set_label("detector intensity" if value_type=="intensity" else value_type.replace("_"," "))
    figure.tight_layout();path.parent.mkdir(parents=True,exist_ok=True);figure.savefig(path,dpi=150);plt.close(figure)


def save_input_image(image: Image.Image, path: Path, preview_size: int|None=None) -> None:
    value=image.convert("RGB")
    if preview_size is not None:
        value=value.copy();value.thumbnail((preview_size,preview_size),Image.Resampling.BICUBIC)
    path.parent.mkdir(parents=True,exist_ok=True);value.save(path)


@torch.no_grad()
def save_debug_example(sample: dict[str,Any], root: Path, epoch: int, phase: str, settings: Any) -> tuple[dict[str,Any],dict[str,Any]]:
    sample_index=int(sample["sample_index"]);directory=root/f"epoch_{epoch:04d}"/f"sample_{sample_index:06d}"
    directory.mkdir(parents=True,exist_ok=True);percentile=float(settings.debug_visualization_percentile_clip);max_tokens=int(settings.debug_visualization_max_tokens)
    input_path=directory/"input_original.png";save_input_image(sample["image"],input_path)
    save_input_image(sample["image"],directory/"input_processor_resized_or_preview.png",int(math.sqrt(settings.processor_max_pixels)))
    raw=bool(settings.debug_visualization_save_raw_tensors);field_report={};negative_counts={"vision":0,"language":0}
    for side in ("vision","language"):
        input_field=sample[f"{side}_input_field"]
        save_tensor_heatmap(input_field,directory/f"{side}_optical_input_field.png",f"{side} optical input field","intensity",percentile,False,max_tokens)
        if raw:torch.save(input_field,directory/f"{side}_optical_input_field.pt")
        field_report[f"{side}_optical_input_field"]=field_metrics(input_field)
        layer_stats=[]
        for layer,intensity in enumerate(sample[f"{side}_detector_fields"],1):
            stats=field_metrics(intensity);layer_stats.append(stats);negative_counts[side]+=stats["num_negative"]
            save_tensor_heatmap(intensity,directory/f"{side}_detector_intensity_layer_{layer}.png",f"{side} detector intensity layer {layer}","intensity",percentile,False,max_tokens)
            if raw:torch.save(intensity,directory/f"{side}_detector_intensity_layer_{layer}.pt")
        field_report[f"{side}_detector_intensity_layers"]=layer_stats
    if negative_counts["vision"] or negative_counts["language"]:
        warnings.warn("detector intensity contains negative values. This should not happen after OpticalConversion. Check whether visualization is using hidden/delta tensors rather than detector intensity.",RuntimeWarning)
    _write_json(directory/"optical_field_metrics.json",field_report)
    vision_directory=directory/"vision_hidden";vision_directory.mkdir(exist_ok=True)
    student_vision=sample["student_vision_hidden"];teacher_vision=sample["teacher_vision_hidden"]
    vision_diff=student_vision-teacher_vision
    _hidden_plots(student_vision,teacher_vision,vision_directory,"vision_hidden",percentile,max_tokens,raw)
    cosine=F.cosine_similarity(student_vision.float(),teacher_vision.float(),dim=-1)
    save_tensor_heatmap(cosine[:,None],vision_directory/"vision_hidden_cosine_per_token.png","vision cosine per token","similarity",percentile,False,max_tokens)
    vision_metrics=hidden_similarity_metrics(student_vision,teacher_vision)
    student_ln=F.layer_norm(student_vision.float(),(student_vision.shape[-1],));teacher_ln=F.layer_norm(teacher_vision.float(),(teacher_vision.shape[-1],))
    ln_metrics=hidden_similarity_metrics(student_ln,teacher_ln)
    vision_metrics.update({f"raw_{key}":value for key,value in list(vision_metrics.items())})
    vision_metrics.update({f"ln_{key}":value for key,value in ln_metrics.items()})
    _write_json(vision_directory/"vision_hidden_metrics.json",vision_metrics)
    answer_directory=directory/"answer_hidden";answer_directory.mkdir(exist_ok=True)
    student_answer=sample["student_answer_hidden"];teacher_answer=sample["teacher_answer_hidden"]
    _hidden_plots(student_answer,teacher_answer,answer_directory,"answer_hidden",percentile,max_tokens,raw)
    answer_metrics=hidden_similarity_metrics(student_answer,teacher_answer);_write_json(answer_directory/"answer_hidden_metrics.json",answer_metrics)
    language_sequence=sample.get("student_language_hidden_sequence")
    if language_sequence is not None:
        save_tensor_heatmap(language_sequence,directory/"student_language_hidden_sequence.png","student language hidden sequence","hidden",percentile,True,max_tokens)
        if raw:torch.save(language_sequence,directory/"student_language_hidden_sequence.pt")
    for side in ("vision","language"):
        delta=sample.get(f"{side}_delta")
        if delta is not None:
            save_tensor_heatmap(delta,directory/f"{side}_delta_heatmap.png",f"{side} restored optical delta","delta",percentile,True,max_tokens)
            if raw:torch.save(delta,directory/f"{side}_delta.pt")
    student_logits=sample["student_logits"].float();teacher_logits=sample["teacher_logits"].float()
    _write_json(directory/"logits.json",{"student_logits":student_logits.tolist(),"teacher_logits":teacher_logits.tolist(),
        "student_softmax":student_logits.softmax(-1).tolist(),"teacher_softmax":teacher_logits.softmax(-1).tolist()})
    metadata=dict(sample["metadata"]);metadata.update({"epoch":epoch,"phase":phase});_write_json(directory/"metadata.json",metadata)
    row={"epoch":epoch,"phase":phase,"sample_index":sample_index,"true_name":metadata["true_name"],
         "teacher_pred_name":metadata["teacher_pred_name"],"student_pred_name":metadata["pred_name"],
         "correct":metadata["correct"],"vision_hidden_mse":vision_metrics["mse"],
         "vision_hidden_cosine_mean_token":vision_metrics["cosine_mean_token"],"answer_hidden_mse":answer_metrics["mse"],
         "answer_hidden_cosine":answer_metrics["cosine"],"input_image_path":str(input_path),"debug_dir":str(directory)}
    summary={"vision_detector_negative_count":negative_counts["vision"],"language_detector_negative_count":negative_counts["language"],
             "vision_cosine":vision_metrics["cosine_mean_token"],"answer_cosine":answer_metrics["cosine"]}
    return row,summary


def append_debug_index(path: Path, rows: Sequence[dict[str,Any]]) -> None:
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True);exists=path.is_file()
    with path.open("a",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=INDEX_FIELDS)
        if not exists:writer.writeheader()
        writer.writerows({name:row.get(name) for name in INDEX_FIELDS} for row in rows)


def _hidden_plots(student:torch.Tensor,teacher:torch.Tensor,directory:Path,prefix:str,percentile:float,max_tokens:int,raw:bool)->None:
    diff=student-teacher
    save_tensor_heatmap(student,directory/f"student_{prefix}.png",f"student {prefix}","hidden",percentile,True,max_tokens)
    save_tensor_heatmap(teacher,directory/f"teacher_{prefix}.png",f"teacher {prefix}","hidden",percentile,True,max_tokens)
    save_tensor_heatmap(diff.abs(),directory/f"{prefix}_abs_diff.png",f"{prefix} absolute difference","absolute_difference",percentile,False,max_tokens)
    save_tensor_heatmap(diff,directory/f"{prefix}_signed_diff.png",f"{prefix} signed difference","signed_difference",percentile,True,max_tokens)
    if prefix=="vision_hidden":
        student_ln=F.layer_norm(student.float(),(student.shape[-1],));teacher_ln=F.layer_norm(teacher.float(),(teacher.shape[-1],));ln_diff=student_ln-teacher_ln
        save_tensor_heatmap(student_ln,directory/"student_vision_hidden_layernorm.png","student vision hidden LayerNorm","hidden",percentile,True,max_tokens)
        save_tensor_heatmap(teacher_ln,directory/"teacher_vision_hidden_layernorm.png","teacher vision hidden LayerNorm","hidden",percentile,True,max_tokens)
        save_tensor_heatmap(ln_diff.abs(),directory/"vision_hidden_layernorm_abs_diff.png","vision LayerNorm absolute difference","absolute_difference",percentile,False,max_tokens)
        save_tensor_heatmap(ln_diff,directory/"vision_hidden_layernorm_signed_diff.png","vision LayerNorm signed difference","signed_difference",percentile,True,max_tokens)
    if raw:
        torch.save(student,directory/f"student_{prefix}.pt");torch.save(teacher,directory/f"teacher_{prefix}.pt");torch.save(diff,directory/f"{prefix}_signed_diff.pt")


def _pearson(left:torch.Tensor,right:torch.Tensor)->float:
    left=left-left.mean();right=right-right.mean();denominator=left.norm()*right.norm()
    return float((left*right).sum()/denominator.clamp_min(1e-12))


def _spearman(left:torch.Tensor,right:torch.Tensor)->float:
    left_rank=torch.argsort(torch.argsort(left)).float();right_rank=torch.argsort(torch.argsort(right)).float()
    return _pearson(left_rank,right_rank)


def _write_json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
