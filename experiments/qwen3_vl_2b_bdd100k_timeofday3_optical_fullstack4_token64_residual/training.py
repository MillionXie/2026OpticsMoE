from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset, TensorDataset

from .datasets import indexed_collate, labels_of, make_indexed_loader, stratified_split_indices
from .features import move_inputs, multimodal_forward_features, pool_answer_hidden_state, preprocess_image_text
from .io_utils import write_csv, write_json
from .metrics import metrics_from_logits
from .modeling import MLPHead
from .sampling import EpochClassMixedSampler
from .teacher_cache import TeacherCacheStore, load_cached_tensor, load_teacher_logits, write_teacher_logits
from .visualization import save_confusion, save_stack_diagnostics, save_training_curves


def train_teacher_head(train_store: TeacherCacheStore, test_store: TeacherCacheStore, settings: Any,
                       class_names: Sequence[str], device: torch.device) -> MLPHead:
    features=load_cached_tensor(train_store,"teacher_answer_hidden").float(); labels=load_cached_tensor(train_store,"labels").long()
    train_idx,val_idx=_split_tensor_labels(labels,settings.validation_fraction,settings.seed)
    head=MLPHead(features.shape[1],settings.hidden_dim,len(class_names),settings.dropout).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=settings.learning_rate,weight_decay=settings.weight_decay)
    history=[]; best=-1.0
    for epoch in range(1,settings.epochs+1):
        head.train(); total=0.0
        loader=DataLoader(TensorDataset(features[train_idx],labels[train_idx]),batch_size=settings.head_batch_size,shuffle=True)
        for x,y in loader:
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True); loss=F.cross_entropy(head(x),y); loss.backward(); optimizer.step(); total+=float(loss)*len(y)
        val_logits=_head_logits(head,features[val_idx],settings.head_batch_size,device); metrics=metrics_from_logits(val_logits,labels[val_idx],class_names)
        row={"epoch":epoch,"loss":total/len(train_idx),**{k:metrics[k] for k in ("top1_accuracy","top5_accuracy","macro_f1","balanced_accuracy")}}
        history.append(row); write_csv(settings.output_dir/"metrics"/"teacher_training_history.csv",history,list(row))
        if metrics["macro_f1"]>best:
            best=metrics["macro_f1"]; _save_head(head,settings.output_dir/"checkpoints"/"teacher_mlp.pt",settings,len(class_names))
    head=load_head(settings.output_dir/"checkpoints"/"teacher_mlp.pt",settings.dropout,device)
    test_features=load_cached_tensor(test_store,"teacher_answer_hidden").float(); test_labels=load_cached_tensor(test_store,"labels").long()
    test_logits=_head_logits(head,test_features,settings.head_batch_size,device); report=metrics_from_logits(test_logits,test_labels,class_names)
    write_json(settings.output_dir/"metrics"/"teacher_inference.json",report)
    return head


def generate_teacher_logits(head: nn.Module, stores: dict[str,TeacherCacheStore], settings: Any, device: torch.device) -> None:
    for split,store in stores.items():
        features=load_cached_tensor(store,"teacher_answer_hidden").float(); labels=load_cached_tensor(store,"labels").long()
        logits=_head_logits(head,features,settings.head_batch_size,device)
        write_teacher_logits(settings.output_dir,split,logits,labels)


def teacher_inference(head: nn.Module, store: TeacherCacheStore, settings: Any, class_names: Sequence[str], device: torch.device) -> dict[str,Any]:
    features=load_cached_tensor(store,"teacher_answer_hidden").float(); labels=load_cached_tensor(store,"labels").long()
    logits=_head_logits(head,features,settings.head_batch_size,device); report=metrics_from_logits(logits,labels,class_names)
    write_json(settings.output_dir/"metrics"/"teacher_inference.json",report); return report


class CachedStudentDataset(Dataset[Any]):
    def __init__(self, images: Dataset[Any], store: TeacherCacheStore, logits: torch.Tensor) -> None:
        self.images=images; self.store=store; self.logits=logits
    def __len__(self): return len(self.images)
    def __getitem__(self,index:int):
        image,label=self.images[index]; target=self.store.get(index)
        if int(target["label"])!=int(label): raise RuntimeError(f"Teacher cache label mismatch at sample {index}")
        return image,int(label),index,target["image_grid_thw"],target["visual_token_count"],target["teacher_vision_stack_output"],target["teacher_answer_hidden"],self.logits[index]


def cached_collate(batch: Sequence[Any]):
    images,labels,indices,grids,counts,vision,answer,logits=zip(*batch)
    return list(images),torch.tensor(labels),torch.tensor(indices),torch.stack(grids),torch.stack(counts).long(),list(vision),torch.stack(answer),torch.stack(logits)


def train_student(model: nn.Module, processor: Any, replacement: Any, head: MLPHead, train_dataset: Dataset[Any],
                  validation_dataset: Dataset[Any], train_store: TeacherCacheStore, settings: Any,
                  class_names: Sequence[str], device: torch.device) -> None:
    teacher_logits=load_teacher_logits(settings.output_dir/"teacher_cache"/"train_teacher_logits.pt")
    train_indices,validation_indices=stratified_split_indices(train_dataset,settings.validation_fraction,settings.seed)
    cached=CachedStudentDataset(train_dataset,train_store,teacher_logits)
    sampler=EpochClassMixedSampler(train_indices,labels_of(train_dataset),len(class_names),settings.student_batch_size,settings.seed,settings.train_samples_per_class_per_epoch,settings.teacher_cache_shard_size)
    loader=DataLoader(cached,batch_size=settings.student_batch_size,sampler=sampler,num_workers=0,collate_fn=cached_collate,pin_memory=True)
    val_loader=make_indexed_loader(Subset(train_dataset,validation_indices),settings.inference_batch_size,settings.num_workers,False,settings.seed)
    model.requires_grad_(False); model.eval(); replacement.use_student(); replacement.vision_surrogate.requires_grad_(True); replacement.language_surrogate.requires_grad_(True); head.requires_grad_(True)
    optimizer=torch.optim.AdamW([*replacement.trainable_parameters(),*head.parameters()],lr=settings.learning_rate,weight_decay=settings.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings.epochs)
    history=[]; best_top1=-1.0; best_macro=-1.0; first_shapes_written=False
    for epoch in range(1,settings.epochs+1):
        epoch_started=time.perf_counter(); train_started=time.perf_counter(); sampler.set_epoch(epoch); print(f"[sampling] epoch={epoch} samples={len(sampler)} per_class={sampler.epoch_class_counts()} shuffled=True",flush=True)
        replacement.vision_surrogate.train(); replacement.language_surrogate.train(); head.train()
        totals={key:0.0 for key in ("total","vision","answer","kd","ce")}; seen=0; logits_epoch=[]; labels_epoch=[]
        for batch_index,(images,labels,indices,cached_grids,cached_counts,teacher_vision,teacher_answer,teacher_batch_logits) in enumerate(loader,1):
            inputs_cpu=preprocess_image_text(processor,images,settings.classification_prompt); inputs=move_inputs(inputs_cpu,device)
            if not torch.equal(inputs_cpu["image_grid_thw"].cpu(),cached_grids.cpu()):
                raise RuntimeError("Current processor image_grid_thw differs from teacher cache; regenerate teacher_precompute")
            labels=labels.to(device); teacher_answer=teacher_answer.float().to(device); teacher_batch_logits=teacher_batch_logits.float().to(device)
            replacement.prepare_student_batch(inputs["attention_mask"]); optimizer.zero_grad(set_to_none=True)
            hidden=multimodal_forward_features(model,inputs); answer,_=pool_answer_hidden_state(hidden,inputs["attention_mask"]); student_logits=head(answer)
            if replacement.vision_surrogate.last_token_counts != cached_counts.tolist():
                raise RuntimeError("Current visual token boundaries differ from teacher cache; regenerate teacher_precompute")
            student_vision_groups=list(replacement.vision_surrogate.last_output.split(replacement.vision_surrogate.last_token_counts,dim=0))
            loss_vision=torch.stack([F.mse_loss(F.layer_norm(s.float(),(s.shape[-1],)),F.layer_norm(t.float().to(device),(t.shape[-1],))) for s,t in zip(student_vision_groups,teacher_vision)]).mean()
            loss_answer=F.mse_loss(F.layer_norm(answer.float(),(answer.shape[-1],)),F.layer_norm(teacher_answer,(teacher_answer.shape[-1],)))
            temperature=settings.distill_temperature
            loss_kd=temperature**2*F.kl_div(F.log_softmax(student_logits/temperature,dim=1),F.softmax(teacher_batch_logits/temperature,dim=1),reduction="batchmean")
            loss_ce=F.cross_entropy(student_logits,labels)
            loss_total=settings.loss_vision_weight*loss_vision+settings.loss_answer_weight*loss_answer+settings.loss_kd_weight*loss_kd+settings.loss_ce_weight*loss_ce
            loss_total.backward(); optimizer.step(); count=len(labels); seen+=count
            for key,value in (("total",loss_total),("vision",loss_vision),("answer",loss_answer),("kd",loss_kd),("ce",loss_ce)): totals[key]+=float(value.detach())*count
            logits_epoch.append(student_logits.detach().cpu()); labels_epoch.append(labels.detach().cpu())
            if not first_shapes_written:
                _write_first_shapes(settings,inputs_cpu,teacher_vision,teacher_answer,answer,replacement,student_logits); first_shapes_written=True
            if batch_index%settings.log_interval_batches==0 or batch_index==len(loader):
                running=metrics_from_logits(torch.cat(logits_epoch),torch.cat(labels_epoch),class_names)
                scales=_residual_scale_values(replacement)
                print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(loader)}\nloss_total={totals['total']/seen:.5f} loss_vision={totals['vision']/seen:.5f} loss_answer={totals['answer']/seen:.5f} loss_kd={totals['kd']/seen:.5f} loss_ce={totals['ce']/seen:.5f} running_top1={running['top1_accuracy']:.4f} lr={optimizer.param_groups[0]['lr']:.3e} alpha_v={scales['alpha_v']:.6f} beta_v={scales['beta_v']:.6f} alpha_l={scales['alpha_l']:.6f} beta_l={scales['beta_l']:.6f}",flush=True)
        train_time=time.perf_counter()-train_started; validation_started=time.perf_counter()
        validation=evaluate_student(model,processor,replacement,head,val_loader,class_names,settings,device)
        validation_time=time.perf_counter()-validation_started; train_metrics=metrics_from_logits(torch.cat(logits_epoch),torch.cat(labels_epoch),class_names)
        row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],"loss_total":totals["total"]/seen,"loss_vision":totals["vision"]/seen,"loss_answer":totals["answer"]/seen,"loss_kd":totals["kd"]/seen,"loss_ce":totals["ce"]/seen,"train_top1_accuracy":train_metrics["top1_accuracy"],"train_macro_f1":train_metrics["macro_f1"],"validation_top1_accuracy":validation["metrics"]["top1_accuracy"],"validation_top5_accuracy":validation["metrics"]["top5_accuracy"],"validation_macro_f1":validation["metrics"]["macro_f1"],"validation_balanced_accuracy":validation["metrics"]["balanced_accuracy"],"epoch_time_sec":time.perf_counter()-epoch_started,"train_time_sec":train_time,"validation_time_sec":validation_time,**_residual_scale_values(replacement)}
        history.append(row); _write_student_epoch_outputs(settings,history,row,validation,epoch,class_names)
        improved=row["validation_top1_accuracy"]>best_top1 or row["validation_macro_f1"]>best_macro
        if improved:
            best_top1=max(best_top1,row["validation_top1_accuracy"]); best_macro=max(best_macro,row["validation_macro_f1"])
            _save_student_parts(settings,replacement,head,"best"); write_json(settings.output_dir/"metrics"/"best_validation.json",row)
        _save_student_parts(settings,replacement,head,"last")
        if epoch%settings.save_visualization_interval_epochs==0 or epoch==1 or epoch==settings.epochs:
            save_stack_diagnostics(replacement.vision_surrogate,settings.output_dir/"figures"/"vision_phase_masks",settings.output_dir/"figures"/"vision_light_fields",epoch)
            save_stack_diagnostics(replacement.language_surrogate,settings.output_dir/"figures"/"language_phase_masks",settings.output_dir/"figures"/"language_light_fields",epoch)
        save_training_curves(history,settings.output_dir/"figures"/"student_training_curves.png"); scheduler.step()
    write_json(settings.output_dir/"metrics"/"student_training.json",{"epochs":settings.epochs,"best_validation_top1":best_top1,"best_validation_macro_f1":best_macro,"history_rows":len(history),**_residual_scale_values(replacement)})


@torch.no_grad()
def evaluate_student(model: nn.Module, processor: Any, replacement: Any, head: nn.Module, loader: Any,
                     class_names: Sequence[str], settings: Any, device: torch.device) -> dict[str,Any]:
    replacement.use_student(); replacement.vision_surrogate.eval(); replacement.language_surrogate.eval(); head.eval()
    logits_chunks=[]; labels_chunks=[]; indices_all=[]
    for batch_index,(images,labels,indices) in enumerate(loader):
        if settings.benchmark_batches is not None and batch_index>=settings.benchmark_batches: break
        inputs=move_inputs(preprocess_image_text(processor,images,settings.classification_prompt),device)
        replacement.prepare_student_batch(inputs["attention_mask"]); hidden=multimodal_forward_features(model,inputs); answer,_=pool_answer_hidden_state(hidden,inputs["attention_mask"])
        logits_chunks.append(head(answer).cpu()); labels_chunks.append(labels); indices_all.extend(indices.tolist())
    logits=torch.cat(logits_chunks); labels=torch.cat(labels_chunks)
    return {"metrics":metrics_from_logits(logits,labels,class_names),"logits":logits,"labels":labels,"indices":indices_all}


def _write_student_epoch_outputs(settings: Any, history: list[dict], row: dict, validation: dict, epoch: int, names: Sequence[str]) -> None:
    write_csv(settings.output_dir/"metrics"/"student_training_history.csv",history,list(row)); write_json(settings.output_dir/"metrics"/"student_training_latest.json",row)
    if epoch%settings.save_predictions_interval_epochs==0:
        rows=_prediction_rows(validation["indices"],validation["labels"],validation["logits"],names)
        write_csv(settings.output_dir/"metrics"/f"validation_predictions_epoch_{epoch:04d}.csv",rows,list(rows[0]))
        _write_confusion_csv(settings.output_dir/"metrics"/f"validation_confusion_matrix_epoch_{epoch:04d}.csv",validation["metrics"]["confusion_matrix"],names)


def save_student_inference(result: dict[str,Any], settings: Any, names: Sequence[str], replacement: Any | None=None) -> list[dict]:
    report=dict(result["metrics"])
    if replacement is not None:
        report.update(_residual_scale_values(replacement))
    write_json(settings.output_dir/"metrics"/"student_inference.json",report)
    rows=_prediction_rows(result["indices"],result["labels"],result["logits"],names)
    write_csv(settings.output_dir/"metrics"/"student_predictions.csv",rows,list(rows[0])); _write_confusion_csv(settings.output_dir/"metrics"/"student_confusion_matrix.csv",result["metrics"]["confusion_matrix"],names)
    save_confusion(result["metrics"]["confusion_matrix"],names,settings.output_dir/"figures"/"student_confusion_matrix.png"); return rows


def _prediction_rows(indices: Sequence[int], labels: torch.Tensor, logits: torch.Tensor, names: Sequence[str]) -> list[dict]:
    predictions=logits.argmax(1); rows=[]
    for i,(index,truth,pred,values) in enumerate(zip(indices,labels.tolist(),predictions.tolist(),logits.tolist())):
        row={"sample_index":index,"true_label":truth,"true_name":names[truth],"pred_label":pred,"pred_name":names[pred],"correct":truth==pred}
        row.update({f"logit_{name}":value for name,value in zip(names,values)}); rows.append(row)
    return rows


def _write_confusion_csv(path: Path, matrix: list[list[int]], names: Sequence[str]) -> None:
    rows=[{"true\\predicted":name,**{pred:value for pred,value in zip(names,row)}} for name,row in zip(names,matrix)]
    write_csv(path,rows,["true\\predicted",*names])


def _save_student_parts(settings: Any,replacement:Any,head:nn.Module,suffix:str)->None:
    root=settings.output_dir/"checkpoints"; root.mkdir(parents=True,exist_ok=True)
    torch.save(head.state_dict(),root/f"student_mlp_{suffix}.pt"); torch.save(replacement.vision_surrogate.state_dict(),root/f"vision_optical_stack_{suffix}.pt"); torch.save(replacement.language_surrogate.state_dict(),root/f"language_optical_stack_{suffix}.pt")
    write_json(root/f"student_{suffix}_metadata.json",{
        "optical_residual_enabled":settings.optical_residual_enabled,
        "optical_identity_scale_trainable":settings.optical_identity_scale_trainable,
        "optical_modulated_scale_trainable":settings.optical_modulated_scale_trainable,
        **_residual_scale_values(replacement),
    })


def load_student_parts(settings: Any,replacement:Any,head:nn.Module,device:torch.device,suffix:str="best")->None:
    root=settings.output_dir/"checkpoints"
    head.load_state_dict(torch.load(root/f"student_mlp_{suffix}.pt",map_location=device,weights_only=True)); replacement.vision_surrogate.load_state_dict(torch.load(root/f"vision_optical_stack_{suffix}.pt",map_location=device,weights_only=True)); replacement.language_surrogate.load_state_dict(torch.load(root/f"language_optical_stack_{suffix}.pt",map_location=device,weights_only=True))


def _save_head(head:MLPHead,path:Path,settings:Any,num_classes:int)->None:
    path.parent.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":head.state_dict(),"feature_dim":head.feature_dim,"hidden_dim":settings.hidden_dim,"num_classes":num_classes},path)


def load_head(path:Path,dropout:float,device:torch.device)->MLPHead:
    if not path.is_file(): raise FileNotFoundError(f"Teacher MLP missing: {path}. Run teacher_train first.")
    p=torch.load(path,map_location="cpu",weights_only=True); head=MLPHead(p["feature_dim"],p["hidden_dim"],p["num_classes"],dropout); head.load_state_dict(p["state_dict"]); return head.to(device)


@torch.no_grad()
def _head_logits(head:nn.Module,features:torch.Tensor,batch_size:int,device:torch.device)->torch.Tensor:
    head.eval(); return torch.cat([head(features[i:i+batch_size].to(device)).cpu() for i in range(0,len(features),batch_size)])


def _split_tensor_labels(labels:torch.Tensor,fraction:float,seed:int):
    dataset=type("Labels",(),{"labels":labels.tolist(),"__len__":lambda self:len(self.labels)})()
    return [torch.tensor(v,dtype=torch.long) for v in stratified_split_indices(dataset,fraction,seed)]


def _write_first_shapes(settings:Any,inputs:dict,teacher_vision:list,teacher_answer:torch.Tensor,student_answer:torch.Tensor,replacement:Any,logits:torch.Tensor)->None:
    write_json(settings.output_dir/"metrics"/"first_batch_shapes.json",{
        "input_ids":list(inputs["input_ids"].shape),"attention_mask":list(inputs["attention_mask"].shape),"pixel_values":list(inputs["pixel_values"].shape),"image_grid_thw":inputs["image_grid_thw"].tolist(),"visual_token_counts":replacement.vision_surrogate.last_token_counts,"language_token_counts":replacement.language_surrogate.last_token_counts,"teacher_vision_stack_output":[list(x.shape) for x in teacher_vision],"student_vision_stack_output":list(replacement.vision_surrogate.last_output.shape),"teacher_answer_hidden":list(teacher_answer.shape),"student_answer_hidden":list(student_answer.shape),"vision_zero_padded_input_field":list(replacement.vision_surrogate.last_input_fields.shape),"language_zero_padded_input_field":list(replacement.language_surrogate.last_input_fields.shape),"vision_optical_field_shapes":[list(x.shape) for x in replacement.vision_surrogate.last_fields],"language_optical_field_shapes":[list(x.shape) for x in replacement.language_surrogate.last_fields],"logits":list(logits.shape),**_residual_scale_values(replacement)})


def _residual_scale_values(replacement: Any) -> dict[str, float]:
    vision=replacement.vision_surrogate.scale_values(); language=replacement.language_surrogate.scale_values()
    return {
        "alpha_v":vision["modulated_scale"],
        "beta_v":vision["identity_scale"],
        "alpha_l":language["modulated_scale"],
        "beta_l":language["identity_scale"],
    }
