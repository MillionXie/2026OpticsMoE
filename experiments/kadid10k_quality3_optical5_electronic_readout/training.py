from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any,Sequence

import torch
from torch import nn

from .data import DataBundle,labels_of,make_loader
from .metrics import classification_metrics,write_confusion_csv,write_history,write_json
from .models import Optical5ElectronicReadoutQualityClassifier
from .sampling import EpochClassMixedSampler
from .visualization import save_confusion_matrix,save_optical_diagnostics,save_training_curves


def train_model(model:nn.Module,data:DataBundle,settings:Any,device:torch.device)->list[dict[str,Any]]:
    train_sampler=EpochClassMixedSampler(range(len(data.train)),labels_of(data.train),len(data.class_names),settings.batch_size,settings.seed,getattr(settings,"train_samples_per_class_per_epoch",None))
    train_loader=make_loader(data.train,settings.batch_size,settings.num_workers,False,settings.seed,train_sampler)
    val_loader=make_loader(data.validation,settings.batch_size,settings.num_workers,False,settings.seed+1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings.learning_rate,weight_decay=settings.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings.epochs);criterion=nn.CrossEntropyLoss();history=[];best_top1=-1.0;best_macro=-1.0
    diagnostic_images=next(iter(val_loader))[0][:8].to(device)
    for epoch in range(1,settings.epochs+1):
        epoch_started=time.perf_counter();train_started=time.perf_counter();model.train();train_sampler.set_epoch(epoch);print(f"[sampling] epoch={epoch} samples={len(train_sampler)} per_class={train_sampler.epoch_class_counts()} shuffled=True",flush=True)
        if hasattr(model,"set_epoch"):model.set_epoch(epoch)
        loss_totals={"total":0.0,"classification":0.0,"detector_region":0.0,"detector_concentration":0.0};targets=[];predictions=[];detector_predictions=[];seen=0
        for batch_index,(images,labels,indices,metadata) in enumerate(train_loader,1):
            images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True);logits,loss,components,aux=_forward_objective(model,images,labels,criterion);loss.backward();optimizer.step()
            count=len(labels);seen+=count
            for name,value in components.items():loss_totals[name]+=float(value.detach())*count
            targets.extend(labels.detach().cpu().tolist());predictions.extend(logits.argmax(1).detach().cpu().tolist())
            if aux is not None:detector_predictions.extend(aux["region_logits"].argmax(1).detach().cpu().tolist())
            if batch_index%settings.log_interval_batches==0 or batch_index==len(train_loader):
                running=sum(int(a==b) for a,b in zip(targets,predictions))/seen
                detector_running=sum(int(a==b) for a,b in zip(targets,detector_predictions))/seen if detector_predictions else None;detector_text=f" detector_region_top1={detector_running:.4f}" if detector_running is not None else ""
                print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(train_loader)}\nloss_total={loss_totals['total']/seen:.6f} loss_classification={loss_totals['classification']/seen:.6f} loss_detector_region={loss_totals['detector_region']/seen:.6f} loss_detector_concentration={loss_totals['detector_concentration']/seen:.6f}\nrunning_top1={running:.4f}{detector_text}\nlr={optimizer.param_groups[0]['lr']:.3e}",flush=True)
        train_time=time.perf_counter()-train_started;validation_started=time.perf_counter();validation=evaluate(model,val_loader,device,data.class_names);validation_time=time.perf_counter()-validation_started
        train_metrics=classification_metrics(targets,predictions,data.class_names)
        train_detector_top1=sum(int(a==b) for a,b in zip(targets,detector_predictions))/seen if detector_predictions else None
        row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],"train_loss":loss_totals["total"]/seen,"train_classification_loss":loss_totals["classification"]/seen,"train_detector_region_loss":loss_totals["detector_region"]/seen,"train_detector_concentration_loss":loss_totals["detector_concentration"]/seen,"train_top1_accuracy":train_metrics["top1_accuracy"],"train_macro_f1":train_metrics["macro_f1"],"train_detector_region_accuracy":train_detector_top1,"validation_loss":validation["metrics"]["loss"],"validation_classification_loss":validation["metrics"]["classification_loss"],"validation_detector_region_loss":validation["metrics"]["detector_region_loss"],"validation_detector_concentration_loss":validation["metrics"]["detector_concentration_loss"],"validation_detector_region_accuracy":validation["metrics"].get("detector_region_accuracy"),"validation_mean_detector_energy_fraction":validation["metrics"].get("mean_detector_energy_fraction"),"validation_mean_target_region_energy_fraction":validation["metrics"].get("mean_target_region_energy_fraction"),"validation_top1_accuracy":validation["metrics"]["top1_accuracy"],"validation_macro_f1":validation["metrics"]["macro_f1"],"validation_balanced_accuracy":validation["metrics"]["balanced_accuracy"],"phase_dropout_active":bool(hasattr(model,"layers") and model.layers[0].phase_dropout is not None and model.layers[0].phase_dropout_active),"epoch_time_sec":time.perf_counter()-epoch_started,"train_time_sec":train_time,"validation_time_sec":validation_time}
        history.append(row);write_history(settings.output_dir/"metrics"/"training_history.csv",history);write_json(settings.output_dir/"metrics"/"training_latest.json",row)
        if epoch%settings.save_predictions_interval_epochs==0:_write_predictions(settings.output_dir/"metrics"/f"validation_predictions_epoch_{epoch:04d}.csv",validation,data.class_names)
        improved=row["validation_top1_accuracy"]>best_top1 or row["validation_macro_f1"]>best_macro
        if improved:
            best_top1=max(best_top1,row["validation_top1_accuracy"]);best_macro=max(best_macro,row["validation_macro_f1"]);_save_checkpoint(settings.output_dir/"checkpoints"/"best.pt",model,optimizer,scheduler,epoch,row);write_json(settings.output_dir/"metrics"/"best_validation.json",{"epoch":epoch,**validation["metrics"]})
        _save_checkpoint(settings.output_dir/"checkpoints"/"last.pt",model,optimizer,scheduler,epoch,row)
        if epoch==1 or epoch%settings.save_interval_epochs==0 or epoch==settings.epochs:save_optical_diagnostics(model,diagnostic_images,settings.output_dir/"figures",epoch)
        save_training_curves(history,settings.output_dir/"figures"/"training_curves.png");scheduler.step()
        print(f"[epoch {epoch:03d}] val_top1={row['validation_top1_accuracy']:.4f} val_macro_f1={row['validation_macro_f1']:.4f} time={row['epoch_time_sec']:.1f}s",flush=True)
    return history


@torch.no_grad()
def evaluate(model:nn.Module,loader:Any,device:torch.device,class_names:Sequence[str])->dict[str,Any]:
    model.eval();criterion=nn.CrossEntropyLoss();loss_totals={"total":0.0,"classification":0.0,"detector_region":0.0,"detector_concentration":0.0};logits_all=[];labels_all=[];indices_all=[];metadata_all=[];region_logits_all=[];region_fractions_all=[];detector_fractions_all=[]
    for images,labels,indices,metadata in loader:
        images=images.to(device,non_blocking=True);labels_device=labels.to(device,non_blocking=True);logits,_,components,aux=_forward_objective(model,images,labels_device,criterion);count=len(labels)
        for name,value in components.items():loss_totals[name]+=float(value)*count
        logits_all.append(logits.cpu());labels_all.append(labels);indices_all.extend(indices.tolist());metadata_all.extend(metadata)
        if aux is not None:region_logits_all.append(aux["region_logits"].cpu());region_fractions_all.append(aux["region_fractions"].cpu());detector_fractions_all.append(aux["detector_fraction"].cpu())
    logits=torch.cat(logits_all);labels=torch.cat(labels_all);predictions=logits.argmax(1);metrics=classification_metrics(labels.tolist(),predictions.tolist(),list(class_names));samples=max(1,len(labels));metrics.update({"loss":loss_totals["total"]/samples,"classification_loss":loss_totals["classification"]/samples,"detector_region_loss":loss_totals["detector_region"]/samples,"detector_concentration_loss":loss_totals["detector_concentration"]/samples});result={"metrics":metrics,"logits":logits,"labels":labels,"indices":indices_all,"metadata":metadata_all}
    if region_logits_all:
        region_logits=torch.cat(region_logits_all);region_fractions=torch.cat(region_fractions_all);detector_fractions=torch.cat(detector_fractions_all);region_predictions=region_logits.argmax(1);target_fractions=region_fractions.gather(1,labels[:,None]).squeeze(1);metrics.update({"detector_region_accuracy":float(region_predictions.eq(labels).float().mean()),"mean_detector_energy_fraction":float(detector_fractions.mean()),"mean_target_region_energy_fraction":float(target_fractions.mean()),"per_class_mean_target_region_energy_fraction":{name:float(target_fractions[labels.eq(index)].mean()) if labels.eq(index).any() else 0 for index,name in enumerate(class_names)}});result.update({"detector_region_logits":region_logits,"detector_region_fractions":region_fractions,"detector_fractions":detector_fractions})
    return result


def _forward_objective(model:nn.Module,images:torch.Tensor,labels:torch.Tensor,criterion:nn.Module)->tuple[torch.Tensor,torch.Tensor,dict[str,torch.Tensor],dict[str,torch.Tensor]|None]:
    if isinstance(model,Optical5ElectronicReadoutQualityClassifier):
        logits,aux=model(images,return_aux=True);classification=criterion(logits,labels);region=criterion(aux["region_logits"],labels);concentration=-torch.log(aux["detector_fraction"].clamp_min(1e-8)).mean();total=classification+model.detector_region_loss_weight*region+model.detector_concentration_loss_weight*concentration
        return logits,total,{"total":total,"classification":classification,"detector_region":region,"detector_concentration":concentration},aux
    logits=model(images);classification=criterion(logits,labels);zero=classification.new_zeros(())
    return logits,classification,{"total":classification,"classification":classification,"detector_region":zero,"detector_concentration":zero},None


def test_model(model:nn.Module,data:DataBundle,settings:Any,device:torch.device)->dict[str,Any]:
    checkpoint=settings.output_dir/"checkpoints"/"best.pt"
    if not checkpoint.is_file():raise FileNotFoundError(f"Best checkpoint missing: {checkpoint}. Run train first.")
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)["model_state_dict"])
    result=evaluate(model,make_loader(data.test,settings.batch_size,settings.num_workers,False,settings.seed+2),device,data.class_names)
    write_json(settings.output_dir/"metrics"/"test_metrics.json",result["metrics"]);write_json(settings.output_dir/"metrics"/"per_class_metrics.json",result["metrics"]["per_class"])
    write_confusion_csv(settings.output_dir/"metrics"/"confusion_matrix.csv",result["metrics"]["confusion_matrix"],data.class_names);_write_predictions(settings.output_dir/"metrics"/"test_predictions.csv",result,data.class_names)
    save_confusion_matrix(result["metrics"]["confusion_matrix"],data.class_names,settings.output_dir/"figures"/"confusion_matrix.png");return result["metrics"]


def _write_predictions(path:Path,result:dict[str,Any],names:Sequence[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True);predictions=result["logits"].argmax(1).tolist()
    has_detector="detector_region_logits" in result;detector_fields=["detector_pred_label","detector_pred_name","detector_energy_fraction",*[f"detector_region_fraction_{name}" for name in names]] if has_detector else [];fieldnames=["sample_index","image_path","image_name","reference_image","quality_score","distortion_level","distortion_type","true_label","true_name","pred_label","pred_name","correct",*detector_fields,*[f"logit_{name}" for name in names]]
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames);writer.writeheader()
        detector_predictions=result["detector_region_logits"].argmax(1).tolist() if has_detector else [None]*len(predictions);detector_fractions=result["detector_fractions"].tolist() if has_detector else [None]*len(predictions);region_fractions=result["detector_region_fractions"].tolist() if has_detector else [None]*len(predictions)
        for index,metadata,truth,pred,values,detector_pred,detector_fraction,region_values in zip(result["indices"],result["metadata"],result["labels"].tolist(),predictions,result["logits"].tolist(),detector_predictions,detector_fractions,region_fractions):
            row={"sample_index":index,"image_path":metadata.get("image_path"),"image_name":metadata.get("image_name"),"reference_image":metadata.get("reference_image"),"quality_score":metadata.get("quality_score"),"distortion_level":metadata.get("distortion_level"),"distortion_type":metadata.get("distortion_type"),"true_label":truth,"true_name":names[truth],"pred_label":pred,"pred_name":names[pred],"correct":truth==pred};row.update({f"logit_{name}":value for name,value in zip(names,values)})
            if has_detector:row.update({"detector_pred_label":detector_pred,"detector_pred_name":names[detector_pred],"detector_energy_fraction":detector_fraction,**{f"detector_region_fraction_{name}":value for name,value in zip(names,region_values)}})
            writer.writerow(row)


def _save_checkpoint(path:Path,model:nn.Module,optimizer:Any,scheduler:Any,epoch:int,row:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"metrics":row},path)
