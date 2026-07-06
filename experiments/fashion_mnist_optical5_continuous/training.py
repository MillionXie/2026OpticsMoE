from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .data import DataBundle,labels_of,make_loader
from .metrics import classification_metrics,write_csv,write_json
from .models import FashionMNISTOptical5Continuous
from .sampling import EpochClassMixedSampler
from .visualization import save_confusion,save_curves,save_diagnostics


def train_model(model:FashionMNISTOptical5Continuous,data:DataBundle,settings:Any,device:torch.device)->list[dict[str,Any]]:
    sampler=EpochClassMixedSampler(range(len(data.train)),labels_of(data.train),10,settings.batch_size,settings.seed,settings.train_samples_per_class_per_epoch);train_loader=make_loader(data.train,settings.batch_size,settings.num_workers,False,settings.seed,sampler);val_loader=make_loader(data.validation,settings.batch_size,settings.num_workers,False,settings.seed+1);optimizer=torch.optim.AdamW(model.parameters(),lr=settings.learning_rate,weight_decay=settings.weight_decay);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings.epochs);criterion=nn.CrossEntropyLoss();history=[];phase_rows=[];best=-1.0;diagnostic_images=next(iter(val_loader))[0][:4].to(device);initial=[layer.phase_mask.detach().clone() for layer in model.layers];save_diagnostics(model,diagnostic_images,settings.output_dir/"figures",0)
    for epoch in range(1,settings.epochs+1):
        started=time.perf_counter();model.train();model.set_epoch(epoch);sampler.set_epoch(epoch);totals={name:0.0 for name in ("total","class","region","concentration","smoothness")};targets=[];predictions=[];seen=0
        print(f"[sampling] epoch={epoch} samples={len(sampler)} per_class={sampler.epoch_class_counts()} shuffled=True",flush=True)
        for batch_index,(images,labels,_,_) in enumerate(train_loader,1):
            images=images.to(device);labels=labels.to(device);optimizer.zero_grad(set_to_none=True);logits,aux=model(images,return_aux=True);loss_class=criterion(logits,labels);loss_region=criterion(aux["region_logits"],labels);loss_concentration=-torch.log(aux["detector_fraction"].clamp_min(1e-8)).mean();loss_smooth=model.phase_tv_loss();loss=loss_class+model.detector_region_loss_weight*loss_region+model.detector_concentration_loss_weight*loss_concentration+model.phase_smoothness_weight*loss_smooth;loss.backward();optimizer.step();count=len(labels);seen+=count
            for name,value in (("total",loss),("class",loss_class),("region",loss_region),("concentration",loss_concentration),("smoothness",loss_smooth)):totals[name]+=float(value.detach())*count
            targets.extend(labels.cpu().tolist());predictions.extend(logits.argmax(1).detach().cpu().tolist())
            if batch_index%settings.log_interval_batches==0 or batch_index==len(train_loader):print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(train_loader)} loss={totals['total']/seen:.5f} top1={sum(a==b for a,b in zip(targets,predictions))/seen:.4f}",flush=True)
        validation=evaluate(model,val_loader,device,data.class_names);train_metrics=classification_metrics(targets,predictions,data.class_names);row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],"loss_total":totals["total"]/seen,"loss_classification":totals["class"]/seen,"loss_region":totals["region"]/seen,"loss_concentration":totals["concentration"]/seen,"loss_smoothness_unweighted":totals["smoothness"]/seen,"phase_smoothness_weight":model.phase_smoothness_weight,"phase_dropout_enabled":bool(model.layers[0].phase_dropout.enabled),"phase_dropout_active":bool(model.layers[0].phase_dropout_active),"train_top1":train_metrics["top1_accuracy"],"validation_top1":validation["metrics"]["top1_accuracy"],"validation_macro_f1":validation["metrics"]["macro_f1"],"validation_detector_region_accuracy":validation["metrics"]["detector_region_accuracy"],"epoch_time_sec":time.perf_counter()-started};history.append(row);write_csv(settings.output_dir/"metrics"/"training_history.csv",history);write_json(settings.output_dir/"metrics"/"training_latest.json",row);phase_rows.extend(_phase_statistics(model,initial,epoch));write_csv(settings.output_dir/"metrics"/"phase_statistics.csv",phase_rows)
        if row["validation_macro_f1"]>best:best=row["validation_macro_f1"];_checkpoint(settings.output_dir/"checkpoints"/"best.pt",model,optimizer,scheduler,epoch,row);write_json(settings.output_dir/"metrics"/"best_validation.json",validation["metrics"])
        _checkpoint(settings.output_dir/"checkpoints"/"last.pt",model,optimizer,scheduler,epoch,row)
        if epoch==1 or epoch%settings.save_interval_epochs==0 or epoch==settings.epochs:save_diagnostics(model,diagnostic_images,settings.output_dir/"figures",epoch)
        save_curves(history,settings.output_dir/"figures"/"training_curves.png");scheduler.step()
    return history


@torch.no_grad()
def evaluate(model:FashionMNISTOptical5Continuous,loader:Any,device:torch.device,names:list[str])->dict[str,Any]:
    model.eval();logits_all=[];labels_all=[];indices=[];paths=[];regions=[];fractions=[]
    for images,labels,batch_indices,batch_paths in loader:
        logits,aux=model(images.to(device),return_aux=True);logits_all.append(logits.cpu());labels_all.append(labels);regions.append(aux["region_logits"].cpu());fractions.append(aux["detector_fraction"].cpu());indices.extend(batch_indices.tolist());paths.extend(batch_paths)
    logits=torch.cat(logits_all);labels=torch.cat(labels_all);region_logits=torch.cat(regions);detector_fractions=torch.cat(fractions);metrics=classification_metrics(labels.tolist(),logits.argmax(1).tolist(),names);metrics.update({"detector_region_accuracy":float(region_logits.argmax(1).eq(labels).float().mean()),"mean_detector_energy_fraction":float(detector_fractions.mean())});return {"metrics":metrics,"logits":logits,"labels":labels,"region_logits":region_logits,"indices":indices,"paths":paths}


def test_model(model:FashionMNISTOptical5Continuous,data:DataBundle,settings:Any,device:torch.device)->dict[str,Any]:
    checkpoint=settings.output_dir/"checkpoints"/"best.pt";model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)["model_state_dict"]);result=evaluate(model,make_loader(data.test,settings.batch_size,settings.num_workers,False,settings.seed+2),device,data.class_names);write_json(settings.output_dir/"metrics"/"test_metrics.json",result["metrics"]);_predictions(settings.output_dir/"metrics"/"test_predictions.csv",result,data.class_names);save_confusion(result["metrics"]["confusion_matrix"],data.class_names,settings.output_dir/"figures"/"confusion_matrix.png");return result["metrics"]


def _phase_statistics(model:FashionMNISTOptical5Continuous,initial:list[torch.Tensor],epoch:int)->list[dict[str,Any]]:
    rows=[]
    for index,(layer,start) in enumerate(zip(model.layers,initial),1):
        phase=layer.phase_mask.detach().float();wrapped=torch.remainder(phase,2*math.pi);resultant=torch.abs(torch.exp(1j*wrapped).mean());grad=layer.phase_mask.grad
        rows.append({"epoch":epoch,"layer":index,"raw_mean":float(phase.mean()),"raw_std":float(phase.std()),"raw_min":float(phase.min()),"raw_max":float(phase.max()),"update_rms":float((phase-start.to(phase.device)).square().mean().sqrt()),"circular_variance":float(1-resultant),"total_variation":float((phase[1:]-phase[:-1]).abs().mean()+(phase[:,1:]-phase[:,:-1]).abs().mean()),"gradient_norm_last_batch":float(grad.norm()) if grad is not None else 0.0,"dropout_mask_present":layer.last_phase_dropout_mask is not None})
    return rows


def _predictions(path:Path,result:dict[str,Any],names:list[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True);pred=result["logits"].argmax(1).tolist();region=result["region_logits"].argmax(1).tolist();fields=["sample_index","true_label","true_name","pred_label","pred_name","region_pred_label","region_pred_name","correct",*[f"logit_{name}" for name in names]]
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for index,truth,prediction,region_prediction,values in zip(result["indices"],result["labels"].tolist(),pred,region,result["logits"].tolist()):row={"sample_index":index,"true_label":truth,"true_name":names[truth],"pred_label":prediction,"pred_name":names[prediction],"region_pred_label":region_prediction,"region_pred_name":names[region_prediction],"correct":truth==prediction,**{f"logit_{name}":value for name,value in zip(names,values)}};writer.writerow(row)


def _checkpoint(path:Path,model:nn.Module,optimizer:Any,scheduler:Any,epoch:int,metrics:dict[str,Any])->None:path.parent.mkdir(parents=True,exist_ok=True);torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"metrics":metrics},path)

