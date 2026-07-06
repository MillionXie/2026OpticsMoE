from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any,Sequence

import torch
from torch import nn

from .data import DataBundle,make_loader
from .metrics import classification_metrics,write_confusion_csv,write_history,write_json
from .models import Optical5EnhancedTimeOfDayClassifier
from .visualization import save_confusion_matrix,save_optical_diagnostics,save_training_curves


def train_model(model:nn.Module,data:DataBundle,settings:Any,device:torch.device)->list[dict[str,Any]]:
    train_loader=make_loader(data.train,settings.batch_size,settings.num_workers,True,settings.seed)
    val_loader=make_loader(data.validation,settings.batch_size,settings.num_workers,False,settings.seed+1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings.learning_rate,weight_decay=settings.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=settings.epochs);criterion=nn.CrossEntropyLoss();history=[];best_top1=-1.0;best_macro=-1.0
    diagnostic_images=next(iter(val_loader))[0][:8].to(device)
    for epoch in range(1,settings.epochs+1):
        epoch_started=time.perf_counter();train_started=time.perf_counter();model.train();total_loss=0.0;targets=[];predictions=[];seen=0
        for batch_index,(images,labels,indices,paths) in enumerate(train_loader,1):
            images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True);logits=model(images);loss=criterion(logits,labels);loss.backward();optimizer.step()
            count=len(labels);seen+=count;total_loss+=float(loss.detach())*count;targets.extend(labels.detach().cpu().tolist());predictions.extend(logits.argmax(1).detach().cpu().tolist())
            if batch_index%settings.log_interval_batches==0 or batch_index==len(train_loader):
                running=sum(int(a==b) for a,b in zip(targets,predictions))/seen
                print(f"epoch {epoch}/{settings.epochs} batch {batch_index}/{len(train_loader)}\nloss={total_loss/seen:.6f}\nrunning_top1={running:.4f}\nlr={optimizer.param_groups[0]['lr']:.3e}",flush=True)
        train_time=time.perf_counter()-train_started;validation_started=time.perf_counter();validation=evaluate(model,val_loader,device,data.class_names);validation_time=time.perf_counter()-validation_started
        train_metrics=classification_metrics(targets,predictions,data.class_names)
        row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],"train_loss":total_loss/seen,"train_top1_accuracy":train_metrics["top1_accuracy"],"train_macro_f1":train_metrics["macro_f1"],"validation_loss":validation["metrics"]["loss"],"validation_top1_accuracy":validation["metrics"]["top1_accuracy"],"validation_macro_f1":validation["metrics"]["macro_f1"],"validation_balanced_accuracy":validation["metrics"]["balanced_accuracy"],"epoch_time_sec":time.perf_counter()-epoch_started,"train_time_sec":train_time,"validation_time_sec":validation_time}
        history.append(row);write_history(settings.output_dir/"metrics"/"training_history.csv",history);write_json(settings.output_dir/"metrics"/"training_latest.json",row)
        if epoch%settings.save_predictions_interval_epochs==0:_write_predictions(settings.output_dir/"metrics"/f"validation_predictions_epoch_{epoch:04d}.csv",validation,data.class_names)
        improved=row["validation_top1_accuracy"]>best_top1 or row["validation_macro_f1"]>best_macro
        if improved:
            best_top1=max(best_top1,row["validation_top1_accuracy"]);best_macro=max(best_macro,row["validation_macro_f1"]);_save_checkpoint(settings.output_dir/"checkpoints"/"best.pt",model,optimizer,scheduler,epoch,row);write_json(settings.output_dir/"metrics"/"best_validation.json",{"epoch":epoch,**validation["metrics"]})
        _save_checkpoint(settings.output_dir/"checkpoints"/"last.pt",model,optimizer,scheduler,epoch,row)
        if isinstance(model,Optical5EnhancedTimeOfDayClassifier) and (epoch==1 or epoch%settings.save_interval_epochs==0 or epoch==settings.epochs):save_optical_diagnostics(model,diagnostic_images,settings.output_dir/"figures",epoch)
        save_training_curves(history,settings.output_dir/"figures"/"training_curves.png");scheduler.step()
        print(f"[epoch {epoch:03d}] val_top1={row['validation_top1_accuracy']:.4f} val_macro_f1={row['validation_macro_f1']:.4f} time={row['epoch_time_sec']:.1f}s",flush=True)
    return history


@torch.no_grad()
def evaluate(model:nn.Module,loader:Any,device:torch.device,class_names:Sequence[str])->dict[str,Any]:
    model.eval();criterion=nn.CrossEntropyLoss(reduction="sum");loss=0.0;logits_all=[];labels_all=[];indices_all=[];paths_all=[]
    for images,labels,indices,paths in loader:
        images=images.to(device,non_blocking=True);labels_device=labels.to(device,non_blocking=True);logits=model(images);loss+=float(criterion(logits,labels_device));logits_all.append(logits.cpu());labels_all.append(labels);indices_all.extend(indices.tolist());paths_all.extend(paths)
    logits=torch.cat(logits_all);labels=torch.cat(labels_all);predictions=logits.argmax(1);metrics=classification_metrics(labels.tolist(),predictions.tolist(),list(class_names));metrics["loss"]=loss/max(1,len(labels))
    return {"metrics":metrics,"logits":logits,"labels":labels,"indices":indices_all,"paths":paths_all}


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
    fieldnames=["sample_index","image_path","true_label","true_name","pred_label","pred_name","correct",*[f"logit_{name}" for name in names]]
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames);writer.writeheader()
        for index,image_path,truth,pred,values in zip(result["indices"],result["paths"],result["labels"].tolist(),predictions,result["logits"].tolist()):
            row={"sample_index":index,"image_path":image_path,"true_label":truth,"true_name":names[truth],"pred_label":pred,"pred_name":names[pred],"correct":truth==pred};row.update({f"logit_{name}":value for name,value in zip(names,values)});writer.writerow(row)


def _save_checkpoint(path:Path,model:nn.Module,optimizer:Any,scheduler:Any,epoch:int,row:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True);torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"scheduler_state_dict":scheduler.state_dict(),"metrics":row},path)

