import argparse
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from data import load_data, make_loader
from metrics import denormalize_quality, regression_metrics
from model import FullOpticalIQARegressor, soft_quality_targets
from utils import BASE_DIR, choose_device, environment_info, git_info, load_yaml, save_json, save_yaml, set_seed, write_rows
from visualization import save_epoch_artifacts, save_scatter, save_training_curves


def build_optimizer(model,config):
    cfg=config.get("optimizer",{});opt_type=str(cfg.get("type","adam")).strip().lower();kwargs={"lr":float(cfg.get("lr",cfg.get("learning_rate",.01))),"weight_decay":float(cfg.get("weight_decay",0.0))}
    if opt_type=="adam":return torch.optim.Adam(model.parameters(),**kwargs)
    if opt_type=="adamw":return torch.optim.AdamW(model.parameters(),**kwargs)
    raise ValueError(f"Unsupported optimizer.type: {opt_type}. Expected 'adam' or 'adamw'.")


def parse_args():
    parser=argparse.ArgumentParser(description="Train a six-layer pure-optical KADID-10k no-reference IQA regressor.")
    parser.add_argument("--config",default="configs/kadid10k_iqa_regression.yaml");parser.add_argument("--phase",choices=("prepare_data","train","test","all"),default="all")
    parser.add_argument("--device",default=None);parser.add_argument("--epochs",type=int,default=None);parser.add_argument("--output-dir",default=None);parser.add_argument("--smoke-test",action="store_true");parser.add_argument("--disable-visualization",action="store_true")
    return parser.parse_args()


def compute_loss(model,images,targets,config):
    predictions,items=model(images,return_intermediates=True,capture_layer_fields=False);cfg=config.get("loss",{})
    score=F.mse_loss(predictions,targets);soft=soft_quality_targets(targets,model.quality_anchors.to(targets),float(cfg.get("target_distribution_sigma",.08)))
    distribution=F.mse_loss(items["region_probabilities"],soft);concentration=(1.0-items["region_energies"].sum(1)).clamp_min(0).mean()
    total=float(cfg.get("score_mse_weight",1.0))*score+float(cfg.get("detector_distribution_weight",.2))*distribution+float(cfg.get("detector_concentration_weight",.05))*concentration
    return predictions,total,{"score":score,"distribution":distribution,"concentration":concentration}


def run_epoch(model,loader,config,device,optimizer=None,print_freq=50):
    training=optimizer is not None;model.train(training);totals=defaultdict(float);predictions=[];targets_all=[];indices_all=[];count=0
    context=torch.enable_grad() if training else torch.no_grad()
    with context:
        for step,(images,targets,indices) in enumerate(loader,1):
            images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
            if training:optimizer.zero_grad(set_to_none=True)
            prediction,loss,parts=compute_loss(model,images,targets,config)
            if training:loss.backward();optimizer.step()
            batch=len(targets);totals["total"]+=float(loss)*batch
            for key,value in parts.items():totals[key]+=float(value)*batch
            count+=batch;predictions.append(prediction.detach().cpu());targets_all.append(targets.detach().cpu());indices_all.extend(indices.tolist())
            if training and print_freq>0 and step%print_freq==0:print(f"  batch {step}/{len(loader)} loss={totals['total']/count:.6f} rmse={regression_metrics(torch.cat(targets_all),torch.cat(predictions))['rmse']:.4f}")
    predictions=torch.cat(predictions);targets=torch.cat(targets_all);metrics=regression_metrics(targets,predictions)
    metrics.update({"loss":totals["total"]/max(1,count),"score_mse_loss":totals["score"]/max(1,count),"detector_distribution_loss":totals["distribution"]/max(1,count),"detector_concentration_loss":totals["concentration"]/max(1,count)})
    return {"metrics":metrics,"predictions":predictions,"targets":targets,"indices":indices_all}


def fixed_batch(loader,device,count):
    images,targets,indices=next(iter(loader));count=min(int(count),len(images));return images[:count].to(device),targets[:count].to(device),indices[:count]


def save_checkpoint(path,model,optimizer,epoch,metrics,config):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"metrics":metrics,"config":config},path)


def detector_bounds(detector):
    result=[]
    for mask in detector.masks.cpu():
        points=mask.nonzero();y0,x0=points.min(0).values;y1,x1=points.max(0).values+1;result.append([int(y0),int(y1),int(x0),int(x1)])
    return result


def model_report(model,config):
    optimizer=config.get("optimizer",{})
    return {"model":"FullOpticalIQARegressor","task":"KADID-10k no-reference continuous quality regression","path":"grayscale amplitude -> 6 phase planes -> 10 fixed quality-anchor detectors -> fixed energy expectation","input_size":model.input_size,"canvas_size":model.canvas_size,"num_layers":model.num_layers,"quality_anchors":model.quality_anchors.tolist(),"detector_bounds":detector_bounds(model.detector),"optics":config.get("optics",{}),"optimizer":{"type":str(optimizer.get("type","adam")).lower(),"lr":float(optimizer.get("lr",optimizer.get("learning_rate",.01))),"weight_decay":float(optimizer.get("weight_decay",0.0))},"parameters":{"optical_phase":model.optical_parameter_count(),"electronic_trainable":0,"total_trainable":sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}}


def raw_score_results(result,bundle):
    metadata=bundle.metadata;targets=denormalize_quality(result["targets"],metadata["train_score_min"],metadata["train_score_max"],metadata["score_higher_is_better"]);predictions=denormalize_quality(result["predictions"],metadata["train_score_min"],metadata["train_score_max"],metadata["score_higher_is_better"])
    return targets,predictions,regression_metrics(targets,predictions)


def group_metrics(result,dataset,metadata_key):
    groups=defaultdict(lambda:{"targets":[],"predictions":[]})
    for index,target,prediction in zip(result["indices"],result["targets"].tolist(),result["predictions"].tolist()):
        metadata=dataset.sample_metadata(index);key=str(metadata.get(metadata_key) if metadata.get(metadata_key) is not None else "unknown");groups[key]["targets"].append(target);groups[key]["predictions"].append(prediction)
    return {key:regression_metrics(value["targets"],value["predictions"]) for key,value in sorted(groups.items()) if len(value["targets"])>=2}


def write_predictions(path,result,dataset,bundle):
    raw_targets,raw_predictions,_=raw_score_results(result,bundle);rows=[]
    for position,(index,target,prediction) in enumerate(zip(result["indices"],result["targets"].tolist(),result["predictions"].tolist())):
        metadata=dataset.sample_metadata(index);rows.append({"sample_index":index,**metadata,"target_normalized":target,"prediction_normalized":prediction,"absolute_error_normalized":abs(target-prediction),"target_score":float(raw_targets[position]),"prediction_score":float(raw_predictions[position]),"absolute_error_score":abs(float(raw_targets[position]-raw_predictions[position]))})
    write_rows(path,rows)


def train_model(model,bundle,config,device,run_dir,visualization):
    cfg=config.get("dataset",{});train_loader=make_loader(bundle.train,cfg.get("batch_size",8),cfg.get("num_workers",8),True,43);validation_loader=make_loader(bundle.validation,cfg.get("batch_size",8),cfg.get("num_workers",8),False,44)
    optimizer=build_optimizer(model,config)
    fixed=fixed_batch(validation_loader,device,config.get("visualization",{}).get("num_samples",6));save_epoch_artifacts(model,fixed,run_dir,"epoch_0000",visualization)
    epochs=int(config.get("training",{}).get("epochs",100));print_freq=int(config.get("training",{}).get("print_freq",50));interval=int(config.get("visualization",{}).get("save_interval_epochs",10));rows=[];best=-float("inf")
    dropout=config.get("regularization",{}).get("phase_dropout",{})
    for epoch in range(1,epochs+1):
        started=time.perf_counter();active=bool(dropout.get("enabled",False)) and epoch>=int(dropout.get("start_epoch",0));model.set_phase_dropout_active(active)
        train=run_epoch(model,train_loader,config,device,optimizer,print_freq);validation=run_epoch(model,validation_loader,config,device)
        row={"epoch":epoch,"learning_rate":optimizer.param_groups[0]["lr"],"train_loss":train["metrics"]["loss"],"train_mae":train["metrics"]["mae"],"train_rmse":train["metrics"]["rmse"],"train_plcc":train["metrics"]["plcc"],"train_srocc":train["metrics"]["srocc"],"validation_loss":validation["metrics"]["loss"],"validation_mae":validation["metrics"]["mae"],"validation_rmse":validation["metrics"]["rmse"],"validation_plcc":validation["metrics"]["plcc"],"validation_srocc":validation["metrics"]["srocc"],"phase_dropout_active":active,"epoch_time_sec":time.perf_counter()-started}
        rows.append(row);write_rows(run_dir/"metrics"/"training_history.csv",rows);save_json(row,run_dir/"metrics"/"training_latest.json");save_checkpoint(run_dir/"checkpoints"/"last.pt",model,optimizer,epoch,row,config)
        score=row["validation_srocc"]
        if epoch==1 or score>best:
            best=score;save_checkpoint(run_dir/"checkpoints"/"best.pt",model,optimizer,epoch,row,config);save_json(row,run_dir/"metrics"/"best_validation.json");save_epoch_artifacts(model,fixed,run_dir,"best_epoch",visualization)
        if interval>0 and epoch%interval==0:save_epoch_artifacts(model,fixed,run_dir,f"epoch_{epoch:04d}",visualization)
        print(f"epoch {epoch:03d} train_rmse={row['train_rmse']:.4f} val_rmse={row['validation_rmse']:.4f} val_plcc={row['validation_plcc']:.4f} val_srocc={row['validation_srocc']:.4f}")
    save_training_curves(rows,run_dir/"figures"/"training_curves.png")


def test_model(model,bundle,config,device,run_dir,visualization):
    checkpoint=run_dir/"checkpoints"/"best.pt"
    if not checkpoint.is_file():raise FileNotFoundError(f"Best checkpoint not found: {checkpoint}. Run --phase train first.")
    payload=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(payload["model_state_dict"])
    cfg=config.get("dataset",{});loader=make_loader(bundle.test,cfg.get("batch_size",8),cfg.get("num_workers",8),False,45);result=run_epoch(model,loader,config,device)
    raw_targets,raw_predictions,raw_metrics=raw_score_results(result,bundle);summary={"normalized_quality_metrics":result["metrics"],"original_score_metrics":raw_metrics,"score_column":bundle.metadata["score_column"],"score_higher_is_better":bundle.metadata["score_higher_is_better"],"feature":"distorted grayscale image only; reference identity is split metadata only"}
    save_json(summary,run_dir/"metrics"/"test_regression.json");save_json(group_metrics(result,bundle.test,"distortion_type"),run_dir/"metrics"/"per_distortion_type.json");save_json(group_metrics(result,bundle.test,"distortion_level"),run_dir/"metrics"/"per_distortion_level.json");write_predictions(run_dir/"metrics"/"test_predictions.csv",result,bundle.test,bundle);save_scatter(raw_targets,raw_predictions,run_dir/"figures"/"predicted_vs_target.png","KADID-10k predicted vs target quality score")
    save_epoch_artifacts(model,fixed_batch(loader,device,config.get("visualization",{}).get("num_samples",6)),run_dir,"test_examples",visualization);return summary


def main():
    args=parse_args();config_path=Path(args.config);config_path=config_path if config_path.is_absolute() else BASE_DIR/config_path;config=load_yaml(config_path)
    if args.epochs is not None:config.setdefault("training",{})["epochs"]=args.epochs
    if args.output_dir:run_dir=Path(args.output_dir).resolve()
    else:run_dir=BASE_DIR/"runs"/config.get("experiment",{}).get("run_name","kadid10k_iqa_regression")
    if args.disable_visualization:config.setdefault("visualization",{})["enabled"]=False
    run_dir.mkdir(parents=True,exist_ok=True);save_yaml(config,run_dir/"config.yaml");save_json(config,run_dir/"config_resolved.json");save_json(environment_info(),run_dir/"environment.json");save_json(git_info(),run_dir/"git_info.json");shutil.copy2(config_path,run_dir/"source_config.yaml");(run_dir/"command.txt").write_text(" ".join(sys.argv),encoding="utf-8")
    seed=int(config.get("seed",42));set_seed(seed);bundle=load_data(config,args.smoke_test);save_json(bundle.metadata,run_dir/"dataset.json")
    if args.phase=="prepare_data":print(f"KADID-10k ready: train={len(bundle.train)} validation={len(bundle.validation)} test={len(bundle.test)} output={run_dir}");return 0
    device=choose_device(args.device or config.get("device","auto"));model=FullOpticalIQARegressor(config).to(device);save_json(model_report(model,config),run_dir/"model.json");visualization=bool(config.get("visualization",{}).get("enabled",True))
    print(f"device={device} train={len(bundle.train)} validation={len(bundle.validation)} test={len(bundle.test)} optical_params={model.optical_parameter_count()} electronic_params=0")
    if args.phase in {"train","all"}:train_model(model,bundle,config,device,run_dir,visualization)
    if args.phase in {"test","all"}:summary=test_model(model,bundle,config,device,run_dir,visualization);print(f"test original-score metrics: {summary['original_score_metrics']}")
    print(f"saved to {run_dir}");return 0


if __name__=="__main__":raise SystemExit(main())
