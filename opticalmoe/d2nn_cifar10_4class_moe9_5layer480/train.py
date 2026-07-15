import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data import create_loaders
from model import OpticalMoEClassifier
from slm_export import export_best_slm_package
from utils import BASE_DIR,choose_device,environment_info,git_info,load_yaml,save_json,save_yaml,set_seed,write_rows
from visualization import confusion_matrix,save_confusion,save_epoch_artifacts,save_training_curves


def detector_plane_mse_loss(intensity,target_plane,scale,normalize,eps):
    """Full-plane MSE with optional per-sample total-energy matching."""
    eps=float(eps)
    if eps<=0:raise ValueError("loss.detector_plane_mse_normalization_eps must be positive")
    prediction=intensity
    if bool(normalize):
        prediction_energy=prediction.sum(dim=(-2,-1),keepdim=True)
        target_energy=target_plane.sum(dim=(-2,-1),keepdim=True)
        prediction=prediction*target_energy/(prediction_energy+eps)
    return float(scale)*F.mse_loss(prediction,target_plane)


def build_optimizer(model,config):
    cfg=config.get("optimizer",{});opt_type=str(cfg.get("type","adam")).strip().lower();lr=float(cfg.get("lr",cfg.get("learning_rate",0.01)));weight_decay=float(cfg.get("weight_decay",0.0))
    if opt_type=="adam":return torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    if opt_type=="adamw":return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer.type: {opt_type}. Expected 'adam' or 'adamw'.")


def args_parser():
    parser=argparse.ArgumentParser(description="Train CIFAR-10 pure-optical 9-expert 5-layer D2NN MoE (4 or 10 classes).")
    parser.add_argument("--config",default="configs/config.yaml");parser.add_argument("--device",default=None);parser.add_argument("--epochs",type=int,default=None);parser.add_argument("--batch-size",type=int,default=None);parser.add_argument("--train-samples-per-class-per-epoch",type=int,default=None);parser.add_argument("--run-name","--run_name",dest="run_name",default=None);parser.add_argument("--smoke-test","--smoke_test",dest="smoke_test",action="store_true");parser.add_argument("--disable-visualization","--disable_visualization",dest="disable_visualization",action="store_true")
    return parser.parse_args()


def forward_loss(model,images,targets,loss_cfg):
    loss_type=loss_cfg.get("type","detector_plane_mse");scale=float(loss_cfg.get("scale",100.0))
    # Layer fields are only needed by visualization. Avoid retaining five
    # complex 480x480 tensors and their autograd history in every train batch.
    logits,items=model(images,return_intermediates=True,capture_layer_fields=False)
    if loss_type=="detector_plane_mse":
        target=model.detector.masks[targets].to(images.device)
        classification=detector_plane_mse_loss(items["detector_intensity"],target,scale,loss_cfg.get("normalize_detector_plane_mse",False),loss_cfg.get("detector_plane_mse_normalization_eps",1.0e-8))
    elif loss_type=="cross_entropy":classification=scale*F.cross_entropy(logits,targets)
    else:raise ValueError(f"Unsupported loss.type: {loss_type}")
    balance=items["router_balance_loss"];balance_weight=float(loss_cfg.get("router_balance_weight",0.01))
    importance=items["router_importance_loss"];importance_weight=float(loss_cfg.get("router_importance_weight",0.0))
    total=classification+balance_weight*balance+importance_weight*importance
    return logits,total,{"classification":classification,"router_balance":balance,"router_importance":importance,"router_entropy":items["router_normalized_entropy"],"routing_weights":items["routing_weights"],"selected_mask":items["routing_selected_mask"]}


def run_epoch(model,loader,loss_cfg,device,optimizer=None,print_freq=50,collect_phase_diagnostics=False):
    training=optimizer is not None;model.train(training);total_loss=0.0;total_classification=0.0;total_balance=0.0;total_importance=0.0;total_entropy=0.0;correct=0;count=0;predictions=[];targets_all=[];routing_weight_sum=torch.zeros(9);selection_count=torch.zeros(9)
    # With diagnostics disabled (the default), do not enumerate, clone, join,
    # or inspect any phase tensor/gradient in the training loop.
    phase_parameters=[parameter for name,parameter in model.named_parameters() if name.endswith("raw_phase")] if training and collect_phase_diagnostics else []
    phase_before=[parameter.detach().clone() for parameter in phase_parameters]
    grad_mean=grad_max=grad_nonzero_ratio=0.0
    context=torch.enable_grad() if training else torch.no_grad()
    with context:
        for step,(images,targets) in enumerate(loader,1):
            images=images.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
            if training:optimizer.zero_grad(set_to_none=True)
            logits,loss,parts=forward_loss(model,images,targets,loss_cfg)
            if training:
                loss.backward()
                if collect_phase_diagnostics and step==1:
                    gradients=[parameter.grad.detach().reshape(-1) for parameter in phase_parameters if parameter.grad is not None]
                    if gradients:
                        gradient=torch.cat(gradients);grad_mean=float(gradient.abs().mean());grad_max=float(gradient.abs().max());grad_nonzero_ratio=float((gradient!=0).float().mean())
                optimizer.step()
            batch=len(targets);total_loss+=float(loss.item())*batch;total_classification+=float(parts["classification"].item())*batch;total_balance+=float(parts["router_balance"].item())*batch;total_importance+=float(parts["router_importance"].item())*batch;total_entropy+=float(parts["router_entropy"].item())*batch;correct+=int((logits.argmax(1)==targets).sum());count+=batch
            routing_weight_sum+=parts["routing_weights"].detach().cpu().sum(0);selection_count+=parts["selected_mask"].detach().cpu().float().sum(0)
            predictions.append(logits.argmax(1).detach().cpu());targets_all.append(targets.detach().cpu())
            if training and print_freq>0 and step%print_freq==0:print(f"  batch {step}/{len(loader)} loss={total_loss/count:.5f} acc={correct/count:.4f}")
    result={"loss":total_loss/max(1,count),"classification_loss":total_classification/max(1,count),"router_balance_loss":total_balance/max(1,count),"router_importance_loss":total_importance/max(1,count),"router_normalized_entropy":total_entropy/max(1,count),"acc":correct/max(1,count),"preds":torch.cat(predictions),"targets":torch.cat(targets_all),"mean_routing_weights":(routing_weight_sum/max(1,count)).tolist(),"expert_selection_rates":(selection_count/max(1,count)).tolist()}
    if training and collect_phase_diagnostics and phase_parameters:
        raw=torch.cat([parameter.detach().reshape(-1) for parameter in phase_parameters]);effective=2.0*torch.pi*torch.sigmoid(raw);delta=torch.cat([(parameter.detach()-before).reshape(-1) for parameter,before in zip(phase_parameters,phase_before)])
        result.update({"phase_first_grad_abs_mean":grad_mean,"phase_first_grad_abs_max":grad_max,"phase_first_grad_nonzero_ratio":grad_nonzero_ratio,"raw_phase_std":float(raw.std()),"effective_phase_std_rad":float(effective.std()),"phase_epoch_delta_abs_mean":float(delta.abs().mean()),"phase_epoch_delta_abs_max":float(delta.abs().max())})
    return result


def fixed_batch(loader,device,count=4):
    images=[];targets=[];n=0
    for x,y in loader:
        take=min(len(x),count-n);images.append(x[:take]);targets.append(y[:take]);n+=take
        if n>=count:break
    return torch.cat(images).to(device),torch.cat(targets).to(device)


def save_checkpoint(path,model,optimizer,epoch,metrics,config):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"metrics":metrics,"config":config},path)


def report(model,config,class_names,train_loader,test_loader):
    optics=config.get("optics",{});distances=optics.get("distances_m",{})
    conversion_parameters=model.interlayer_conversion_parameter_count()
    return {
        "model":"OpticalMoEClassifier","task":f"CIFAR-10 {len(class_names)}-class pure optical MoE","class_names":class_names,
        "dataset":{"train_samples":len(train_loader.dataset),"test_samples":len(test_loader.dataset),"train_samples_per_class":config.get("dataset",{}).get("train_samples_per_class"),"test_samples_per_class":config.get("dataset",{}).get("test_samples_per_class"),"train_samples_per_class_per_epoch":config.get("dataset",{}).get("train_samples_per_class_per_epoch"),"train_samples_per_epoch":len(train_loader.sampler),"batch_size":train_loader.batch_size,"shuffle_train":True,"full_dataset_retained":config.get("dataset",{}).get("train_samples_per_class") is None},
        "layout":model.layout.to_dict(),"num_layers_per_expert":model.num_layers,"distances_m":distances,
        "optoelectronic_interlayers":{**config.get("optoelectronic_interlayers",{}),"trainable_parameters":conversion_parameters,"conversion_count":5 if model.optoelectronic_enabled else 0,"sequence":"phase -> propagation -> square detection -> per-expert affine LayerNorm -> ReLU -> zero-phase amplitude reload" if model.optoelectronic_enabled else "disabled; continuous coherent propagation","routing_amplitude_reapplied":False},
        "wavelength_m":float(optics.get("wavelength_m",5.32e-7)),"pixel_size_m":float(optics.get("pixel_size_m",16e-6)),
        "k_space":{"enabled":model.to_detector.k_space_constraint_enabled,"theta_max_deg":model.to_detector.theta_max_deg,"max_sampled_angle_deg":model.to_detector.max_sampled_angle_deg,"pass_fraction":model.to_detector.k_space_pass_fraction},
        "detector_bounds":[[int(v) for v in [p.nonzero().min(0).values[0],p.nonzero().max(0).values[0]+1,p.nonzero().min(0).values[1],p.nonzero().max(0).values[1]+1]] for p in model.detector.masks.cpu()],
        "routing":{"type":"input_topk","top_k":int(config.get("prompt",{}).get("top_k",3)),"balance_loss_weight":float(config.get("loss",{}).get("router_balance_weight",0.01)),"importance_loss_weight":float(config.get("loss",{}).get("router_importance_weight",0.0)),"importance_loss":"num_experts * sum(mean_probability^2) - 1; 0 is uniform, num_experts-1 is one-expert collapse","inactive_grating_amplitude":0.0,"max_abs_grating_frequency":model.prompt.max_abs_grating_frequency,"nyquist_frequency":model.prompt.nyquist_frequency,"edge_grating_period_pixels":model.prompt.edge_grating_period_pixels,"minimum_grating_period_pixels":model.prompt.min_grating_period_pixels},
        "optimizer":{"type":str(config.get("optimizer",{}).get("type","adam")).lower(),"lr":float(config.get("optimizer",{}).get("lr",config.get("optimizer",{}).get("learning_rate",0.01))),"weight_decay":float(config.get("optimizer",{}).get("weight_decay",0.0))},
        "parameters":{"expert_phase":model.expert_phase_parameter_count(),"global_fc_phase":model.global_fc_parameter_count(),"optical_total":model.optical_parameter_count(),"electronic_router":model.router_parameter_count(),"interlayer_affine":conversion_parameters,"electronic_total":model.electronic_parameter_count(),"electronic_classifier":0,"trainable_total":sum(p.numel() for p in model.parameters() if p.requires_grad)},
        "slm_export":config.get("slm_export",{}),
    }


def main():
    args=args_parser();config_path=Path(args.config)
    if not config_path.is_absolute():config_path=config_path.resolve() if config_path.is_file() else BASE_DIR/config_path
    config=load_yaml(config_path)
    if args.epochs is not None:config.setdefault("training",{})["epochs"]=args.epochs
    if args.batch_size is not None:config.setdefault("dataset",{})["batch_size"]=args.batch_size
    if args.train_samples_per_class_per_epoch is not None:config.setdefault("dataset",{})["train_samples_per_class_per_epoch"]=args.train_samples_per_class_per_epoch
    if args.run_name:config.setdefault("experiment",{})["run_name"]=args.run_name
    if args.smoke_test:config["dataset"]["batch_size"]=2;config["dataset"]["num_workers"]=0;config["training"]["epochs"]=1
    if args.disable_visualization:config.setdefault("visualization",{})["enabled"]=False
    seed=int(config.get("seed",7));set_seed(seed);device=choose_device(args.device or config.get("device","auto"))
    run_dir=BASE_DIR/"runs"/config.get("experiment",{}).get("run_name","cifar4_moe9");run_dir.mkdir(parents=True,exist_ok=True)
    save_yaml(config,run_dir/"config.yaml");save_json(config,run_dir/"config_resolved.json");save_json(environment_info(),run_dir/"environment.json");save_json(git_info(),run_dir/"git_info.json");shutil.copy2(config_path,run_dir/"source_config.yaml");(run_dir/"command.txt").write_text(" ".join(sys.argv),encoding="utf-8")
    train_loader,test_loader,class_names=create_loaders(config,seed,args.smoke_test)
    save_json({
        "dataset":config.get("dataset",{}).get("name",f"cifar10_{len(class_names)}class"),"class_names":class_names,"source_class_indices":config.get("dataset",{}).get("class_indices",list(range(len(class_names)))),
        "train_samples":len(train_loader.dataset),"test_samples":len(test_loader.dataset),
        "per_class_train_counts":{class_names[index]:count for index,count in train_loader.dataset.class_counts.items()},
        "per_class_test_counts":{class_names[index]:count for index,count in test_loader.dataset.class_counts.items()},
        "image_size":int(config.get("dataset",{}).get("image_size",100)),"input_size_after_zero_padding":int(config.get("dataset",{}).get("input_size",120)),"grayscale":True,
        "batch_size":train_loader.batch_size,"shuffle_train":True,
        "train_samples_per_class_per_epoch":config.get("dataset",{}).get("train_samples_per_class_per_epoch"),"train_samples_per_epoch":len(train_loader.sampler),
        "sampling_note":"train/test_samples_per_class are dataset-wide caps. train_samples_per_class_per_epoch rotates through a smaller balanced subset without deleting the remaining data. batch_size controls optimizer mini-batches.",
    },run_dir/"dataset.json")
    model=OpticalMoEClassifier(config,len(class_names)).to(device)
    optimizer=build_optimizer(model,config)
    loss_cfg=config.get("loss",{"type":"detector_plane_mse","scale":100.0});save_json(report(model,config,class_names,train_loader,test_loader),run_dir/"architecture_report.json")
    print(f"device={device} train={len(train_loader.dataset)} test={len(test_loader.dataset)} classes={class_names}")
    print(f"experts=9 layers_per_expert=5 expert_phase_params={model.expert_phase_parameter_count()} global_fc_params={model.global_fc_parameter_count()} router_params={model.router_parameter_count()} interlayer_affine_params={model.interlayer_conversion_parameter_count()} electronic_classifier_params=0")
    print(f"distances={config['optics']['distances_m']} detector_bounds={report(model,config,class_names,train_loader,test_loader)['detector_bounds']}")
    fixed=fixed_batch(test_loader,device,int(config.get("visualization",{}).get("num_samples",4)));viz=bool(config.get("visualization",{}).get("enabled",True));interval=int(config.get("visualization",{}).get("save_interval_epochs",10));save_epoch_artifacts(model,fixed,run_dir,"epoch_0000",class_names,viz)
    epochs=int(config.get("training",{}).get("epochs",200));print_freq=int(config.get("training",{}).get("print_freq",50));phase_diagnostics=bool(config.get("training",{}).get("phase_diagnostics_enabled",False));rows=[];best_acc=-1.0;best_epoch=0;start=time.perf_counter()
    dropout_cfg=config.get("regularization",{}).get("phase_dropout",{})
    for epoch in range(1,epochs+1):
        active=bool(dropout_cfg.get("enabled",False)) and epoch>=int(dropout_cfg.get("start_epoch",0));model.set_phase_dropout_active(active)
        epoch_start=time.perf_counter();train_metrics=run_epoch(model,train_loader,loss_cfg,device,optimizer,print_freq,phase_diagnostics);test_metrics=run_epoch(model,test_loader,loss_cfg,device)
        row={"epoch":epoch,"train_loss":train_metrics["loss"],"train_classification_loss":train_metrics["classification_loss"],"train_router_balance_loss":train_metrics["router_balance_loss"],"train_router_importance_loss":train_metrics["router_importance_loss"],"train_router_normalized_entropy":train_metrics["router_normalized_entropy"],"train_acc":train_metrics["acc"],"test_loss":test_metrics["loss"],"test_classification_loss":test_metrics["classification_loss"],"test_router_balance_loss":test_metrics["router_balance_loss"],"test_router_importance_loss":test_metrics["router_importance_loss"],"test_router_normalized_entropy":test_metrics["router_normalized_entropy"],"test_acc":test_metrics["acc"],"lr":optimizer.param_groups[0]["lr"],"phase_dropout_active":active,"epoch_time_sec":time.perf_counter()-epoch_start}
        if phase_diagnostics:
            for key in ("phase_first_grad_abs_mean","phase_first_grad_abs_max","phase_first_grad_nonzero_ratio","raw_phase_std","effective_phase_std_rad","phase_epoch_delta_abs_mean","phase_epoch_delta_abs_max"):
                row[key]=train_metrics[key]
        for expert_index in range(9):
            row[f"train_expert_{expert_index}_selection_rate"]=train_metrics["expert_selection_rates"][expert_index]
            row[f"train_expert_{expert_index}_mean_weight"]=train_metrics["mean_routing_weights"][expert_index]
        rows.append(row);write_rows(run_dir/"metrics"/"epoch_metrics.csv",rows);save_checkpoint(run_dir/"checkpoints"/"last.pt",model,optimizer,epoch,row,config)
        if row["test_acc"]>best_acc:
            best_acc=row["test_acc"];best_epoch=epoch;save_checkpoint(run_dir/"checkpoints"/"best.pt",model,optimizer,epoch,row,config);save_epoch_artifacts(model,fixed,run_dir,"best_epoch",class_names,viz)
        if interval>0 and epoch%interval==0:save_epoch_artifacts(model,fixed,run_dir,f"epoch_{epoch:04d}",class_names,viz)
        message=f"epoch {epoch:03d} train_loss={row['train_loss']:.5f} balance={row['train_router_balance_loss']:.5f} importance={row['train_router_importance_loss']:.5f} entropy={row['train_router_normalized_entropy']:.4f} train_acc={row['train_acc']:.4f} test_acc={row['test_acc']:.4f}"
        if phase_diagnostics:
            message+=f" phase_std={row['effective_phase_std_rad']:.6f}rad phase_delta_mean={row['phase_epoch_delta_abs_mean']:.3e} grad_mean={row['phase_first_grad_abs_mean']:.3e}"
        print(message+f" selection_rates={[round(v,3) for v in train_metrics['expert_selection_rates']]}")
    final=run_epoch(model,test_loader,loss_cfg,device);matrix=confusion_matrix(final["preds"],final["targets"],len(class_names));save_confusion(matrix,run_dir/"figures"/"confusion_matrix.png",class_names);save_training_curves(rows,run_dir/"figures"/"training_curves.png");save_epoch_artifacts(model,fixed,run_dir,"final_epoch",class_names,viz)
    metrics={"best_epoch":best_epoch,"best_test_acc":best_acc,"final_test_acc":final["acc"],"final_test_loss":final["loss"],"final_router_balance_loss":final["router_balance_loss"],"final_router_importance_loss":final["router_importance_loss"],"final_router_normalized_entropy":final["router_normalized_entropy"],"final_expert_selection_rates":final["expert_selection_rates"],"final_mean_routing_weights":final["mean_routing_weights"],"wall_time_sec":time.perf_counter()-start};save_json(metrics,run_dir/"metrics"/"final_metrics.json")
    write_rows(run_dir/"metrics"/"test_predictions.csv",[
        {"sample_index":index,"true_label":int(target),"true_name":class_names[int(target)],"pred_label":int(pred),"pred_name":class_names[int(pred)],"correct":bool(pred==target)}
        for index,(target,pred) in enumerate(zip(final["targets"].tolist(),final["preds"].tolist()))
    ])
    export_best_slm_package(model,test_loader,run_dir/"checkpoints"/"best.pt",run_dir/"slm_bmp_best",config,device,class_names)
    print(f"saved to {run_dir}")


if __name__=="__main__":raise SystemExit(main())
