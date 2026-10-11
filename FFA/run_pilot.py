"""Bounded, validation-only LSP pilot; never resumes or rewrites a P12 run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback
from dataclasses import replace

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa import training as p12
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.datasets import build_loaders
from experiments.qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa.settings import load_settings, IMPLEMENTATION_FILES
from FFA.optics import install_full_response, uninstall_full_response

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"head_only", "bp_current", "fa_source_connector", "source_full_response", "random_full_response"}


def write_json(path, value):
    p12.write_json(Path(path), value)


def source_manifest():
    paths = sorted((ROOT / "FFA").rglob("*.py"))
    paths += [ROOT / item for item in IMPLEMENTATION_FILES]
    paths += [ROOT / "FFA" / "pilot.json"]
    return {str(p.relative_to(ROOT)): p12.sha256_file(p) for p in paths
            if not any(x in p.parts for x in ("private", "runs", "cache", "__pycache__"))}


def configure(model, method, seed):
    uninstall_full_response(model.backbone)
    model.configure_feedback("bp_current")
    if method == "fa_source_connector":
        model.configure_feedback("fa_pretrained")
    elif method in {"bp_current", "head_only"}:
        install_full_response(model.backbone, "current_response")
    elif method in {"source_full_response", "random_full_response"}:
        phase = model.source_phases
        if method == "random_full_response":
            gen = torch.Generator(device="cpu").manual_seed(seed + 8_000_003)
            phase = 2 * math.pi * torch.rand(model.source_phases.shape, generator=gen)
        install_full_response(model.backbone, method, source_phases=phase)
    else:
        raise ValueError(method)


def policy(model, phase_enabled):
    model.requires_grad_(False)
    for p in model.phase_parameters():
        p.requires_grad_(phase_enabled)
    model.head.requires_grad_(True)
    model.backbone.eval()


def electronic_hash(model):
    digest = hashlib.sha256()
    for name, value in model.backbone.state_dict().items():
        if not name.endswith(".raw_phase"):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def feedback_identity(model, method, seed):
    digest=hashlib.sha256()
    if method in {"source_full_response", "random_full_response"}:
        for stage in model.backbone.stages:
            digest.update(stage.ffa_response_phase.detach().cpu().contiguous().numpy().tobytes())
    elif method=="fa_source_connector":
        for stage in model.backbone.stages:
            digest.update(stage.feedback_phase.detach().cpu().contiguous().numpy().tobytes())
    else:
        return {"fixed":False}
    return {"fixed":True,"physical_phase_tensor_sha256":digest.hexdigest(),
            "random_seed":seed+8_000_003 if method=="random_full_response" else None}


def gradients(model, batch, settings, method, seed, legacy=False):
    configure(model, method, seed)
    if legacy:
        uninstall_full_response(model.backbone)
        model.configure_feedback("bp_current")
    model.eval()
    model.zero_grad(set_to_none=True)
    output = model(batch["image"])
    loss, _ = p12.task_loss(settings, output, batch)
    loss.backward()
    phase = [x.grad.detach().clone() for x in model.phase_parameters()]
    head = [x.grad.detach().clone() for x in model.head.parameters()]
    return phase, head, output["logits"].detach().clone(), float(loss)


def compare(a, b):
    ref_sq = cand_sq = dot = diff_sq = 0.0
    max_abs = 0.0
    for x, y in zip(a, b):
        x = x.double(); y = y.double()
        ref_sq += float(x.square().sum()); cand_sq += float(y.square().sum())
        dot += float((x*y).sum()); diff_sq += float((x-y).square().sum())
        max_abs = max(max_abs, float((x-y).abs().max()))
    return {"cosine": dot / max(math.sqrt(ref_sq*cand_sq), 1e-30),
            "relative_l2": math.sqrt(diff_sq/max(ref_sq, 1e-30)),
            "max_absolute_difference": max_abs,
            "reference_norm": math.sqrt(ref_sq), "candidate_norm": math.sqrt(cand_sq)}


def diagnose(model, batch, settings, seed, candidate, include_legacy=False):
    ref, refhead, refout, loss = gradients(model,batch,settings,"bp_current",seed)
    cand, candhead, candout, _ = gradients(model,batch,settings,candidate,seed)
    result = {"loss": loss, "phase": compare(ref,cand), "head": compare(refhead,candhead),
              "per_stage": [compare([x],[y]) for x,y in zip(ref,cand)],
              "forward_max_abs": float((refout-candout).abs().max())}
    if include_legacy:
        old, oldhead, oldout, _ = gradients(model,batch,settings,"bp_current",seed,legacy=True)
        result["legacy_2d_bp_phase"] = compare(old,ref)
        result["legacy_2d_bp_head"] = compare(oldhead,refhead)
        result["legacy_2d_bp_forward_max_abs"] = float((oldout-refout).abs().max())
        if (result["legacy_2d_bp_phase"]["cosine"] < 0.9999 or result["legacy_2d_bp_phase"]["relative_l2"] > 0.01
                or result["legacy_2d_bp_head"]["relative_l2"] > 0.01 or result["legacy_2d_bp_forward_max_abs"] > 5e-4):
            raise RuntimeError(f"Analytic 1D baseline differs from legacy BP: {result}")
    configure(model,candidate,seed)
    model.zero_grad(set_to_none=True)
    return result


def warm_start(model, settings, data_config, bundle, profile, device, out):
    """Cache deterministic float32 source features, with paired targets."""
    cache_config = replace(data_config, augmentation_enabled=False)
    cache_loaders = build_loaders(bundle, cache_config)
    policy(model,False)
    configure(model,"head_only",profile["seed"])
    model.eval()
    cached = []
    with torch.no_grad():
        for raw in cache_loaders["train"]:
            batch = p12._move_batch(raw,device)
            with torch.autocast("cuda",dtype=torch.float16,enabled=profile["use_amp"]):
                feature,_ = model.forward_features(batch["image"])
            cached.append({"feature":feature.detach().float().cpu(),
                           **{k:v.detach().cpu() for k,v in batch.items() if torch.is_tensor(v) and k != "image"}})
    fields = {k:torch.cat([b[k] for b in cached]) for k in cached[0]}
    write_json(out/"warmup_cache.json", {"samples":len(fields["feature"]),
        "feature_bytes":fields["feature"].numel()*4, "feature_dtype":"float32",
        "cache_compute_amp":profile["use_amp"],
        "augmentation":False, "used_only_for_head_warmup":True})
    keys = list(fields)
    gen = torch.Generator().manual_seed(profile["seed"]+11)
    loader = DataLoader(TensorDataset(*(fields[k] for k in keys)),
                        batch_size=profile["train_batch_size"],shuffle=True,generator=gen)
    optimizer = torch.optim.AdamW(model.head.parameters(),lr=profile["head_learning_rate"],weight_decay=5e-4)
    best=-math.inf; selected=None; history=[]
    for epoch in range(1,profile["warmup_epochs"]+1):
        model.head.train(); total=0.; count=0; started=time.time()
        for raw in loader:
            batch = {k:v.to(device) for k,v in zip(keys,raw)}
            optimizer.zero_grad(set_to_none=True)
            loss,_=p12.task_loss(settings,model.head(batch["feature"]),batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(),5.,error_if_nonfinite=True)
            optimizer.step(); n=len(batch["feature"]); total+=float(loss.detach())*n;count+=n
        # Source features are fixed; validation still uses the actual image pipeline.
        with torch.no_grad():
            validation=p12.evaluate(model,cache_loaders["val"],settings,device)
        metric=validation["pck_torso_0p2"]
        if metric>best:
            best=metric; selected=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()});best_epoch=epoch
        history.append({"epoch":epoch,"loss":total/count,"validation":validation,"seconds":time.time()-started})
        write_json(out/"warmup_history.json",history)
        print(f"warmup {epoch}/{profile['warmup_epochs']} val={metric:.6f} loss={total/count:.6f}",flush=True)
    model.load_state_dict(selected,strict=True)
    p12.save_torch(out/"shared_start.pt", {"format":"ffa-new-shared-start-v1","model":selected,
        "seed":profile["seed"],"best_epoch":best_epoch,"completed_warmup_epochs":profile["warmup_epochs"],
        "source_checkpoint_sha256":model.source_manifest["sha256"],
        "dataset_manifest_sha256":bundle.manifest_sha256,"profile":profile})
    del fields,cached,loader,cache_loaders
    return selected, best


def adapt(model, settings, data_config, bundle, profile, method, shared, device, out):
    p12.seed_everything(profile["seed"]+29)
    loaders=build_loaders(bundle,data_config)
    del loaders["test"]
    model.load_state_dict(shared,strict=True)
    policy(model,method != "head_only")
    configure(model,method,profile["seed"])
    feedback=feedback_identity(model,method,profile["seed"])
    frozen_before=electronic_hash(model)
    batch=p12._move_batch(next(iter(loaders["val"])),device)
    batch={k:v[:4] if torch.is_tensor(v) and v.ndim else v for k,v in batch.items()}
    if method != "head_only":
        initial=diagnose(model,batch,settings,profile["seed"],method,include_legacy=method=="bp_current")
        if method in {"bp_current","fa_source_connector","source_full_response"}:
            if initial["phase"]["cosine"]<0.9999 or initial["phase"]["relative_l2"]>0.01:
                raise RuntimeError(f"Matched-source initial gradient failed: {initial}")
        write_json(out/"initial_diagnostic.json",initial)
    phase=list(model.phase_parameters());head=list(model.head.parameters())
    groups=[{"params":head,"lr":profile["head_learning_rate"],"weight_decay":5e-4}]
    if method != "head_only":
        groups.append({"params":phase,"lr":profile["phase_learning_rate"],"weight_decay":0.})
    optimizer=torch.optim.AdamW(groups)
    total_steps=len(loaders["train"])*profile["adaptation_epochs"]
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:0.05+0.95*0.5*(1+math.cos(math.pi*min(s,total_steps)/total_steps)))
    scaler=torch.amp.GradScaler("cuda",enabled=profile["use_amp"],init_scale=256,growth_interval=100000)
    with torch.no_grad():
        baseline=p12.evaluate(model,loaders["val"],settings,device)
    history=[];best=baseline["pck_torso_0p2"];best_epoch=0
    best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
    write_json(out/"baseline_validation.json",baseline)
    for epoch in range(1,profile["adaptation_epochs"]+1):
        model.backbone.eval();model.head.train();started=time.time();loss_sum=0.;count=0
        phase_nonzero=[False]*len(phase); skipped_steps=0
        for raw in loaders["train"]:
            b=p12._move_batch(raw,device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=profile["use_amp"]):
                output=model(b["image"]);loss,_=p12.task_loss(settings,output,b)
            if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite loss")
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if method != "head_only":
                for j,p in enumerate(phase):
                    if p.grad is None:raise RuntimeError(f"Missing phase gradient {j}")
                    phase_nonzero[j] |= bool(p.grad.abs().sum()>0)
                torch.nn.utils.clip_grad_norm_(phase,2.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(head,5.,error_if_nonfinite=True)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<old_scale: skipped_steps+=1
            else:scheduler.step()
            n=len(b["image"]);loss_sum+=float(loss.detach())*n;count+=n
        with torch.no_grad():validation=p12.evaluate(model,loaders["val"],settings,device)
        metric=validation["pck_torso_0p2"]
        if metric>best:
            best=metric;best_epoch=epoch;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        row={"epoch":epoch,"training_loss":loss_sum/count,"validation":validation,
            "phase":model.phase_report(),"phase_nonzero":phase_nonzero,
            "seconds":time.time()-started,"skipped_optimizer_steps":skipped_steps}
        if method != "head_only":
            row["gradient_diagnostic"]=diagnose(model,batch,settings,profile["seed"],method)
        history.append(row);write_json(out/"history.json",history)
        if method != "head_only" and not all(phase_nonzero):raise RuntimeError("A phase layer did not learn")
        print(f"{method} {epoch}/{profile['adaptation_epochs']} val={metric:.6f} loss={loss_sum/count:.6f}",flush=True)
    if electronic_hash(model)!=frozen_before:raise RuntimeError("Frozen electronics changed")
    if feedback_identity(model,method,profile["seed"]) != feedback:raise RuntimeError("Fixed response changed")
    terminal=None
    if method != "head_only":
        terminal=diagnose(model,batch,settings,profile["seed"],method)
        write_json(out/"last_diagnostic.json",terminal)
    p12.save_torch(out/"last.pt", {"model":model.state_dict(),"profile":profile,"method":method,
        "epoch":profile["adaptation_epochs"],"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
        "scaler":scaler.state_dict(),"rng":p12.capture_rng_state()})
    p12.save_torch(out/"best.pt", {"model":best_state,"profile":profile,"method":method,"best_epoch":best_epoch})
    result={"status":"complete","method":method,"best_validation_pck":best,"best_epoch":best_epoch,
        "last_validation_pck":history[-1]["validation"]["pck_torso_0p2"],
        "epochs":len(history),"frozen_electronic_parameters_unchanged":True,"test_evaluated":False,
        "shared_start_sha256":p12.sha256_file(out.parent/"shared_start.pt"),
        "dataset_manifest_sha256":bundle.manifest_sha256,"source_checkpoint_sha256":model.source_manifest["sha256"],
        "feedback_identity":feedback,"best_checkpoint_sha256":p12.sha256_file(out/"best.pt"),
        "last_checkpoint_sha256":p12.sha256_file(out/"last.pt"),
        "last_phase_diagnostic":terminal,"seconds":sum(x["seconds"] for x in history)}
    write_json(out/"result.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",type=Path,default=ROOT/"FFA/pilot.json")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--gpu",required=True,help="Authorized physical GPU UUID")
    args=parser.parse_args();out=args.out.resolve()
    if (ROOT/"FFA/runs").resolve() not in out.parents:raise ValueError("Output must be inside FFA/runs")
    if out.exists():raise FileExistsError("Use a fresh run identity; no silent overwrite/resume")
    out.mkdir(parents=True)
    profile=json.loads(args.profile.read_text(encoding="utf-8-sig"))
    if set(profile["methods"])-ALLOWED:raise ValueError("Unknown method")
    if profile["task"] != "lsp":raise ValueError("This pilot only implements LSP")
    inventory=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],text=True)
    if any(line.strip().startswith(args.gpu) for line in inventory.splitlines()):
        write_json(out/"failure.json",{"status":"blocked_gpu_busy","inventory":inventory,"gpu_uuid":args.gpu})
        raise RuntimeError(f"Requested GPU already has a compute process: {inventory}")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    started=time.time()
    try:
        p12.seed_everything(profile["seed"])
        torch.set_num_threads(4)
        device=torch.device("cuda:0")
        # Load the existing recipe as an architecture/data contract, then define
        # a separate bounded FFA protocol. P12's formal settings are never edited.
        base=ROOT/"FixedFeedbackSFT/projects/qwen3_vl_patch_stem_8stage_separable_optical_downstream_fa/configs/base_50e.yaml"
        settings=load_settings(base,task="lsp",method="bp",seed=profile["seed"],output_dir=out/"data",
            limits={"max_train_samples":profile["train_samples"],"max_validation_samples":profile["validation_samples"],"max_test_samples":1})
        task=replace(settings.task_settings,train_batch_size=profile["train_batch_size"],evaluation_batch_size=profile["evaluation_batch_size"],num_workers=profile["num_workers"])
        settings=replace(settings,task_settings=task)
        identity={"format":"ffa-intensity-response-pilot-v1","profile":profile,
            "git_commit":p12.git_commit(ROOT),"implementation_files":source_manifest(),
            "tracked_dirty":bool(subprocess.check_output(["git","diff","--name-only","HEAD"],cwd=ROOT,text=True).strip()),
            "gpu_uuid":args.gpu,"gpu_name":torch.cuda.get_device_name(),"started_unix":started,
            "protocol":"new warmup on cached deterministic train features; augmented phase+head adaptation; validation only",
            "test_evaluated":False}
        write_json(out/"identity.json",identity)
        write_json(out/"resolved_p12_contract.json",settings.to_dict())
        model=p12.build_model(settings).to(device)
        data_config,bundle,loaders=p12.build_data(settings)
        write_json(out/"asset_identity.json", {"source":model.source_manifest,
            "stem_file_sha256":p12.sha256_file(settings.paths.stem_checkpoint),
            "dataset_manifest_sha256":bundle.manifest_sha256,"dataset_summary":bundle.metadata,
            "initial_electronic_hash":electronic_hash(model)})
        del loaders
        shared,common_val=warm_start(model,settings,data_config,bundle,profile,device,out)
        results=[]
        for method in profile["methods"]:
            dest=out/method;dest.mkdir()
            results.append(adapt(model,settings,data_config,bundle,profile,method,shared,device,dest))
            write_json(out/"progress.json",{"status":"running","finished_methods":results})
        by={r["method"]:r for r in results}
        gain=by["bp_current"]["best_validation_pck"]-by["head_only"]["best_validation_pck"]
        recovery=None if gain<=0 else (by["source_full_response"]["best_validation_pck"]-by["head_only"]["best_validation_pck"])/gain
        report={"status":"complete","profile":profile,"common_validation_pck":common_val,"results":results,
            "bp_gain_over_equal_budget_head_only":gain,"source_full_response_recovery":recovery,
            "test_evaluated":False,"diagnostic_bp_gain_exists":gain>0,"hardware_experiment":False,
            "dynamic_feedback_formation":False,"total_seconds":time.time()-started,
            "git_commit":identity["git_commit"],"gpu_uuid":args.gpu}
        write_json(out/"result.json",report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except Exception:
        write_json(out/"failure.json",{"status":"failed","traceback":traceback.format_exc(),"seconds":time.time()-started})
        raise
    finally:
        if torch.cuda.is_initialized():torch.cuda.empty_cache()


if __name__=="__main__":
    main()
