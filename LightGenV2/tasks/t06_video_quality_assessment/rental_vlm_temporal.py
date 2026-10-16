"""Rental A10080: full DeepSeek-VL2/Kimi-VL, frozen backbone and five native rows.

No LoRA, adapter, generation, backbone training, quantization or model patches.
Accuracy/cost and resident utilization always run in separate fresh processes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import threading
import time

from .rental_vlm_assets import MODELS, sha, check
from .rental_lgvq_transfer import MANIFEST_SHA

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
RUN = TASK / "runs/simulation/lgvq_full_deepseek_kimi_rental_20261017"
GPU = "GPU-780326a6-79bc-7a45-36f6-683c5b278bee"
WORDS = ("Bad", "Poor", "Fair", "Good", "Excellent")
WIDTHS = {"deepseek":2560,"kimi":2048}
DEADLINE = 1792185600.0  # 2026-10-17 05:20 +0800, reserve twenty minutes for backup.

def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    tmp.replace(path)

def protocol(key):
    return dict(model=MODELS[key][0],revision=MODELS[key][1],frames=4,
        frame_fractions=[.1,.37,.63,.9],crop=.65,frame_size=384 if key=="deepseek" else 448,
        train=2250,test=558,validation=0,feature_batch=1,seed=42,epochs=50,
        training_batch=512,lr=.001,weight_decay=0,optimizer="AdamW",scheduler="cosine",loss="CE",
        backbone_trainable_parameters=0,quality_output_trainable_parameters=5*WIDTHS[key],
        feature_cache_dtype="float16",head_dtype="float32",backbone_dtype="bfloat16",
        selection="highest_TEST_SRCC_development_no_VAL",inference_batch=1,
        native_last_token_full_vocabulary_projection_in_evaluation=True,
        attention_backend="official_flash_attention_2" if key=="kimi" else "original_official_DeepSeek_runtime",
        quantization=False,lora=False,adapters=False,generation=False,
        original_manifest_sha256=MANIFEST_SHA,
        host_proxy_w=338.2,host_proxy_note="Retained comparison proxy, NOT measured rental-host wall power",
        hardware_note="New rented A10080 at original default400W, not historical A10040/250W",
        resident=dict(repeats=3,warmup_s=5,measurement_min_s=30,queue_depth=8,poll_ms=50))

def sources():
    files=[Path(__file__),TASK/"rental_vlm_assets.py",TASK/"rental_lgvq_transfer.py",
        TASK/"deepseek_vl2_quality.py",TASK/"quality_token_common.py",TASK/"quality_token_train.py",
        TASK/"batch_telemetry.py",REPO/"LightGenV2/common/baseline_measurement.py"]
    return {str(p.relative_to(REPO)):sha(p) for p in files}

def rows():
    from . import quality_token_common as core
    base=Path("/data/lgvq")
    receipt=json.loads((base/"transfer_receipt.json").read_text())
    if receipt["original_manifest_sha256"]!=MANIFEST_SHA or sha(base/"original_manifest.csv")!=MANIFEST_SHA:
        raise RuntimeError("Original manifest identity changed")
    if sha(base/"manifest.csv")!=receipt["remapped_manifest_sha256"]:
        raise RuntimeError("Remapped paths/labels changed")
    result=core.read_manifest(base/"manifest.csv")
    if len(result)!=2808 or len({r["sample_id"] for r in result})!=2808:
        raise RuntimeError("Exactly2808unique original videos required")
    if [sum(r["split"]==s for r in result) for s in ("train","test")]!=[2250,558]:
        raise RuntimeError("Original2250/558 split required")
    return result

def gpu_records():
    raw=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits"],text=True)
    return [dict(zip(("pid","gpu_uuid","memory_mib"),map(str.strip,r)))
            for r in csv.reader(raw.splitlines()) if len(r)==3 and r[1].strip()==GPU]

class Guard:
    def __init__(self,args,identity):
        self.args,self.identity=args,identity
        self.expected=None;self.audit=[]
        self.monitor_error=None;self.monitor_stop=threading.Event();self.thread=None
    def check(self):
        if time.time()>=min(self.args.deadline,DEADLINE):
            raise TimeoutError("Fixed rental deadline reached")
        if sources()!=self.identity["source_sha256"]:
            raise RuntimeError("Bound source changed during stage")
        records=gpu_records()
        self.audit.append(dict(time=time.time(),container_pid=os.getpid(),processes=records))
        if self.expected is None:
            if records:raise RuntimeError("Rented GPU must be empty before our CUDA initialization")
        elif {r["pid"] for r in records}!=self.expected:
            raise RuntimeError("GPU process identity changed or foreign process entered")
    def bind(self):
        records=gpu_records()
        if len(records)!=1:
            raise RuntimeError("Cannot uniquely map sole initialized CUDA process")
        self.expected={records[0]["pid"]}
        self.identity["cuda_pid_mapping"]=dict(container_pid=os.getpid(),nvidia_smi_pid=records[0]["pid"],gpu_uuid=GPU)
        self.check()
    def start_monitor(self):
        def work():
            while not self.monitor_stop.wait(1):
                try:self.check()
                except BaseException as error:
                    self.monitor_error=error;return
        self.thread=threading.Thread(target=work,daemon=True);self.thread.start()
    def fast_check(self):
        if self.monitor_error:raise RuntimeError("Independent process/source monitor failed") from self.monitor_error
        if time.time()>=self.args.deadline:raise TimeoutError("Rental deadline reached")
    def stop_monitor(self):
        self.monitor_stop.set()
        if self.thread:self.thread.join(timeout=10)
        self.fast_check()

def verify_assets(model):
    metadata=json.loads((model/"official_metadata.json").read_text())
    if metadata["revision"]!=MODELS[model.name][1]:raise RuntimeError("Official revision changed")
    hashes={}
    for record in metadata["assets"]:
        path=model/record["rfilename"]
        if not check(path,record):raise RuntimeError("Official asset mismatch: "+path.name)
        hashes[path.name]=sha(path) if not record.get("lfs") else record["lfs"]["sha256"]
    return hashes

def processor(key,model):
    from transformers import AutoProcessor,AutoConfig
    from . import deepseek_vl2_quality as tiny
    if key=="deepseek":
        from deepseek_vl2.models import DeepseekVLV2Processor
        p=DeepseekVLV2Processor.from_pretrained(str(model))
        cfg=AutoConfig.from_pretrained(str(model),local_files_only=True)
        width=cfg.language_config.hidden_size
        prompt=tiny.build_conversation("temporal",4)
    else:
        p=AutoProcessor.from_pretrained(str(model),trust_remote_code=True,local_files_only=True)
        cfg=AutoConfig.from_pretrained(str(model),trust_remote_code=True,local_files_only=True)
        width=cfg.text_config.hidden_size
        from .quality_token_common import PROMPTS
        content=[]
        for i in range(4):
            content += [dict(type="text",text=f"Frame {i+1}: "),dict(type="image",image=f"frame_{i+1}.png")]
        content.append(dict(type="text",text="\n"+PROMPTS["temporal"]))
        prompt=p.apply_chat_template([dict(role="user",content=content)],add_generation_prompt=True,tokenize=False)
    if width!=WIDTHS[key]:raise RuntimeError("Wrong full model native width")
    ids=[p.tokenizer.encode(w,add_special_tokens=False) for w in WORDS]
    if any(len(v)!=1 for v in ids) or len({v[0] for v in ids})!=5:
        raise RuntimeError(f"Five distinct native single-token quality words required: {ids}")
    return p,prompt,[v[0] for v in ids]

def prepare(key,row,p,prompt):
    from . import deepseek_vl2_quality as tiny
    frames,positions=tiny.decode_ordered_frames(Path(row["video_path"]),[.1,.37,.63,.9],384 if key=="deepseek" else 448)
    if key=="deepseek":
        result=p(conversations=prompt,images=frames,force_batchify=True,system_prompt="")
        values={n:result[n] for n in ("input_ids","attention_mask","images","images_seq_mask","images_spatial_crop")}
        if list(values["images"].shape)!=[1,8,3,384,384] or not bool(values["images_spatial_crop"].eq(1).all()):
            raise RuntimeError("Original four ordered images/global-local tiles changed")
    else:
        values=dict(p(images=frames,text=prompt,return_tensors="pt",padding=True,truncation=False))
        if values["image_grid_hws"].tolist()!=[[32,32]]*4:
            raise RuntimeError("Kimi four448 frames/1024 merged visual tokens required")
    if len(frames)!=4 or not bool(values["attention_mask"][:,-1].all()):
        raise RuntimeError("B1 last prompt token must be valid; no extra frames")
    return values,positions

def transfer(values):
    import torch
    return {k:v.to("cuda",dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in values.items()}

def load_model(key,path):
    import torch
    from transformers import AutoModelForCausalLM,AutoConfig
    options={}
    if key=="kimi":
        cfg=AutoConfig.from_pretrained(str(path),trust_remote_code=True,local_files_only=True)
        # This official config predates sub_configs dispatch in Transformers.
        # Use its supported native per-config backend setting, never patch a forward.
        cfg.vision_config._attn_implementation="flash_attention_2"
        cfg.text_config._attn_implementation="flash_attention_2"
        options=dict(config=cfg,attn_implementation="flash_attention_2")
    model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,trust_remote_code=True,
                torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,**options).eval().requires_grad_(False).cuda()
    if key=="deepseek":
        language=model.language
        first=model.vision.blocks[0]
    else:
        language=model.language_model
        first=model.vision_tower.encoder.blocks[0]
        if first.attn_implementation!="flash_attention_2" or type(language.model.layers[0].self_attn).__name__!="DeepseekV3FlashAttention2":
            raise RuntimeError("Requested official Kimi vision/text backend was not applied")
    return model,language,first

def runtime_identity(model):
    import inspect
    import importlib.metadata
    source=Path(inspect.getfile(type(model)))
    root=source.parent
    files={p.name:sha(p) for p in root.glob("*.py")}
    packages={}
    for name in ("torch","torchvision","transformers","accelerate","xformers","flash-attn","numpy","opencv-python"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    if hasattr(model,"vision_tower"):
        attention=dict(vision=model.vision_tower.encoder.blocks[0].attn_implementation,
                       text_class=type(model.language_model.model.layers[0].self_attn).__name__)
    else:
        attention=dict(vision_class=type(model.vision.blocks[0]).__name__,
                       text_class=type(model.language.model.layers[0].self_attn).__name__)
    return dict(model_class=type(model).__name__,official_runtime_source_sha256=files,packages=packages,
                actual_attention=attention,
                total_model_parameters=sum(p.numel() for p in model.parameters()),
                backbone_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

class ClockSampler:
    """Independent raw clocks/temperature stream; no clock/power mutations."""
    def start(self):
        self.records=[]
        self.process=subprocess.Popen(["nvidia-smi",f"--id={GPU}",
            "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,pstate,power.limit",
            "--format=csv,noheader,nounits","--loop-ms=200"],stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,text=True)
        def read():
            for line in self.process.stdout:
                fields=[v.strip() for v in line.split(",")]
                if len(fields)==5:self.records.append([time.time(),time.monotonic(),*fields])
        self.thread=threading.Thread(target=read,daemon=True);self.thread.start()
    def stop(self,path):
        self.process.terminate();self.process.wait(timeout=5);self.thread.join(timeout=5)
        if not self.records:raise RuntimeError("Missing independent clock telemetry")
        with path.open("x",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["host_time_unix","host_monotonic_s","sm_clock_mhz",
                "memory_clock_mhz","temperature_c","pstate","power_limit_w"])
            writer.writerows(self.records)

def features(key,model,language,inputs):
    import torch
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        if key=="deepseek":
            embeds=model.prepare_inputs_embeds(**inputs)
        else:
            embeds=model.get_input_embeddings()(inputs["input_ids"])
            image_features=model._extract_image_features(inputs["pixel_values"].to(model.vision_tower.dtype),inputs["image_grid_hws"])
            embeds=embeds.to(image_features[0].dtype)
            embeds=model._merge_with_image_features(embeds,inputs["input_ids"],image_features)
        hidden=language.model(inputs_embeds=embeds,attention_mask=inputs["attention_mask"],
                    use_cache=False,return_dict=True).last_hidden_state[:,-1].float()
    if hidden.shape!=(1,WIDTHS[key]):raise RuntimeError("Native last-token feature shape changed")
    return hidden

def five_rows(initial):
    import torch
    class FiveRows(torch.nn.Module):
        def __init__(self,value):
            super().__init__()
            if value.ndim!=2 or value.shape[0]!=5:raise ValueError("Only five native output rows")
            self.weight=torch.nn.Parameter(value.detach().float().clone())
        def forward(self,value):return value.float()@self.weight.t()
    return FiveRows(initial)

def parameter_digest(model):
    """Audit actual in-memory frozen parameters, not only unchanged disk files."""
    import torch
    h=hashlib.sha256()
    for name,value in model.named_parameters():
        if value.requires_grad or value.grad is not None:raise RuntimeError("Backbone not frozen")
        h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def train_head(payload,initial,ids,args,identity):
    import numpy as np
    import torch
    from . import quality_token_train as q
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    x,y=payload["features"].float(),payload["targets"].float()
    tr=torch.tensor([i for i,s in enumerate(payload["splits"]) if s=="train"])
    te=torch.tensor([i for i,s in enumerate(payload["splits"]) if s=="test"])
    bounds=torch.linspace(float(y[tr].min()),float(y[tr].max()),6)[1:-1]
    levels=torch.linspace(float(y[tr].min()),float(y[tr].max()),5)
    labels=torch.bucketize(y,bounds,right=False)
    head=five_rows(initial).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[tr],labels[tr]),batch_size=512,
        shuffle=True,generator=torch.Generator().manual_seed(42),pin_memory=True)
    history=[];best=-float("inf");best_state=None;best_epoch=None
    for epoch in range(1,51):
        if time.time()>=args.deadline:raise TimeoutError("Fixed rental deadline")
        head.train();loss_sum=0.;count=0
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(head(bx.cuda(non_blocking=True)),by.cuda(non_blocking=True))
            if not bool(torch.isfinite(loss)):raise RuntimeError("Nonfinite CE")
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(by);count+=len(by)
        scheduler.step();head.eval()
        metrics,*_=q.evaluate(head,x[te],y[te],labels[te],levels,torch.device("cuda"))
        if not math.isfinite(metrics["srcc"]):raise RuntimeError("Nonfinite TEST metric")
        if metrics["srcc"]>best:
            best=metrics["srcc"];best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append(dict(epoch=epoch,train_cross_entropy=loss_sum/count,test=metrics,
                            learning_rate=optimizer.param_groups[0]["lr"]))
        write(args.output/"train_history.json",history)
        write(args.output/"status.json",dict(status="training_head",epoch=epoch,loss=loss_sum/count,best_test_srcc=best,time=time.time()))
        print(f"[head] {args.model} e{epoch}/50 CE={loss_sum/count:.6f} SRCC={metrics['srcc']:.9f}",flush=True)
    last={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};verification={}
    for name,state,epoch in (("best",best_state,best_epoch),("last",last,50)):
        head.load_state_dict(state)
        gm,gp,*_=q.evaluate(head,x[te],y[te],labels[te],levels,torch.device("cuda"))
        cpu=five_rows(state["weight"]).eval().requires_grad_(False)
        cm,cp,*_=q.evaluate(cpu,x[te],y[te],labels[te],levels,torch.device("cpu"))
        q.atomic_save(args.output/f"{name}_checkpoint.pt",dict(state_dict=state,initial_native_rows=initial,
            quality_token_ids=ids,selected_epoch=epoch,level_scores=levels,boundaries=bounds,
            identity=identity,trainable_parameters=initial.numel(),selection=protocol(args.model)["selection"]))
        with (args.output/f"cpu_{name}_predictions.csv").open("x",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["sample_id","target_mos","prediction_mos"])
            writer.writerows((payload["sample_ids"][i],float(y[i]),float(v)) for i,v in zip(te.tolist(),cp))
        verification[name]=dict(epoch=epoch,gpu=gm,strict_cpu=cm,max_cpu_gpu_difference=float((cp-gp).abs().max()))
    write(args.output/"cpu_verification.json",verification)
    write(args.output/"training_report.json",dict(status="complete",results=verification,identity=identity))

def train(args,identity,guard):
    import torch
    from . import quality_token_train as q
    data=rows();path=Path("/data/models")/args.model
    identity["model_sha256"]=verify_assets(path)
    p,prompt,ids=processor(args.model,path)
    model,language,_=load_model(args.model,path);guard.bind()
    identity["runtime"]=runtime_identity(model)
    write(args.output/"identity.json",identity)
    initial=language.lm_head.weight.detach()[ids].float().cpu().clone()
    identity["initial_rows_sha256"]=q.sha256_tensor(initial)
    write(args.output/"status.json",dict(status="auditing_backbone_before",time=time.time()))
    before=parameter_digest(model)
    # TRAIN-only native-output parity and isolated head update, discarded before fixed-seed training.
    cpu,positions=prepare(args.model,next(r for r in data if r["split"]=="train"),p,prompt)
    values=transfer(cpu)
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        tapped=features(args.model,model,language,values)
        full=model(**values,use_cache=False,return_dict=True).logits[:,-1].float()
        projection=language.lm_head(tapped.to(language.lm_head.weight.dtype)).float()
    if not bool(torch.isfinite(full).all()) or not torch.allclose(full,projection,rtol=.01,atol=.125):
        raise RuntimeError("Official full forward/last-token readout parity failed")
    smoke=five_rows(initial).cuda();old=smoke.weight.detach().clone()
    ys=torch.tensor([r["temporal"] for r in data if r["split"]=="train"])
    bounds=torch.linspace(float(ys.min()),float(ys.max()),6)[1:-1]
    label=torch.bucketize(ys[:1],bounds).cuda()
    loss=torch.nn.functional.cross_entropy(smoke(tapped.cpu().half().float().cuda()),label)
    loss.backward();torch.optim.AdamW(smoke.parameters(),lr=.001,weight_decay=0.).step()
    if torch.equal(old,smoke.weight):raise RuntimeError("Independent five-row smoke did not update")
    write(args.output/"native_smoke.json",dict(status="passed",real_train_videos=1,
        native_logit_max_abs=float((full-projection).abs().max()),rtol=.01,atol=.125,
        trainable_parameters=initial.numel(),probe_ce=float(loss.detach()),positions=positions,
        input_shapes={k:list(v.shape) for k,v in cpu.items()},quality_token_ids=ids))
    del values,cpu,tapped,full,projection,smoke,old,loss;torch.cuda.empty_cache()
    x=torch.empty((2808,WIDTHS[args.model]),dtype=torch.float16)
    (args.output/"feature_parts").mkdir()
    for begin in range(0,2808,64):
        guard.check();end=min(begin+64,2808)
        for i in range(begin,end):
            cpu,_=prepare(args.model,data[i],p,prompt);values=transfer(cpu)
            value=features(args.model,model,language,values).cpu().half()
            if not bool(torch.isfinite(value).all()):raise RuntimeError("Nonfinite features")
            x[i:i+1]=value
            del cpu,values,value
        q.atomic_save(args.output/f"feature_parts/rows_{begin:05d}_{end:05d}.pt",
            dict(features=x[begin:end].clone(),sample_ids=[r["sample_id"] for r in data[begin:end]],identity=identity))
        write(args.output/"status.json",dict(status="extracting",videos=end,total=2808,time=time.time()))
        print(f"[features] {args.model} {end}/2808",flush=True)
    write(args.output/"status.json",dict(status="auditing_backbone_after",videos=2808,time=time.time()))
    after=parameter_digest(model)
    if before!=after or not torch.equal(initial,language.lm_head.weight.detach()[ids].float().cpu()):
        raise RuntimeError("Frozen in-memory backbone/native rows changed")
    write(args.output/"frozen_audit.json",dict(status="passed",before_sha256=before,after_sha256=after,
                all_parameters_require_grad_false=True,all_backbone_gradients_none=True))
    payload=dict(features=x,targets=torch.tensor([r["temporal"] for r in data]),splits=[r["split"] for r in data],
                sample_ids=[r["sample_id"] for r in data],initial_native_rows=initial,quality_token_ids=ids,identity=identity)
    q.atomic_save(args.output/"features.pt",payload)
    del model,language;torch.cuda.empty_cache()
    train_head(payload,initial,ids,args,identity)
    if verify_assets(path)!=identity["model_sha256"]:raise RuntimeError("Official files changed")
    guard.check()
    # Persist all fields before hashing; main's final same-byte write is then safe.
    write(args.output/"identity.json",identity)
    artifacts={str(f.relative_to(args.output)):sha(f) for f in args.output.rglob("*") if f.is_file() and f.name not in ("status.json","training_receipt.json")}
    write(args.output/"training_receipt.json",dict(status="complete",identity=identity,artifacts=artifacts))

def inference_runtime(args,identity,guard):
    import torch
    path=Path("/data/models")/args.model
    ptpath=RUN/args.model/"train/best_checkpoint.pt"
    receipt=json.loads((ptpath.parent/"training_receipt.json").read_text())
    for name,value in receipt["artifacts"].items():
        if sha(ptpath.parent/name)!=value:raise RuntimeError("Training artifact SHA mismatch: "+name)
    identity["model_sha256"]=verify_assets(path)
    if receipt["identity"]["model_sha256"]!=identity["model_sha256"] or receipt["identity"]["source_sha256"]!=identity["source_sha256"]:
        raise RuntimeError("Training/evaluation identity differs")
    p,prompt,ids=processor(args.model,path)
    model,language,first=load_model(args.model,path);guard.bind()
    identity["runtime"]=runtime_identity(model)
    if receipt["identity"]["runtime"]!=identity["runtime"]:raise RuntimeError("Frozen runtime changed")
    write(args.output/"identity.json",identity)
    pt=torch.load(ptpath,map_location="cpu",weights_only=False)
    if ids!=pt["quality_token_ids"]:raise RuntimeError("Quality token IDs changed")
    head=five_rows(pt["state_dict"]["weight"]).cuda().eval().requires_grad_(False)
    levels=pt["level_scores"].cuda()
    identity.update(checkpoint_sha256=sha(ptpath),quality_token_ids=ids,
                    training_receipt_sha256=sha(ptpath.parent/"training_receipt.json"))
    def compute(values):
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            hidden=features(args.model,model,language,values)
            native_logits=language.lm_head(hidden.to(language.lm_head.weight.dtype)).float()
        # FP32 quality head is OUTSIDE autocast, as in training and prior baselines.
        with torch.inference_mode():
            mos=head(hidden).softmax(-1)@levels
        return mos,native_logits
    return model,p,prompt,first,compute

def evaluate(args,identity,guard):
    import numpy as np
    import torch
    from . import quality_token_common as core
    from . import batch_telemetry as telemetry
    from LightGenV2.common import baseline_measurement as measurement
    test=[r for r in rows() if r["split"]=="test"]
    model,p,prompt,first,compute=inference_runtime(args,identity,guard)
    timer=measurement.FirstBlockTimer(first);sampler=telemetry.TelemetrySampler(GPU);clocks=ClockSampler()
    clocks.start()
    sampler.start();sampler.set_phase("idle");time.sleep(2);sampler.set_phase(None)
    predictions=[];timings=[]
    try:
        with (args.output/"predictions.csv").open("x",newline="") as ps, (args.output/"batch_timing.csv").open("x",newline="") as ts:
            pw,tw=csv.writer(ps),csv.writer(ts)
            pw.writerow(["sample_index","sample_id","target_mos","prediction","frame_positions"])
            tw.writerow(["batch_index","videos","in_cost_subset","synchronized_wall_ms","cuda_ms","preprocessing_audit_ms","h2d_audit_ms"])
            for i,row in enumerate(test):
                if i%25==0:guard.check()
                start=time.perf_counter();cpu,positions=prepare(args.model,row,p,prompt)
                prep=1000*(time.perf_counter()-start)
                start=time.perf_counter();values=transfer(cpu);torch.cuda.synchronize()
                h2d=1000*(time.perf_counter()-start)
                timer.reset();sampler.set_phase(f"active:{i}" if i<200 else None)
                try:
                    mos,raw=compute(values);result=timer.finish()
                finally:sampler.set_phase(None)
                value=float(mos.detach().cpu()[0])
                if not math.isfinite(value):raise RuntimeError("Nonfinite native MOS")
                predictions.append(value)
                record=[i,1,i<200,result["host_ms"],result["cuda_ms"],prep,h2d];timings.append(record)
                tw.writerow(record);pw.writerow([i,row["sample_id"],row["temporal"],value,json.dumps(positions)])
                ps.flush();ts.flush();del values,cpu,mos,raw
                if i%25==0 or i==557:
                    write(args.output/"status.json",dict(status="evaluating",videos=i+1,total=558,time=time.time()))
                    print(f"[B1] {args.model} {i+1}/558 wall={result['host_ms']:.3f}ms",flush=True)
        samples=sampler.stop();telemetry.save_samples(args.output/"telemetry.csv",samples)
        cost=[r for r in timings if r[2]]
        power=measurement.power_report(samples,[r[3] for r in cost],power_limit_w=identity["power_limit_w"])
        report=dict(status="complete",identity=identity,performance=core.metrics(np.array([r["temporal"] for r in test]),np.array(predictions)),
            test_videos=558,cost_videos=200,batch=1,cost=telemetry.workload_cost(cost,power["active_mean_w"]),
            power=power,telemetry=telemetry.telemetry_report(samples),explicit_warmup_forwards=0,first_batch_included=True,
            boundary="native vision block0 input to GPU MOS synchronization",excluded=["decode","preprocessing","H2D","model loading","audit"],
            resident_utilization="separate fresh-process experiment; never used for dataset-once cost",
            host_energy_note=protocol(args.model)["host_proxy_note"])
        if verify_assets(Path("/data/models")/args.model)!=identity["model_sha256"]:raise RuntimeError("Official model changed")
        guard.check();write(args.output/"report.json",report)
    finally:
        if sampler.process is not None:telemetry.save_samples(args.output/"telemetry.csv",sampler.stop())
        clocks.stop(args.output/"clock_telemetry.csv")
        timer.close()

def utilization(args,identity,guard):
    import torch
    from . import batch_telemetry as telemetry
    test=[r for r in rows() if r["split"]=="test"]
    model,p,prompt,_,compute=inference_runtime(args,identity,guard)
    cpu,positions=prepare(args.model,test[0],p,prompt);inputs=transfer(cpu)
    torch.cuda.synchronize();reports=[]
    guard.start_monitor()
    # No hook/timer overhead in resident forwards; same one-stream queue8 method.
    for repeat in range(1,4):
        folder=args.output/f"repeat{repeat:02d}";folder.mkdir()
        sampler=telemetry.TelemetrySampler(GPU);groups=[];sampler.start();clocks=ClockSampler();clocks.start()
        def phase(name,minimum):
            sampler.set_phase(name);started=time.monotonic()
            while time.monotonic()-started<minimum:
                guard.fast_check();tick=time.monotonic()
                for _ in range(8):mos,raw=compute(inputs)
                torch.cuda.synchronize()
                groups.append(dict(phase=name,start=tick,end=time.monotonic(),forwards=8))
            return started,time.monotonic(),mos
        try:
            phase("warmup",5);start,end,final_mos=phase("measured",30)
            samples=sampler.stop();selected=[s for s in samples if start<=s.host_monotonic_s<end]
            result=telemetry.telemetry_report(selected,prefix="")
            result.update(start_monotonic_s=start,end_monotonic_s=end,duration_s=end-start,
                measured_forwards=sum(g["forwards"] for g in groups if g["phase"]=="measured"),
                sample_id=test[0]["sample_id"],positions=positions,batch=1,
                final_mos=final_mos.detach().cpu().tolist())
            result["full_forward_mean_ms"]=1000*(end-start)/result["measured_forwards"]
            write(folder/"report.json",result);reports.append(result)
        finally:
            if sampler.process is not None:samples=sampler.stop()
            telemetry.save_samples(folder/"telemetry.csv",samples);write(folder/"forward_groups.json",groups)
            clocks.stop(folder/"clock_telemetry.csv")
    total=sum(r["samples"] for r in reports)
    guard.stop_monitor()
    pooled=sum(r["mean_gpu_utilization_percent"]*r["samples"] for r in reports)/total
    if verify_assets(Path("/data/models")/args.model)!=identity["model_sha256"]:raise RuntimeError("Official model changed")
    guard.check();write(args.output/"report.json",dict(status="complete",identity=identity,repeats=reports,
        pooled_mean_utilization_percent=pooled,total_samples=total,method=protocol(args.model)["resident"],
        costs_not_derived_from_resident_power=True))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--phase",choices=("cpu-preflight","train","evaluate","utilization"),required=True)
    p.add_argument("--deadline",type=float,default=DEADLINE)
    args=p.parse_args();args.deadline=min(args.deadline,DEADLINE)
    args.output=RUN/args.model/{"cpu-preflight":"cpu_preflight","train":"train","evaluate":"a100_batch01_cost","utilization":"resident_batch01"}[args.phase]
    if not REPO.is_relative_to("/data") or args.output.exists():raise RuntimeError("New canonical /data output only")
    args.output.mkdir(parents=True)
    identity=dict(model=args.model,phase=args.phase,source_sha256=sources(),contract=protocol(args.model),
        command=sys.argv,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        pid=os.getpid(),gpu_uuid=GPU,created=time.time(),python=platform.python_version())
    write(args.output/"identity.json",identity)
    guard=Guard(args,identity)
    try:
        if args.phase=="cpu-preflight":
            data=rows();processor_object,prompt,ids=processor(args.model,Path("/data/models")/args.model)
            values,positions=prepare(args.model,next(r for r in data if r["split"]=="train"),processor_object,prompt)
            import torch
            write(args.output/"report.json",dict(status="passed",quality_token_ids=ids,prompt=prompt,
                input_shapes={k:list(v.shape) for k,v in values.items()},positions=positions,
                cuda_initialized=torch.cuda.is_initialized(),identity=identity))
        else:
            if os.environ.get("CUDA_VISIBLE_DEVICES")!=GPU:raise RuntimeError("Bind the rented GPU by exact UUID")
            guard.check()
            raw=subprocess.check_output(["nvidia-smi",f"--id={GPU}","--query-gpu=name,power.limit,memory.free","--format=csv,noheader,nounits"],text=True)
            name,power,free=next(csv.reader(raw.splitlines()))
            if "A100" not in name or float(free)<75000:raise RuntimeError("Exclusive rented80GB A100 initial capacity required")
            import torch,transformers
            torch.set_num_threads(8)
            identity.update(gpu_name=name,power_limit_w=float(power),torch=torch.__version__,cuda=torch.version.cuda,
                            transformers=transformers.__version__)
            {"train":train,"evaluate":evaluate,"utilization":utilization}[args.phase](args,identity,guard)
        write(args.output/"identity.json",identity)
        write(args.output/"status.json",dict(status="complete",time=time.time()))
    except BaseException as error:
        write(args.output/"status.json",dict(status="failed",error=repr(error),time=time.time()))
        raise
    finally:
        try:
            if guard.thread and guard.thread.is_alive():guard.stop_monitor()
        finally:
            write(args.output/"process_audit.json",guard.audit)

if __name__=="__main__":main()
