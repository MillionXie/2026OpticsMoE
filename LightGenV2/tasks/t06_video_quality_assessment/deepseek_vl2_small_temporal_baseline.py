"""DeepSeek-VL2-Small four ordered native images / five-row head, one B1 trial.

All official dependency files stay byte-identical; no generation or thinking.
Original data, output-row initialization, loss, optimizer and budget are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from . import internvl35_temporal_baseline as previous

shared, frozen, TASK, REPO = previous.shared, previous.frozen, previous.TASK, previous.REPO
CONFIG = TASK / "configs/baselines/deepseek_vl2_small_temporal_4f.json"


def config():
    c = json.loads(CONFIG.read_text())
    for k,v in dict(model_id="deepseek-ai/deepseek-vl2-small",
        model_revision="6033e16432a1d771cf9fe4a6f894ff5e5e1459af",
        hidden_width=2048, frames=4, frame_fractions=[.1,.37,.63,.9],
        center_crop_short_side_fraction=.65,native_frame_size=384,
        quality_output_trainable_parameters=10240,backbone_trainable_parameters=0,
        epochs=50,training_batch_size=512,feature_batch_size=1,learning_rate=.001,
        weight_decay=0,seed=42,loss="cross_entropy_only",scheduler="cosine",
        train_samples=2250,test_samples=558,inference_batches=[1],thinking=False,
        manifest_sha256=frozen.MANIFEST_SHA).items():
        if c.get(k)!=v:raise ValueError("Original protocol changed: "+k)
    return c

def sources():
    names=[Path(__file__),CONFIG,TASK/"deepseek_vl2_quality.py",TASK/"glm46v_assets.py"]
    return {**previous.sources(),**{str(p.relative_to(REPO)):frozen.sha256(p) for p in names}}

def guard(args,identity):
    if sources()!=identity["source_sha256"]:raise RuntimeError("Source changed during run; preserve evidence")
    records=previous.gpu_guard(args)
    raw=subprocess.check_output(["nvidia-smi",f"--id={frozen.A100_UUID}",
        "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True)
    if not any(r["pid"]==str(os.getpid()) for r in records) and float(raw.strip())<37000:
        raise RuntimeError("Small initial admission needs37000MiB free; no reservation or eviction")
    return records

def runtime_sources():
    import importlib.util,importlib.metadata
    if importlib.metadata.version("transformers")!="4.38.2" or importlib.metadata.version("xformers")!="0.0.29.post2":
        raise RuntimeError("Use existing pinned Tiny environment; no dependency upgrades")
    spec=importlib.util.find_spec("deepseek_vl2")
    root=Path(next(iter(spec.submodule_search_locations)))
    names=["__init__.py","models/__init__.py","models/modeling_deepseek_vl_v2.py",
        "models/modeling_deepseek.py","models/processing_deepseek_vl_v2.py",
        "models/configuration_deepseek.py","models/siglip_vit.py","models/conversation.py"]
    hashes={n:frozen.sha256(root/n) for n in names}
    expected={"models/modeling_deepseek_vl_v2.py":"ffb3d419ef869e77587402a890807efd8c4c719a773e630b0234fe662e2d9610",
        "models/modeling_deepseek.py":"9c7309b9ba7e8c2548ec9e11d6f8fc23256103ed0b58925ea9429e014995578f",
        "models/processing_deepseek_vl_v2.py":"b575ec3dbc83fcc989f753722d63180efd577fe781b7b5f492da310d5548927b",
        "models/configuration_deepseek.py":"6ab21f29a4722e26fa28c8e0d4277591689a598df17cf6c712330e8f62b3fc7c"}
    if any(hashes[n]!=v for n,v in expected.items()):
        raise RuntimeError("Audited original DeepSeek runtime changed; do not patch third-party graph")
    return hashes

def native_model(path,*,meta=False):
    import torch
    from accelerate import init_empty_weights
    from deepseek_vl2.models import DeepseekVLV2Processor
    from transformers import AutoConfig,AutoModelForCausalLM
    from . import deepseek_vl2_quality as tiny
    runtime_sources()
    original_receipt=TASK/"runs/simulation/lgvq_temporal_reproduction_a100_20261013/deepseek/reproduction_receipt.json"
    if json.loads(original_receipt.read_text())["original_source_sha256"]!=frozen.sha256(Path(tiny.__file__)):
        raise RuntimeError("Original Tiny wrapper differs from its reproduction receipt")
    processor=DeepseekVLV2Processor.from_pretrained(str(path))
    encoded=[processor.tokenizer.encode(w,add_special_tokens=False) for w in tiny.QUALITY_WORDS]
    if any(len(v)!=1 for v in encoded):raise ValueError("Audit non-single quality token before changing initialization")
    cfg=AutoConfig.from_pretrained(str(path),local_files_only=True)
    if cfg.language_config.hidden_size!=2048:raise ValueError("Small native hidden2048 required")
    if meta:
        with init_empty_weights():model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=True)
    else:
        model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,trust_remote_code=True,
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).cuda()
    model=model.eval().requires_grad_(False)
    model._quality_processor=processor
    model._quality_prompt=tiny.build_conversation("temporal",4)
    return model,[v[0] for v in encoded]

def prepare_batch(rows,model):
    from . import deepseek_vl2_quality as tiny
    from . import quality_token_common as core
    if len(rows)!=1:raise ValueError("Small feature and inference batch must be1 for40GB capacity")
    frames,positions=tiny.decode_ordered_frames(Path(rows[0]["video_path"]),core.FRAME_FRACTIONS[4],384)
    prepared=model._quality_processor(conversations=tiny.build_conversation("temporal",4),
        images=frames,force_batchify=True,system_prompt="")
    if len(frames)!=4 or tuple(prepared.images.shape)!=(1,8,3,384,384):
        raise RuntimeError("Expected original four384 frames, each native global/local tile")
    if tuple(prepared.images_spatial_crop.shape)!=(1,4,2) or not bool(prepared.images_spatial_crop.eq(1).all()):
        raise RuntimeError("No additional dynamic crops allowed for four-image native input")
    names=("input_ids","attention_mask","images","images_seq_mask","images_spatial_crop")
    return {k:prepared[k] for k in names},[positions]

def features(model,inputs):
    import torch
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        embeds=model.prepare_inputs_embeds(**inputs)
        hidden=model.language.model(inputs_embeds=embeds,attention_mask=inputs["attention_mask"],
            use_cache=False,return_dict=True).last_hidden_state
        mask=inputs["attention_mask"].bool()
        last=torch.arange(mask.shape[1],device=hidden.device).expand_as(mask).masked_fill(~mask,-1).amax(1)
        value=hidden[torch.arange(len(hidden),device=hidden.device),last].float()
    if value.shape!=(1,2048) or not bool(torch.isfinite(value).all()):
        raise RuntimeError("Invalid native Small last-token features")
    return value

def five_native_rows(rows):
    """Independent Small output rows at its native2048 width.

    Import Torch lazily so metadata-only contract checks remain CPU-runtime-free.
    Only the native hidden width differs; initialization and FP32 linear algebra
    are identical to the original five-row protocol.
    """
    import torch

    class SmallFiveNativeRows(torch.nn.Module):
        def __init__(self, native_rows):
            super().__init__()
            if native_rows.shape != (5, 2048):
                raise ValueError(f"Expected Small [5,2048] native rows, got {tuple(native_rows.shape)}")
            self.weight = torch.nn.Parameter(native_rows.detach().float().clone())

        def forward(self, hidden):
            return hidden.float() @ self.weight.t()

    return SmallFiveNativeRows(rows)

def transfer(inputs):
    import torch
    return {k:v.to("cuda",dtype=torch.bfloat16 if k=="images" else v.dtype) for k,v in inputs.items()}


def native_logit_parity(native_logits,tapped_logits):
    """Compare values in FP32; official forward casts logits to FP32 itself.

    This is an audit-only cast, not a change to either model forward path.
    """
    import torch
    if native_logits.shape!=tapped_logits.shape:
        raise RuntimeError("Native final-token logit shape mismatch")
    a,b=native_logits.detach().float(),tapped_logits.detach().float()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise RuntimeError("Nonfinite native final-token logits")
    difference=float((a-b).abs().max())
    if not torch.allclose(a,b,rtol=.01,atol=.125):
        raise RuntimeError(f"Native final-token feature tap parity failed: {difference}")
    return dict(status="passed",max_abs=difference,rtol=.01,atol=.125,
        native_dtype=str(native_logits.dtype),tapped_dtype=str(tapped_logits.dtype),
        comparison_dtype="torch.float32",audit_only_cast=True)


def verify_model_assets(args,c,stage):
    """Full official checks once per asset, before AND after every GPU stage.

    Do not call the shared checker here: it separately validates and then hashes
    each large shard, reading the same bytes twice. No mtime-only cache is used.
    """
    result={};started=time.time()
    for name,spec in c["official_assets"].items():
        path=args.model/name
        if not path.is_file() or ("size" in spec and path.stat().st_size!=spec["size"]):
            raise ValueError("Official model asset size mismatch: "+name)
        if "sha256" in spec:
            actual=frozen.sha256(path)
            if actual!=spec["sha256"]:raise ValueError("Official model asset SHA mismatch: "+name)
        else:
            data=path.read_bytes()
            blob=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            if blob!=spec["git_blob_sha1"]:raise ValueError("Official model Git blob mismatch: "+name)
            actual=hashlib.sha256(data).hexdigest()
        result[name]=actual
        frozen.write_json(args.output/"status.json",dict(status="verifying_model_assets_"+stage,
            verified_assets=len(result),total_assets=len(c["official_assets"]),last_verified_asset=name,
            elapsed_s=time.time()-started,updated_at_unix=time.time()))
        print(f"[assets {stage}] {len(result)}/{len(c['official_assets'])} {name}",flush=True)
    return result


def prepare_assets(args,c,identity):
    import requests
    from .glm46v_assets import range_request,CHUNK
    from .prepare_vlm_assets import validate_profile
    validate_profile(c)
    model=args.model
    if model!=Path("/DATA/DATA1/guest3/models/deepseek-vl2-small-6033e164"):
        raise ValueError("Only canonical approved Small model path")
    model.mkdir(parents=True,exist_ok=True)
    deadline=min(time.time()+3600,args.deadline_unix) if args.deadline_unix else time.time()+3600
    if deadline<=time.time():raise TimeoutError("Original asset deadline passed")
    for name in sorted(c["official_assets"],key=lambda n:n.endswith(".safetensors")):
        if sources()!=identity["source_sha256"]:raise RuntimeError("Source changed during preparation")
        spec=c["official_assets"][name];target=model/name
        if target.exists():
            if not shared.asset_valid(target,spec):raise ValueError("Preserve conflicting official asset:"+name)
            continue
        partial=model/(name+".partial")
        if partial.exists():raise FileExistsError("Preserve partial; no blind restart:"+name)
        url=f"https://hf-mirror.com/{c['model_id']}/resolve/{c['model_revision']}/{name}"
        print("[prepare] "+name,flush=True)
        if not name.endswith(".safetensors"):
            for attempt in range(1,4):
                if time.time()>=deadline:raise TimeoutError("Asset deadline")
                with requests.get(url,timeout=(20,60)) as response:
                    if response.status_code in (429,500,502,503,504) and attempt<3:
                        from email.utils import parsedate_to_datetime
                        h=response.headers.get("Retry-After")
                        try:delay=max(0,float(h)) if h else 30*attempt
                        except ValueError:delay=max(0,parsedate_to_datetime(h).timestamp()-time.time())
                        if time.time()+delay>=deadline:raise TimeoutError("Retry-After exceeds original bound")
                        frozen.write_json(args.output/f"metadata_{name}_{attempt}.json",
                            dict(status=response.status_code,retry_delay=delay,at_unix=time.time()))
                        time.sleep(delay);continue
                    response.raise_for_status()
                    with partial.open("xb") as stream:stream.write(response.content)
                    break
        else:
            with partial.open("xb"):pass
            start=0;total=spec["size"]
            while start<total:
                end=min(start+CHUNK,total)-1;count=0
                with range_request(requests,url,start,end,total,deadline,args.output) as response:
                    with partial.open("ab") as stream:
                        for chunk in response.iter_content(4*1024*1024):
                            if time.time()>=deadline:raise TimeoutError("Original asset deadline")
                            if chunk:
                                if count+len(chunk)>end-start+1:raise RuntimeError("Oversized range")
                                stream.write(chunk);count+=len(chunk)
                if count!=end-start+1:raise RuntimeError("Truncated range; preserve partial no automatic retry")
                start+=count
                frozen.write_json(args.output/"status.json",dict(status="downloading",file=name,
                    bytes=start,total=total,updated_at_unix=time.time(),gpu_initialized=False))
                print(f"[download] {name} {start}/{total}",flush=True)
        if not shared.asset_valid(partial,spec):raise ValueError("Official SHA mismatch; preserve partial:"+name)
        if target.exists():raise FileExistsError("Target appeared; no overwrite")
        partial.rename(target)
    hashes=shared.model_files(model,c)
    frozen.write_json(args.output/"model_asset_manifest.json",dict(identity=identity,
        sha256=hashes,verified_official_git_blobs_and_lfs_sha=True,gpu_initialized=False))


def train_rows(payload, initial, ids, args, c, identity):
    import numpy as np
    import torch
    from . import quality_token_train as q
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    x, y = payload["features"].float(), payload["targets"].float()
    tr = torch.tensor([i for i,s in enumerate(payload["splits"]) if s == "train"])
    te = torch.tensor([i for i,s in enumerate(payload["splits"]) if s == "test"])
    if (len(tr), len(te)) != (2250, 558):
        raise RuntimeError("Original split changed")
    boundaries = torch.linspace(float(y[tr].min()), float(y[tr].max()), 6)[1:-1]
    levels = torch.linspace(float(y[tr].min()), float(y[tr].max()), 5)
    labels = torch.bucketize(y, boundaries, right=False)
    head = five_native_rows(initial).cuda()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[tr], labels[tr]), batch_size=512,
        shuffle=True, generator=torch.Generator().manual_seed(42), pin_memory=True)
    history, best, best_state, best_epoch = [], -float("inf"), None, None
    for epoch in range(1, 51):
        guard(args, identity); head.train(); loss_sum, count = 0., 0
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(bx.cuda(non_blocking=True)), by.cuda(non_blocking=True))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite CE")
            loss.backward(); optimizer.step(); loss_sum += float(loss.detach())*len(by); count += len(by)
        scheduler.step(); head.eval()
        metrics, *_ = q.evaluate(head, x[te], y[te], labels[te], levels, torch.device("cuda"))
        if not np.isfinite(metrics["srcc"]):
            raise RuntimeError("Nonfinite TEST SRCC")
        history.append(dict(epoch=epoch, train_cross_entropy=loss_sum/count,
            learning_rate=optimizer.param_groups[0]["lr"], test=metrics))
        if metrics["srcc"] > best:
            best, best_epoch = metrics["srcc"], epoch
            best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        frozen.write_json(args.output/"train_history.json", history)
        frozen.write_json(args.output/"status.json", dict(status="training_head", epoch=epoch,
            train_loss=loss_sum/count, best_test_srcc=best, updated_at_unix=time.time()))
        print(f"[head] e{epoch}/50 CE={loss_sum/count:.6f} SRCC={metrics['srcc']:.9f}", flush=True)
    last = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    results = {}
    for name,state,epoch in (("best",best_state,best_epoch),("last",last,50)):
        head.load_state_dict(state)
        gm,gp,*_ = q.evaluate(head,x[te],y[te],labels[te],levels,torch.device("cuda"))
        cpu = five_native_rows(state["weight"]).eval().requires_grad_(False)
        cm,cp,*_ = q.evaluate(cpu,x[te],y[te],labels[te],levels,torch.device("cpu"))
        q.atomic_save(args.output/f"{name}_checkpoint.pt",dict(architecture="frozen_deepseek_vl2_small_five_untied_native_rows",
            state_dict=state, initial_native_rows=initial, quality_token_ids=ids, selected_epoch=epoch,
            boundaries=boundaries, level_scores=levels, metrics=gm, trainable_parameters=10240,
            all_backbone_parameters_frozen=True, selection=c["selection"], identity=identity))
        with (args.output/f"cpu_{name}_predictions.csv").open("x",newline="") as f:
            w=csv.writer(f);w.writerow(["sample_id","target_mos","prediction_mos"])
            w.writerows((payload["sample_ids"][i],float(y[i]),float(v)) for i,v in zip(te.tolist(),cp))
        results[name]=dict(epoch=epoch,gpu_metrics=gm,strict_cpu_metrics=cm,
            max_cpu_gpu_mos_difference=float((cp-gp).abs().max()))
    frozen.write_json(args.output/"cpu_verification.json",results)
    frozen.write_json(args.output/"training_report.json",dict(status="complete",contract=c,results=results,
        output_row_parameters=10240,backbone_updates=0,initial_rows_sha256=q.sha256_tensor(initial),identity=identity))


def run(args,c,identity):
    import torch
    from . import quality_token_common as core
    from . import quality_token_train as q
    rows=core.read_manifest(args.manifest)
    if len(rows)!=2808 or len({r["sample_id"] for r in rows})!=2808:
        raise RuntimeError("Expected2808unique")
    model,ids=native_model(args.model,meta=args.phase=="cpu-preflight")
    if args.phase=="cpu-preflight":
        results=[]
        for n in (1,):
            inputs,pos=prepare_batch([r for r in rows if r["split"]=="train"][:n],model)
            results.append(dict(batch=n,shapes={k:list(v.shape) for k,v in inputs.items()},frame_positions=pos,
                images_spatial_crop=inputs["images_spatial_crop"].tolist(),
                visual_tokens_per_video=inputs["images_seq_mask"].sum(1).tolist()))
        frozen.write_json(args.output/"cpu_preflight_report.json",dict(status="passed",results=results,
            native_quality_token_ids=ids,thinking=False,gpu_initialized=torch.cuda.is_initialized(), prompt=model._quality_prompt,
            trainable_output_parameters=10240, runtime_sha256=runtime_sources()))
        return
    initial=model.language.lm_head.weight.detach()[ids].float().cpu().clone()
    if args.phase=="gpu-smoke":
        inputs,pos=prepare_batch([r for r in rows if r["split"]=="train"][:1],model)
        gpu_inputs=transfer(inputs)
        tapped=features(model,gpu_inputs)
        # Compare unmodified official full forward with the original last-prompt-token readout.
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            native_logits=model(**gpu_inputs,use_cache=False,return_dict=True).logits[:,-1]
            tapped_logits=model.language.lm_head(tapped.to(model.language.lm_head.weight.dtype))
        parity=native_logit_parity(native_logits,tapped_logits)
        frozen.write_json(args.output/"native_logit_parity.json",parity)
        native_difference=parity["max_abs"]
        x=tapped.cpu().half().clone()
        head=five_native_rows(initial).cuda()
        # Isolated TRAIN-only optimization probe; no performance claim or model selection.
        ys=[r["temporal"] for r in rows if r["split"]=="train"]
        boundaries=torch.linspace(min(ys),max(ys),6)[1:-1]
        labels=torch.bucketize(torch.tensor(ys[:1]),boundaries).cuda()
        loss=torch.nn.functional.cross_entropy(head(x.float().cuda()),labels)
        loss.backward();torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0.).step()
        if not bool(torch.isfinite(loss)) or torch.equal(head.weight.detach().cpu(),initial):
            raise RuntimeError("Five-row update smoke failed")
        q.atomic_save(args.output/"smoke_evidence.pt",dict(features=x,initial=initial,updated=head.weight.detach().cpu()))
        frozen.write_json(args.output/"gpu_smoke_report.json",dict(status="passed",trainable_parameters=10240,
            real_train_videos=1,probe_loss=float(loss.detach()),frame_positions=pos,not_a_performance_result=True,
            native_forward_last_logits_vs_feature_tap_max_abs=native_difference))
    elif args.phase=="train":
        x=torch.empty((2808,2048),dtype=torch.float16);parts=args.output/"feature_parts";parts.mkdir()
        for begin in range(0,2808,64):
            guard(args,identity);end=min(begin+64,2808)
            for start in range(begin,end):
                selected=rows[start:start+1];inputs,_=prepare_batch(selected,model)
                gpu_inputs=transfer(inputs)
                x[start:start+len(selected)]=features(model,gpu_inputs).cpu().half()
                del inputs,gpu_inputs
            guard(args,identity)
            q.atomic_save(parts/f"rows_{begin:05d}_{end:05d}.pt",dict(features=x[begin:end].clone(),
                sample_ids=[r["sample_id"] for r in rows[begin:end]],identity=identity))
            frozen.write_json(args.output/"status.json",dict(status="extracting",videos=end,total=2808,updated_at_unix=time.time()))
            print(f"[features] {end}/2808",flush=True)
        payload=dict(features=x,targets=torch.tensor([r["temporal"] for r in rows]),splits=[r["split"] for r in rows],
            sample_ids=[r["sample_id"] for r in rows],initial_native_rows=initial,quality_token_ids=ids,identity=identity)
        q.atomic_save(args.output/"features.pt",payload)
    elif args.phase=="evaluate":
        import numpy as np
        receipt=json.loads((args.checkpoint.parent/"training_receipt.json").read_text())
        for name,sha in receipt["artifacts"].items():
            if frozen.sha256(args.checkpoint.parent/name)!=sha:raise ValueError("Training receipt mismatch: "+name)
        for key in ("source_sha256","model_sha256","manifest_sha256","runtime_sha256"):
            if receipt["identity"][key]!=identity[key]:raise ValueError("Train/eval identity differs: "+key)
        pt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        if pt["quality_token_ids"]!=ids:raise ValueError("Native quality token ids changed")
        head=five_native_rows(pt["state_dict"]["weight"]).eval().requires_grad_(False).cuda()
        test=[r for r in rows if r["split"]=="test"];pred=[];audits=[]
        with (args.output/"predictions.csv").open("x",newline="") as f:
            w=csv.writer(f);w.writerow(["sample_index","sample_id","target_mos","prediction","frame_positions"])
            for i,row in enumerate(test):
                if i%25==0:audits.append(dict(at=i,processes=guard(args,identity)))
                inputs,pos=prepare_batch([row],model);gpu_inputs=transfer(inputs)
                with torch.inference_mode():v=float((head(features(model,gpu_inputs)).softmax(-1)@pt["level_scores"].cuda()).cpu()[0])
                pred.append(v);w.writerow([i,row["sample_id"],row["temporal"],v,json.dumps(pos[0])]);f.flush()
                del inputs,gpu_inputs
                if i%25==0 or i==557:
                    frozen.write_json(args.output/"status.json",dict(status="evaluating",batch=1,test_videos=i+1,updated_at_unix=time.time()))
                    print(f"[Small B1] {i+1}/558",flush=True)
        frozen.write_json(args.output/"report.json",dict(status="complete",test_videos=558,batch_size_videos=1,
            performance=core.metrics(np.array([r["temporal"] for r in test]),np.array(pred)),
            checkpoint_sha256=frozen.sha256(args.checkpoint),accuracy_only=True,formal_latency_energy_measured=False,
            process_audits=audits,selection=c["selection"],identity=identity))
    if not torch.equal(initial,model.language.lm_head.weight.detach()[ids].float().cpu()) or any(
        p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError("Frozen backbone/native rows changed")
    if args.phase=="train":
        del model;torch.cuda.empty_cache()
        train_rows(payload,initial,ids,args,c,identity)
    if verify_model_assets(args,c,"after_GPU")!=identity["model_sha256"]:raise RuntimeError("Official model files changed")
    if args.phase=="train":
        artifacts={str(p.relative_to(args.output)):frozen.sha256(p) for p in args.output.rglob("*") if p.is_file() and p.name!="status.json"}
        frozen.write_json(args.output/"training_receipt.json",dict(status="complete",identity=identity,artifacts=artifacts))


def suite(args,identity):
    deadline=min(time.time()+10800, args.deadline_unix) if args.deadline_unix is not None else time.time()+10800
    if deadline <= time.time():
        raise TimeoutError("Original suite deadline already passed")
    state=dict(status="waiting_for_assets",pid=os.getpid(),completed=[],deadline_unix=deadline,
        source_sha256=identity["source_sha256"],inference_batches=[1],accuracy_only=True)
    try:
        if args.prepare_run is None:raise ValueError("Existing prepare-run required")
        while True:
            if sources()!=identity["source_sha256"]:raise RuntimeError("Source changed while waiting")
            f=args.prepare_run/"status.json";d=json.loads(f.read_text()) if f.exists() else {}
            if d.get("status")=="complete":break
            if d.get("status")=="failed":raise RuntimeError("Preparation failed: "+str(d.get("error")))
            if time.time()>deadline-7200:raise TimeoutError("One-hour asset wait limit")
            state.update(asset_progress=d,updated_at_unix=time.time())
            frozen.write_json(args.output/"suite_status.json",state);time.sleep(15)
        while time.time()<args.gpu_not_before_unix:
            if time.time()>=deadline:raise TimeoutError("Total budget reached while yielding GPU")
            state.update(status="yielding_for_utilization",updated_at_unix=time.time())
            frozen.write_json(args.output/"suite_status.json",state)
            time.sleep(min(15,args.gpu_not_before_unix-time.time()))
        if args.allocator_recovery:
            # User-authorized single recovery: CPU-only wait, never reserve GPU memory.
            while True:
                if sources()!=identity["source_sha256"]:raise RuntimeError("Source changed during recovery wait")
                if time.time()>=deadline:raise TimeoutError("Original total deadline during memory wait")
                free=float(subprocess.check_output(["nvidia-smi",f"--id={frozen.A100_UUID}",
                    "--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
                if free>=37000:break
                state.update(status="waiting_for_memory_without_reservation",free_memory_mib=free,
                    minimum_free_memory_mib=37000,updated_at_unix=time.time())
                frozen.write_json(args.output/"suite_status.json",state)
                time.sleep(min(15,max(0,deadline-time.time())))
        # Include two full32GB official SHA passes, not just the short GPU probe.
        for phase,name,bound in (("cpu-preflight","cpu_preflight",180),("gpu-smoke","gpu_smoke",600),
                                  ("train","train",7200),("evaluate","a100_batch01",1800)):
            if sources()!=identity["source_sha256"]:raise RuntimeError("Source changed during suite")
            remaining=int(deadline-time.time())
            if remaining<1:raise TimeoutError("Three-hour total limit")
            state.update(status="running",phase=name,updated_at_unix=time.time())
            frozen.write_json(args.output/"suite_status.json",state)
            cmd=["timeout","--signal=TERM","--kill-after=30s",str(min(bound,remaining)),sys.executable,
                "-u","-m",__package__+".deepseek_vl2_small_temporal_baseline","--phase",phase,"--model",str(args.model),
                "--manifest",str(args.manifest),"--output",str(args.output/name),"--accuracy-only-concurrent"]
            if phase=="evaluate":cmd += ["--checkpoint",str(args.output/"train/best_checkpoint.pt")]
            if args.allocator_recovery:cmd += ["--allocator-recovery"]
            frozen.write_json(args.output/f"{name}.command.json",cmd)
            with (args.output/f"{name}.log").open("x") as log:
                result=subprocess.run(cmd,cwd=REPO,env={**os.environ,"HF_HUB_OFFLINE":"1",
                    "CUDA_VISIBLE_DEVICES":"" if phase=="cpu-preflight" else frozen.A100_UUID},
                    stdout=log,stderr=subprocess.STDOUT)
            f=args.output/name/"status.json";d=json.loads(f.read_text()) if f.exists() else {}
            if result.returncode or d.get("status")!="complete":
                raise RuntimeError(f"{phase} exit{result.returncode}: {d}; no automatic retry")
            state["completed"].append(name)
        state.update(status="complete",phase=None,updated_at_unix=time.time())
    except BaseException as e:
        state.update(status="failed",error=repr(e),updated_at_unix=time.time());raise
    finally:frozen.write_json(args.output/"suite_status.json",state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("cpu-preflight","gpu-smoke","train","evaluate","suite","prepare"),required=True)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--checkpoint",type=Path)
    p.add_argument("--prepare-run",type=Path)
    p.add_argument("--deadline-unix",type=float,help="Absolute original suite deadline; never extend a recovery")
    p.add_argument("--accuracy-only-concurrent",action="store_true")
    p.add_argument("--gpu-not-before-unix",type=float,default=0.)
    p.add_argument("--allocator-recovery",action="store_true",
        help="One user-authorized recovery: expandable allocator only; native graph and precision unchanged")
    args=p.parse_args();args.gpu_uuid=frozen.A100_UUID;c=config()
    if args.allocator_recovery:
        if args.phase=="prepare":p.error("Recovery must not download assets again")
        if args.deadline_unix is None and args.phase=="suite":p.error("Recovery requires original absolute deadline")
        # Set before the first Torch import/CUDA context; only allocation policy changes.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
    args.model=args.model.resolve();args.output=args.output.resolve();args.manifest=args.manifest.resolve()
    if frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:raise ValueError("Original manifest differs")
    if args.phase=="evaluate" and args.checkpoint is None:p.error("Evaluation needs fixed best checkpoint")
    shared.new_output(args.output);args.output.mkdir(parents=True)
    identity=dict(source_sha256=sources(),manifest_sha256=frozen.MANIFEST_SHA,contract=c,pid=os.getpid(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        command=sys.argv,created_at_unix=time.time(),accuracy_only=True,gpu_uuid=frozen.A100_UUID)
    identity.update(allocator_recovery=args.allocator_recovery,
        pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    frozen.write_json(args.output/"run_identity.json",identity)
    try:
        if args.phase not in ("suite","prepare"):
            identity["runtime_sha256"] = runtime_sources()
            frozen.write_json(args.output/"run_identity.json",identity)
        if args.phase not in ("cpu-preflight","suite","prepare"):
            identity["process_audit_before"]=guard(args,identity)
            identity["model_sha256"]=verify_model_assets(args,c,"before_GPU")
            import torch,transformers
            identity.update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda)
            frozen.write_json(args.output/"run_identity.json",identity)
        if args.phase=="suite":suite(args,identity)
        elif args.phase=="prepare":prepare_assets(args,c,identity)
        else:run(args,c,identity)
        if args.phase not in ("cpu-preflight","suite","prepare"):guard(args,identity)
        frozen.write_json(args.output/"status.json",dict(status="complete",updated_at_unix=time.time()))
    except BaseException as e:
        frozen.write_json(args.output/"status.json",dict(status="failed",error=repr(e),updated_at_unix=time.time()))
        raise


if __name__=="__main__":main()
