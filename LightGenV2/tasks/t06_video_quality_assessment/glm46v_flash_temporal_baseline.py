"""GLM-4.6V-Flash native video / five-row quality head; one B1 accuracy trial.

All official dependency files stay byte-identical; no generation or thinking.
Original data, output-row initialization, loss, optimizer and budget are retained.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from . import internvl35_temporal_baseline as previous

shared, frozen, TASK, REPO = previous.shared, previous.frozen, previous.TASK, previous.REPO
CONFIG = TASK / "configs/baselines/glm46v_flash_temporal_4f.json"


def config():
    c = json.loads(CONFIG.read_text())
    for k, v in dict(model_id="zai-org/GLM-4.6V-Flash", hidden_width=4096, frames=4,
        frame_fractions=[.1, .37, .63, .9], center_crop_short_side_fraction=.65,
        decode_resize=448, quality_output_trainable_parameters=20480,
        backbone_trainable_parameters=0, epochs=50, training_batch_size=512,
        feature_batch_size=2, learning_rate=.001, weight_decay=0., seed=42,
        loss="cross_entropy_only", scheduler="cosine", train_samples=2250,
        test_samples=558, inference_batches=[1], thinking=False,
        manifest_sha256=frozen.MANIFEST_SHA).items():
        if c.get(k) != v:
            raise ValueError("Original protocol changed: " + k)
    return c


def sources():
    return {**previous.sources(), str(Path(__file__).relative_to(REPO)): frozen.sha256(Path(__file__)),
            str(CONFIG.relative_to(REPO)): frozen.sha256(CONFIG)}


def guard(args, identity):
    if sources() != identity["source_sha256"]:
        raise RuntimeError("Source changed during run; preserve evidence")
    records = previous.gpu_guard(args)
    raw = subprocess.check_output(["nvidia-smi", f"--id={frozen.A100_UUID}",
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    if not any(r["pid"] == str(os.getpid()) for r in records) and float(raw.strip()) < 32000:
        raise RuntimeError("GLM initial admission needs 32000 MiB free, no reservation or eviction")
    return records


def runtime_sources():
    import inspect
    from transformers import Glm4vForConditionalGeneration, Glm46VProcessor
    from transformers.models.glm46v.video_processing_glm46v import Glm46VVideoProcessor
    from transformers.models.glm4v.configuration_glm4v import Glm4vConfig
    return {str(cls.__name__): frozen.sha256(Path(inspect.getfile(cls)))
            for cls in (Glm4vForConditionalGeneration, Glm46VProcessor, Glm46VVideoProcessor, Glm4vConfig)}


def native_model(path, *, meta=False):
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor, Glm4vForConditionalGeneration
    if transformers.__version__ != "5.0.0":
        raise RuntimeError("Dedicated GLM transformers5.0.0 required; do not change shared environment")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    ids = [processor.tokenizer.encode(w, add_special_tokens=False) for w in shared.WORDS]
    if any(len(i) != 1 for i in ids) or len({i[0] for i in ids}) != 5:
        raise RuntimeError(f"Quality words must be five native single tokens: {ids}")
    cfg = AutoConfig.from_pretrained(path, local_files_only=True)
    if cfg.text_config.hidden_size != 4096:
        raise RuntimeError("Native4096 width required")
    if meta:
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = Glm4vForConditionalGeneration(cfg)
    else:
        model = Glm4vForConditionalGeneration.from_pretrained(path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = model.eval().requires_grad_(False)
    model._quality_processor = processor
    model._quality_prompt = processor.apply_chat_template([{"role":"user","content":[
        {"type":"video"},{"type":"text","text":shared.PROMPT}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "/nothink" not in model._quality_prompt or "<think></think>" not in model._quality_prompt:
        raise RuntimeError("Official non-thinking prompt required")
    return model, [i[0] for i in ids]


def prepare_batch(rows, model):
    import numpy as np
    from . import quality_token_common as core
    decoded = [core.decode_random_seek(Path(r["video_path"]), core.FRAME_FRACTIONS[4], 448) for r in rows]
    if any(len(d[0]) != 4 for d in decoded):
        raise RuntimeError("Exactly four decoded frames required")
    inputs = model._quality_processor(text=[model._quality_prompt]*len(rows),
        videos=[np.stack([np.asarray(f) for f in d[0]]) for d in decoded],
        padding=True, return_tensors="pt",
        # GLM5 processor only honors these controls inside videos_kwargs. Frames
        # are already448 from the original decoder: no extra resize or sampling.
        videos_kwargs={"do_sample_frames":False,"do_resize":False,
                       "video_metadata":[d[1] for d in decoded]})
    inputs.pop("token_type_ids", None)
    grid = inputs["video_grid_thw"]
    if len(grid) != len(rows) or not bool(grid[:, 1:].eq(32).all()) or not bool(grid[:, 0].eq(2).all()):
        raise RuntimeError(f"Native448/14 spatial patch geometry differs: {grid.tolist()}")
    counts = inputs["input_ids"].eq(model.config.image_token_id).sum(1)
    if not bool(counts.eq(grid.prod(-1)//4).all()):
        raise RuntimeError("Native visual placeholder count does not match grid")
    return dict(inputs), [d[2] for d in decoded]


def features(model, inputs):
    import torch
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Native multimodal model is unchanged: visual encoder, merge, then language backbone.
        hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        mask = inputs["attention_mask"].bool()
        last = torch.arange(mask.shape[1], device=hidden.device).expand_as(mask).masked_fill(~mask, -1).amax(1)
        value = hidden[torch.arange(len(hidden), device=hidden.device), last].float()
    if value.shape != (len(hidden), 4096) or not bool(torch.isfinite(value).all()):
        raise RuntimeError("Invalid native final-token features")
    return value


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
    head = q.FiveNativeTokenRows(initial).cuda()
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
        cpu = q.FiveNativeTokenRows(state["weight"]).eval().requires_grad_(False)
        cm,cp,*_ = q.evaluate(cpu,x[te],y[te],labels[te],levels,torch.device("cpu"))
        q.atomic_save(args.output/f"{name}_checkpoint.pt",dict(architecture="frozen_glm46v_flash_five_untied_native_rows",
            state_dict=state, initial_native_rows=initial, quality_token_ids=ids, selected_epoch=epoch,
            boundaries=boundaries, level_scores=levels, metrics=gm, trainable_parameters=20480,
            all_backbone_parameters_frozen=True, selection=c["selection"], identity=identity))
        with (args.output/f"cpu_{name}_predictions.csv").open("x",newline="") as f:
            w=csv.writer(f);w.writerow(["sample_id","target_mos","prediction_mos"])
            w.writerows((payload["sample_ids"][i],float(y[i]),float(v)) for i,v in zip(te.tolist(),cp))
        results[name]=dict(epoch=epoch,gpu_metrics=gm,strict_cpu_metrics=cm,
            max_cpu_gpu_mos_difference=float((cp-gp).abs().max()))
    frozen.write_json(args.output/"cpu_verification.json",results)
    frozen.write_json(args.output/"training_report.json",dict(status="complete",contract=c,results=results,
        output_row_parameters=20480,backbone_updates=0,initial_rows_sha256=q.sha256_tensor(initial),identity=identity))


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
        for n in (1,2):
            inputs,pos=prepare_batch([r for r in rows if r["split"]=="train"][:n],model)
            results.append(dict(batch=n,shapes={k:list(v.shape) for k,v in inputs.items()},frame_positions=pos,
                video_grid_thw=inputs["video_grid_thw"].tolist(),
                visual_tokens_per_video=inputs["input_ids"].eq(model.config.image_token_id).sum(1).tolist()))
        frozen.write_json(args.output/"cpu_preflight_report.json",dict(status="passed",results=results,
            native_quality_token_ids=ids,thinking=False,gpu_initialized=torch.cuda.is_initialized(), prompt=model._quality_prompt,
            trainable_output_parameters=20480, runtime_sha256=runtime_sources()))
        return
    initial=model.lm_head.weight.detach()[ids].float().cpu().clone()
    if args.phase=="gpu-smoke":
        inputs,pos=prepare_batch([r for r in rows if r["split"]=="train"][:2],model)
        gpu_inputs=shared.transfer(inputs)
        tapped=features(model,gpu_inputs)
        # A fresh model must not reuse mRoPE deltas from a previous call in the parity probe.
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            native_logits=model(**gpu_inputs,logits_to_keep=1,use_cache=False,return_dict=True).logits[:,-1]
            tapped_logits=model.lm_head(tapped.to(model.lm_head.weight.dtype))
        native_difference=float((native_logits-tapped_logits).abs().max())
        if not torch.allclose(native_logits,tapped_logits,rtol=.01,atol=.125):
            raise RuntimeError(f"Native final-token feature tap parity failed: {native_difference}")
        x=tapped.cpu().half().clone()
        head=q.FiveNativeTokenRows(initial).cuda()
        # Isolated TRAIN-only optimization probe; no performance claim or model selection.
        ys=[r["temporal"] for r in rows if r["split"]=="train"]
        boundaries=torch.linspace(min(ys),max(ys),6)[1:-1]
        labels=torch.bucketize(torch.tensor(ys[:2]),boundaries).cuda()
        loss=torch.nn.functional.cross_entropy(head(x.float().cuda()),labels)
        loss.backward();torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=0.).step()
        if not bool(torch.isfinite(loss)) or torch.equal(head.weight.detach().cpu(),initial):
            raise RuntimeError("Five-row update smoke failed")
        q.atomic_save(args.output/"smoke_evidence.pt",dict(features=x,initial=initial,updated=head.weight.detach().cpu()))
        frozen.write_json(args.output/"gpu_smoke_report.json",dict(status="passed",trainable_parameters=20480,
            real_train_videos=2,probe_loss=float(loss.detach()),frame_positions=pos,not_a_performance_result=True,
            native_forward_last_logits_vs_feature_tap_max_abs=native_difference))
    elif args.phase=="train":
        x=torch.empty((2808,4096),dtype=torch.float16);parts=args.output/"feature_parts";parts.mkdir()
        for begin in range(0,2808,64):
            guard(args,identity);end=min(begin+64,2808)
            for start in range(begin,end,2):
                selected=rows[start:min(start+2,end)];inputs,_=prepare_batch(selected,model)
                gpu_inputs=shared.transfer(inputs)
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
        head=q.FiveNativeTokenRows(pt["state_dict"]["weight"]).eval().requires_grad_(False).cuda()
        test=[r for r in rows if r["split"]=="test"];pred=[];audits=[]
        with (args.output/"predictions.csv").open("x",newline="") as f:
            w=csv.writer(f);w.writerow(["sample_index","sample_id","target_mos","prediction","frame_positions"])
            for i,row in enumerate(test):
                if i%25==0:audits.append(dict(at=i,processes=guard(args,identity)))
                inputs,pos=prepare_batch([row],model);gpu_inputs=shared.transfer(inputs)
                with torch.inference_mode():v=float((head(features(model,gpu_inputs)).softmax(-1)@pt["level_scores"].cuda()).cpu()[0])
                pred.append(v);w.writerow([i,row["sample_id"],row["temporal"],v,json.dumps(pos[0])]);f.flush()
                del inputs,gpu_inputs
                if i%25==0 or i==557:
                    frozen.write_json(args.output/"status.json",dict(status="evaluating",batch=1,test_videos=i+1,updated_at_unix=time.time()))
                    print(f"[GLM B1] {i+1}/558",flush=True)
        frozen.write_json(args.output/"report.json",dict(status="complete",test_videos=558,batch_size_videos=1,
            performance=core.metrics(np.array([r["temporal"] for r in test]),np.array(pred)),
            checkpoint_sha256=frozen.sha256(args.checkpoint),accuracy_only=True,formal_latency_energy_measured=False,
            process_audits=audits,selection=c["selection"],identity=identity))
    if not torch.equal(initial,model.lm_head.weight.detach()[ids].float().cpu()) or any(
        p.requires_grad or p.grad is not None for p in model.parameters()):raise RuntimeError("Frozen backbone/native rows changed")
    if args.phase=="train":
        del model;torch.cuda.empty_cache()
        train_rows(payload,initial,ids,args,c,identity)
    if shared.model_files(args.model,c)!=identity["model_sha256"]:raise RuntimeError("Official model files changed")
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
        for phase,name,bound in (("cpu-preflight","cpu_preflight",180),("gpu-smoke","gpu_smoke",300),
                                  ("train","train",7200),("evaluate","a100_batch01",1800)):
            if sources()!=identity["source_sha256"]:raise RuntimeError("Source changed during suite")
            remaining=int(deadline-time.time())
            if remaining<1:raise TimeoutError("Three-hour total limit")
            state.update(status="running",phase=name,updated_at_unix=time.time())
            frozen.write_json(args.output/"suite_status.json",state)
            cmd=["timeout","--signal=TERM","--kill-after=30s",str(min(bound,remaining)),sys.executable,
                "-u","-m",__package__+".glm46v_flash_temporal_baseline","--phase",phase,"--model",str(args.model),
                "--manifest",str(args.manifest),"--output",str(args.output/name),"--accuracy-only-concurrent"]
            if phase=="evaluate":cmd += ["--checkpoint",str(args.output/"train/best_checkpoint.pt")]
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
    p.add_argument("--phase",choices=("cpu-preflight","gpu-smoke","train","evaluate","suite"),required=True)
    p.add_argument("--model",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--checkpoint",type=Path)
    p.add_argument("--prepare-run",type=Path)
    p.add_argument("--deadline-unix",type=float,help="Absolute original suite deadline; never extend a recovery")
    p.add_argument("--accuracy-only-concurrent",action="store_true")
    args=p.parse_args();args.gpu_uuid=frozen.A100_UUID;c=config()
    args.model=args.model.resolve();args.output=args.output.resolve();args.manifest=args.manifest.resolve()
    if frozen.sha256(args.manifest)!=frozen.MANIFEST_SHA:raise ValueError("Original manifest differs")
    if args.phase=="evaluate" and args.checkpoint is None:p.error("Evaluation needs fixed best checkpoint")
    shared.new_output(args.output);args.output.mkdir(parents=True)
    identity=dict(source_sha256=sources(),manifest_sha256=frozen.MANIFEST_SHA,contract=c,pid=os.getpid(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        command=sys.argv,created_at_unix=time.time(),accuracy_only=True,gpu_uuid=frozen.A100_UUID)
    frozen.write_json(args.output/"run_identity.json",identity)
    try:
        if args.phase not in ("suite",):
            identity["runtime_sha256"] = runtime_sources()
            frozen.write_json(args.output/"run_identity.json",identity)
        if args.phase not in ("cpu-preflight","suite"):
            identity["process_audit_before"]=guard(args,identity)
            identity["model_sha256"]=shared.model_files(args.model,c)
            import torch,transformers
            identity.update(torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda)
            frozen.write_json(args.output/"run_identity.json",identity)
        if args.phase=="suite":suite(args,identity)
        else:run(args,c,identity)
        if args.phase not in ("cpu-preflight","suite"):guard(args,identity)
        frozen.write_json(args.output/"status.json",dict(status="complete",updated_at_unix=time.time()))
    except BaseException as e:
        frozen.write_json(args.output/"status.json",dict(status="failed",error=repr(e),updated_at_unix=time.time()))
        raise


if __name__=="__main__":main()
