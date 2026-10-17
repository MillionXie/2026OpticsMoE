"""Exact historical native Qwen B1 cross-device control, never substitutes anchor."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from . import rental_vlm_temporal as old
from . import rental_vlm_batch_followup as follow
from .rental_vlm_assets import check

REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
MODEL="Qwen/Qwen3-VL-2B-Instruct"
HEAD_SHA="e102adc6a7b33335c8f49ae1cbeea3302a1a10017d282da8d6091a640276d0d2"
WEIGHT_SHA="7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
MODEL_DIR=Path("/data/models/qwen3vl2b-89644892")
OUT=old.RUN/"qwen_device_control_20261017"


def download(deadline):
    import requests
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    response=requests.get(f"https://hf-mirror.com/api/models/{MODEL}/revision/{REVISION}",params={"blobs":"true"},timeout=30)
    response.raise_for_status(); metadata=response.json()
    if metadata["sha"]!=REVISION:raise RuntimeError("Qwen revision changed")
    assets=[r for r in metadata["siblings"] if "/" not in r["rfilename"] and r["rfilename"].endswith((".json",".safetensors",".jinja",".txt",".model"))]
    old.write(MODEL_DIR/"official_metadata.json",dict(model=MODEL,revision=REVISION,assets=assets))
    for spec in assets:
        name=spec["rfilename"];target=MODEL_DIR/name
        if target.exists():
            if not check(target,spec):raise RuntimeError("Existing public asset mismatch")
            continue
        partial=MODEL_DIR/(name+".partial")
        if partial.exists():raise RuntimeError("Preserve partial; no automatic repeated download")
        size=spec["size"]; chunk=32*1024*1024
        def piece(bounds):
            start,end=bounds
            for attempt in range(3):
                if time.time()>=deadline:raise TimeoutError("Public preparation deadline")
                try:
                    r=requests.get(f"https://hf-mirror.com/{MODEL}/resolve/{REVISION}/{name}",headers={"Range":f"bytes={start}-{end}"},timeout=(15,30))
                    if r.status_code==429:
                        retry=r.headers.get("Retry-After","30");r.close();time.sleep(int(retry) if retry.isdigit() else 30);continue
                    r.raise_for_status()
                    if r.status_code==206 and r.headers.get("Content-Range")!=f"bytes {start}-{end}/{size}":raise RuntimeError("HTTP range mismatch")
                    if r.status_code not in (200,206) or (r.status_code==200 and (start!=0 or end+1!=size)):raise RuntimeError("Range ignored")
                    data=r.content;r.close()
                    if len(data)!=end-start+1:raise RuntimeError("Incomplete public chunk")
                    return data
                except requests.RequestException:
                    if attempt==2:raise
            raise RuntimeError("Public asset bounded retries exhausted")
        ranges=[(i,min(i+chunk,size)-1) for i in range(0,size,chunk)]
        with partial.open("xb") as stream,ThreadPoolExecutor(max_workers=4) as pool:
            for data in pool.map(piece,ranges):
                stream.write(data);stream.flush()
                old.write(MODEL_DIR/"download_status.json",dict(file=name,bytes=stream.tell(),total=size,time=time.time()))
        if not check(partial,spec):raise RuntimeError("Official Qwen blob/LFS SHA failed")
        partial.rename(target)
    if old.sha(MODEL_DIR/"model.safetensors")!=WEIGHT_SHA:raise RuntimeError("Historical Qwen backbone mismatch")
    old.write(MODEL_DIR/"asset_receipt.json",dict(status="complete",revision=REVISION,hashes={s["rfilename"]:old.sha(MODEL_DIR/s["rfilename"]) for s in assets}))


class Guard(follow.Guard):
    def check(self):
        super().check()
        if old.sha(Path(__file__))!=self.identity["entry_sha256"]:raise RuntimeError("Qwen control source changed")


def evaluate(args):
    import torch
    import numpy as np
    import transformers
    from . import evaluate_temporal_baseline_batches as original
    from . import quality_token_common as core,batch_telemetry as telemetry
    from LightGenV2.common import baseline_measurement as measurement
    args.output=OUT/"native_batch01_cost"
    args.output.mkdir(parents=True,exist_ok=False)
    identity=dict(sources=follow.sources(),entry_sha256=old.sha(Path(__file__)),gpu_uuid=args.gpu_uuid,model=MODEL,revision=REVISION,batch=1,checkpoint_sha256=HEAD_SHA,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),torch=torch.__version__,transformers=transformers.__version__,historical_anchor_not_substituted=True)
    guard=Guard(args,identity);sampler=None;timer=None;clocks=None
    try:
        guard.check();old.rows()
        receipt=json.loads((MODEL_DIR/"asset_receipt.json").read_text())
        for name,digest in receipt["hashes"].items():
            if old.sha(MODEL_DIR/name)!=digest:raise RuntimeError("Qwen official asset changed")
        if receipt["revision"]!=REVISION or old.sha(args.checkpoint)!=HEAD_SHA:raise RuntimeError("Not original frozen Qwen/PT")
        torch.set_num_threads(8)
        config=argparse.Namespace(baseline="qwen",model=MODEL_DIR,checkpoint=args.checkpoint,manifest=Path("/data/lgvq/manifest.csv"),image_size=448)
        model,head,rows,prepare,transfer,forward,timer,handles=original.load_runtime(config)
        guard.bind();guard.start()
        if any(p.requires_grad for module in (model,head) for p in module.parameters()):raise RuntimeError("Qwen must remain frozen")
        identity["actual_runtime"]=dict(attention=model.config._attn_implementation,parameters=sum(p.numel() for p in model.parameters()),gpu=subprocess.check_output(["nvidia-smi",f"--id={args.gpu_uuid}","--query-gpu=name,power.limit,memory.total","--format=csv,noheader"],text=True).strip())
        old.write(args.output/"identity.json",identity)
        sampler=telemetry.TelemetrySampler(args.gpu_uuid);clocks=follow.ClockSampler(args.gpu_uuid)
        sampler.start();clocks.start();sampler.set_phase("idle");time.sleep(2);sampler.set_phase(None)
        values=[];times=[]
        with (args.output/"predictions.csv").open("x",newline="") as ps,(args.output/"batch_timing.csv").open("x",newline="") as ts:
            pw,tw=csv.writer(ps),csv.writer(ts)
            pw.writerow(["sample_index","sample_id","target_mos","prediction","frame_positions"])
            tw.writerow(["batch_index","videos","in_cost_subset","synchronized_wall_ms","cuda_ms","preprocessing_audit_ms","h2d_audit_ms"])
            for i,row in enumerate(rows):
                guard.fast();tick=time.perf_counter();cpu,positions=prepare([row]);prep=1000*(time.perf_counter()-tick)
                tick=time.perf_counter();inputs=transfer(cpu);torch.cuda.synchronize();h2d=1000*(time.perf_counter()-tick)
                sampler.set_phase(f"active:{i}" if i<200 else None)
                try: prediction,wall,cuda=forward(inputs)
                finally:sampler.set_phase(None)
                values.extend(prediction);record=[i,1,i<200,wall,cuda,prep,h2d];times.append(record);tw.writerow(record)
                pw.writerow([i,row["sample_id"],row["temporal"],prediction[0],json.dumps(positions[0])]);ps.flush();ts.flush();del cpu,inputs
                if i%25==0:old.write(args.output/"status.json",dict(status="evaluating",videos=i+1,time=time.time()))
        samples=sampler.stop();telemetry.save_samples(args.output/"telemetry.csv",samples)
        power=measurement.power_report(samples,[r[3] for r in times if r[2]],power_limit_w=400)
        guard.check()
        for name,digest in receipt["hashes"].items():
            if old.sha(MODEL_DIR/name)!=digest:raise RuntimeError("Qwen model changed after inference")
        old.write(args.output/"report.json",dict(status="complete",identity=identity,performance=core.metrics(np.array([r["temporal"] for r in rows]),np.array(values)),test_videos=558,cost_videos=200,batch=1,power=power,cost=telemetry.workload_cost([r for r in times if r[2]],power["active_mean_w"]),explicit_warmup_forwards=0,first_batch_included=True))
        old.write(args.output/"status.json",dict(status="complete",time=time.time()))
    except BaseException as error:
        old.write(args.output/"status.json",dict(status="failed",error=repr(error),time=time.time()));raise
    finally:
        if sampler is not None and sampler.process is not None:telemetry.save_samples(args.output/"telemetry.csv",sampler.stop())
        if clocks is not None:clocks.stop(args.output/"clock_telemetry.csv")
        if timer is not None:timer.close()
        guard.stop.set()
        if guard.thread:guard.thread.join(10)
        old.write(args.output/"process_audit.json",guard.audit)


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=("prepare","evaluate"),required=True)
    p.add_argument("--deadline",type=float,required=True);p.add_argument("--gpu-uuid")
    p.add_argument("--checkpoint",type=Path,default=OUT/"original_best_checkpoint.pt")
    args=p.parse_args()
    if not old.REPO.is_relative_to("/data"):raise RuntimeError("Rental /data only")
    if args.phase=="prepare":download(args.deadline)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=args.gpu_uuid:raise RuntimeError("Exact GPU binding required")
        evaluate(args)


if __name__=="__main__":main()
