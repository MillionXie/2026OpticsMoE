"""Frozen B2 follow-up on a newly allocated rental GPU; original runs immutable."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from . import rental_vlm_temporal as old


def sources():
    return {**old.sources(), str(Path(__file__).relative_to(old.REPO)): old.sha(Path(__file__))}


class ClockSampler(old.ClockSampler):
    def __init__(self, gpu_uuid): self.gpu_uuid = gpu_uuid
    def start(self):
        self.records=[]
        self.process=subprocess.Popen(["nvidia-smi",f"--id={self.gpu_uuid}","--query-gpu=clocks.sm,clocks.mem,temperature.gpu,pstate,power.limit","--format=csv,noheader,nounits","--loop-ms=200"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        def read():
            for line in self.process.stdout:
                fields=[v.strip() for v in line.split(",")]
                if len(fields)==5:self.records.append([time.time(),time.monotonic(),*fields])
        self.thread=threading.Thread(target=read,daemon=True);self.thread.start()


class Guard:
    def __init__(self, args, identity):
        self.args, self.identity = args, identity
        self.expected = None
        self.error = None
        self.stop = threading.Event()
        self.thread = None
        self.audit = []

    def check(self):
        if time.time() >= self.args.deadline:
            raise TimeoutError("User one-hour rental window reached")
        if sources() != self.identity["sources"]:
            raise RuntimeError("Bound source changed")
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], text=True)
        records = [r for r in csv.reader(raw.splitlines()) if len(r) == 3 and r[1].strip() == self.args.gpu_uuid]
        pids = {r[0].strip() for r in records}
        self.audit.append(dict(time=time.time(), processes=records))
        if pids != (self.expected or set()):
            raise RuntimeError("Unexpected GPU process; refuse overlapping measurement")

    def bind(self):
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
        records = [r for r in csv.reader(raw.splitlines()) if len(r) == 2 and r[1].strip() == self.args.gpu_uuid]
        if len(records) != 1:
            raise RuntimeError("Cannot bind sole CUDA process")
        self.expected = {records[0][0].strip()}
        self.check()

    def start(self):
        def monitor():
            while not self.stop.wait(1):
                try: self.check()
                except BaseException as error:
                    self.error = error
                    return
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()

    def fast(self):
        if self.error: raise RuntimeError("GPU/source monitor failed") from self.error
        if time.time() >= self.args.deadline: raise TimeoutError("Fixed rental deadline")


def prepare_batch(key, selected, processor, prompt):
    from . import deepseek_vl2_quality as tiny
    frames, positions = [], []
    for row in selected:
        images, where = tiny.decode_ordered_frames(Path(row["video_path"]), [.1,.37,.63,.9], 384 if key == "deepseek" else 448)
        frames.extend(images); positions.append(where)
    if key == "deepseek":
        prepared = [processor.process_one(conversations=prompt, images=frames[i:i+4], system_prompt="") for i in range(0,len(frames),4)]
        result = processor.batchify(prepared)
        values = {name: result[name] for name in ("input_ids","attention_mask","images","images_seq_mask","images_spatial_crop")}
        if list(values["images"].shape) != [len(selected),8,3,384,384] or not bool(values["images_spatial_crop"].eq(1).all()):
            raise RuntimeError("Four-image native tiling changed")
    else:
        values = dict(processor(images=frames, text=[prompt]*len(selected), return_tensors="pt", padding=True, truncation=False))
        if values["image_grid_hws"].tolist() != [[32,32]] * (4*len(selected)):
            raise RuntimeError("Kimi frame grids changed")
    if values["input_ids"].shape[0] != len(selected) or not bool(values["attention_mask"][:,-1].all()):
        raise RuntimeError("Real batch / last-valid-token identity changed")
    return values, positions


def hidden(key, model, language, inputs):
    import torch
    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.bfloat16):
        if key == "deepseek": embeds = model.prepare_inputs_embeds(**inputs)
        else:
            embeds = model.get_input_embeddings()(inputs["input_ids"])
            visual = model._extract_image_features(inputs["pixel_values"].to(model.vision_tower.dtype),inputs["image_grid_hws"])
            embeds = model._merge_with_image_features(embeds.to(visual[0].dtype),inputs["input_ids"],visual)
        value = language.model(inputs_embeds=embeds,attention_mask=inputs["attention_mask"],use_cache=False,return_dict=True).last_hidden_state[:,-1].float()
    if value.shape != (inputs["input_ids"].shape[0], old.WIDTHS[key]): raise RuntimeError("Native hidden shape changed")
    return value


def runtime(args, identity, guard):
    import torch
    path = Path("/data/models") / args.model
    train = old.RUN / args.model / "train"
    receipt = json.loads((train/"training_receipt.json").read_text())
    for name, digest in receipt["artifacts"].items():
        if old.sha(train/name) != digest: raise RuntimeError("Training receipt artifact changed: "+name)
    if receipt["identity"]["source_sha256"] != old.sources(): raise RuntimeError("Original training source changed")
    identity["assets"] = old.verify_assets(path)
    if identity["assets"] != receipt["identity"]["model_sha256"]: raise RuntimeError("Official model identity changed")
    processor, prompt, ids = old.processor(args.model,path)
    model, language, first = old.load_model(args.model,path)
    guard.bind()
    identity["runtime"] = old.runtime_identity(model)
    if identity["runtime"] != receipt["identity"]["runtime"]: raise RuntimeError("Original runtime changed")
    pt = torch.load(train/"best_checkpoint.pt",map_location="cpu",weights_only=False)
    if ids != pt["quality_token_ids"]: raise RuntimeError("Quality tokens changed")
    head = old.five_rows(pt["state_dict"]["weight"]).cuda().eval().requires_grad_(False)
    levels = pt["level_scores"].cuda()
    identity.update(checkpoint_sha256=old.sha(train/"best_checkpoint.pt"),training_receipt_sha256=old.sha(train/"training_receipt.json"))
    def compute(inputs):
        value = hidden(args.model,model,language,inputs)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            native = language.lm_head(value.to(language.lm_head.weight.dtype)).float()
        with torch.inference_mode(): mos = head(value).softmax(-1) @ levels
        return mos,native
    return model,processor,prompt,first,compute


def run(args, identity, guard):
    import torch
    import numpy as np
    from . import batch_telemetry as telemetry, quality_token_common as core
    from LightGenV2.common import baseline_measurement as measurement
    data = old.rows(); test = [r for r in data if r["split"] == "test"]
    model,p,prompt,first,compute = runtime(args,identity,guard)
    old.write(args.output/"identity.json",identity)
    guard.start()
    if args.phase == "utilization":
        # Real two-video TRAIN parity before the untouched three-repeat TEST load.
        cpu, _ = prepare_batch(args.model,[r for r in data if r["split"] == "train"][:2],p,prompt)
        inputs = old.transfer(cpu)
        mos,tapped = compute(inputs)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            native = model(**inputs,use_cache=False,return_dict=True).logits[:,-1].float()
        if not torch.isfinite(native).all() or not torch.allclose(native,tapped,rtol=.01,atol=.125): raise RuntimeError("Native B2 parity failed")
        old.write(args.output/"parity.json",dict(max_abs=float((native-tapped).abs().max()),rtol=.01,atol=.125,train_samples=2))
        del cpu,inputs,mos,tapped,native
        torch.cuda.empty_cache()
        cpu,positions = prepare_batch(args.model,test[:2],p,prompt); inputs = old.transfer(cpu)
        torch.cuda.synchronize(); reports=[]
        for repeat in range(1,4):
            folder=args.output/f"repeat{repeat:02d}"; folder.mkdir()
            sampler=telemetry.TelemetrySampler(args.gpu_uuid); clocks=ClockSampler(args.gpu_uuid); groups=[]
            sampler.start(); clocks.start()
            def phase(name,minimum):
                sampler.set_phase(name); start=time.monotonic()
                while time.monotonic()-start<minimum:
                    guard.fast(); tick=time.monotonic()
                    for _ in range(8): mos,raw=compute(inputs)
                    torch.cuda.synchronize();groups.append(dict(phase=name,start=tick,end=time.monotonic(),forwards=8))
                return start,time.monotonic(),mos
            try:
                phase("warmup",5); start,end,mos=phase("measured",30)
                samples=sampler.stop(); selected=[s for s in samples if start<=s.host_monotonic_s<end]
                report=telemetry.telemetry_report(selected,prefix="")
                report.update(start_monotonic_s=start,end_monotonic_s=end,duration_s=end-start,batch=2,sample_ids=[r["sample_id"] for r in test[:2]],positions=positions,final_mos=mos.cpu().tolist(),measured_forwards=sum(g["forwards"] for g in groups if g["phase"]=="measured"))
                old.write(folder/"report.json",report); reports.append(report)
            finally:
                if sampler.process is not None: samples=sampler.stop()
                telemetry.save_samples(folder/"telemetry.csv",samples); old.write(folder/"forward_groups.json",groups); clocks.stop(folder/"clock_telemetry.csv")
        total=sum(r["samples"] for r in reports)
        result=dict(repeats=reports,total_samples=total,pooled_mean_utilization_percent=sum(r["mean_gpu_utilization_percent"]*r["samples"] for r in reports)/total,costs_not_derived_from_resident_power=True)
    else:
        timer=measurement.FirstBlockTimer(first);sampler=telemetry.TelemetrySampler(args.gpu_uuid);clocks=ClockSampler(args.gpu_uuid)
        sampler.start();clocks.start();sampler.set_phase("idle");time.sleep(2);sampler.set_phase(None)
        predictions=[];timings=[]
        try:
            with (args.output/"predictions.csv").open("x",newline="") as ps,(args.output/"batch_timing.csv").open("x",newline="") as ts:
                pw,tw=csv.writer(ps),csv.writer(ts)
                pw.writerow(["sample_index","sample_id","target_mos","prediction","frame_positions"])
                tw.writerow(["batch_index","videos","in_cost_subset","synchronized_wall_ms","cuda_ms","preprocessing_audit_ms","h2d_audit_ms"])
                for begin in range(0,558,2):
                    guard.fast(); selected=test[begin:begin+2]
                    tick=time.perf_counter();cpu,positions=prepare_batch(args.model,selected,p,prompt);prep=1000*(time.perf_counter()-tick)
                    tick=time.perf_counter();inputs=old.transfer(cpu);torch.cuda.synchronize();h2d=1000*(time.perf_counter()-tick)
                    timer.reset();sampler.set_phase(f"active:{begin}" if begin<200 else None)
                    try: mos,raw=compute(inputs); timing=timer.finish()
                    finally: sampler.set_phase(None)
                    values=mos.cpu().tolist()
                    if not all(np.isfinite(values)): raise RuntimeError("Nonfinite MOS")
                    predictions.extend(values); record=[begin//2,len(selected),begin<200,timing["host_ms"],timing["cuda_ms"],prep,h2d];timings.append(record);tw.writerow(record)
                    for j,(row,value) in enumerate(zip(selected,values)): pw.writerow([begin+j,row["sample_id"],row["temporal"],value,json.dumps(positions[j])])
                    ps.flush();ts.flush();del inputs,cpu,mos,raw
                    if begin%50==0: old.write(args.output/"status.json",dict(status="evaluating",videos=begin+2,time=time.time()))
            samples=sampler.stop();telemetry.save_samples(args.output/"telemetry.csv",samples)
            cost=[r for r in timings if r[2]];power=measurement.power_report(samples,[r[3] for r in cost],power_limit_w=400)
            result=dict(performance=core.metrics(np.array([r["temporal"] for r in test]),np.array(predictions)),test_videos=558,cost_videos=200,cost=telemetry.workload_cost(cost,power["active_mean_w"]),power=power,first_batch_included=True,explicit_warmup_forwards=0)
        finally:
            if sampler.process is not None:telemetry.save_samples(args.output/"telemetry.csv",sampler.stop())
            clocks.stop(args.output/"clock_telemetry.csv");timer.close()
    if old.verify_assets(Path("/data/models")/args.model)!=identity["assets"]:raise RuntimeError("Official assets changed")
    guard.check();old.write(args.output/"report.json",dict(status="complete",identity=identity,batch=2,**result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",choices=("deepseek","kimi"),required=True)
    p.add_argument("--phase",choices=("utilization","evaluate"),required=True)
    p.add_argument("--gpu-uuid",required=True);p.add_argument("--deadline",required=True,type=float)
    args=p.parse_args()
    args.output=old.RUN/"batch02_followup_20261017"/args.model/args.phase
    if not old.REPO.is_relative_to("/data") or args.output.exists(): raise RuntimeError("New canonical /data output only")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=args.gpu_uuid:raise RuntimeError("Exact GPU UUID binding required")
    args.output.mkdir(parents=True)
    identity=dict(sources=sources(),model=args.model,batch=2,phase=args.phase,gpu_uuid=args.gpu_uuid,created=time.time(),git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),command=sys.argv,original_B1_untouched=True)
    guard=Guard(args,identity)
    try:
        guard.check()
        gpu=subprocess.check_output(["nvidia-smi",f"--id={args.gpu_uuid}","--query-gpu=name,memory.free,power.limit","--format=csv,noheader,nounits"],text=True)
        name,free,watts=next(csv.reader(gpu.splitlines()))
        if "A100-SXM4-80GB" not in name or float(free)<75000 or float(watts)!=400:raise RuntimeError("Expected free A10080 default400W")
        identity["gpu"]=gpu.strip()
        import torch
        torch.set_num_threads(8)
        run(args,identity,guard)
        old.write(args.output/"status.json",dict(status="complete",time=time.time()))
    except BaseException as error:
        old.write(args.output/"status.json",dict(status="failed",error=repr(error),time=time.time()));raise
    finally:
        guard.stop.set()
        if guard.thread:guard.thread.join(10)
        old.write(args.output/"process_audit.json",guard.audit)


if __name__ == "__main__": main()
