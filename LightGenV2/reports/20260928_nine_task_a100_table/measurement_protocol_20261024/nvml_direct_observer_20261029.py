"""CPU-only direct NVML acquisition, no nvidia-smi stdout buffering stage.

New evidence only: never retimes or repairs a previous raw sample. Each record
keeps the real API-call bracket and power-return receipt timestamp. Requested
50ms polling, the original numeric validation, and original250ms integration
support gate are retained. No CUDA imports, synthetic load, missing-row fill,
catch-up timestamps, interpolation or acquisition-thread filesystem writes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(REPO))
from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import Sample,parse_telemetry_line,save_samples
from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler

GPU='GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
POWER_LIMIT_MW=250000  # laboratory default; explicit rental profile must pin400000
VERSION='direct_NVML_actual_API_receipt_no_stdout_hop_v1'

def validate_device_identity(actual,power_limit_mw):
    if actual!=GPU or power_limit_mw!=POWER_LIMIT_MW:
        raise ValueError('Exact physical UUID and configured power cap required, never changed')

def acquire(nvml,handle,phase,clock=time.monotonic,unix=time.time):
    start=clock();mw=nvml.nvmlDeviceGetPowerUsage(handle);power_receipt=clock();power_unix=unix()
    util=nvml.nvmlDeviceGetUtilizationRates(handle);memory=nvml.nvmlDeviceGetMemoryInfo(handle);end=clock()
    if not start<=power_receipt<=end or end-start>.25:raise ValueError('NVML query bracket exceeds original250ms bound')
    values=(float(mw)/1000.,float(util.gpu),float(memory.used)/(1024**2))
    # The unchanged rejection policy, without a CLI formatted/rounded row.
    parse_telemetry_line(','.join(repr(x) for x in values))
    sample=Sample(power_unix,power_receipt,phase,*values)
    return sample,dict(query_start_monotonic_s=start,power_API_return_monotonic_s=power_receipt,
        query_end_monotonic_s=end,power_milliwatts=mw,utilization_gpu_raw=util.gpu,
        memory_used_bytes_raw=memory.used,phase=phase,valid_numeric_row=True,
        sample=asdict(sample),timestamp_source='actual power API return, not scheduled index or delayed reader')

class Sampler:
    def __init__(self,gpu,raw):
        self.gpu=gpu;self.raw=raw;self.phase='unlabelled';self.samples=[];self.records=[]
        self.error=None;self.stop_event=threading.Event();self.saved=False

    def start(self):
        if self.gpu!=GPU or self.raw.exists():raise ValueError('New exact approved GPU raw acquisition required')
        import pynvml
        self.nvml=pynvml;pynvml.nvmlInit()
        self.handle=pynvml.nvmlDeviceGetHandleByUUID(self.gpu)
        actual=pynvml.nvmlDeviceGetUUID(self.handle)
        if isinstance(actual,bytes):actual=actual.decode()
        power_limit_mw=pynvml.nvmlDeviceGetPowerManagementLimit(self.handle)
        validate_device_identity(actual,power_limit_mw)
        self.identity=dict(version=VERSION,pid=os.getpid(),gpu_uuid=actual,
            power_limit_milliwatts=power_limit_mw,expected_power_limit_milliwatts=POWER_LIMIT_MW,
            requested_poll_interval_s=.05,query_functions=['nvmlDeviceGetPowerUsage','nvmlDeviceGetUtilizationRates','nvmlDeviceGetMemoryInfo'],
            timestamp_origin='power API return in isolated observer process',API_bracket_maximum_s=.25,
            raw_IO_in_acquisition_thread=False,CUDA_context_created=False,
            driver=str(pynvml.nvmlSystemGetDriverVersion()),nvml_version=str(pynvml.nvmlSystemGetNVMLVersion()),
            observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        def read():
            try:
                while not self.stop_event.is_set():
                    sample,record=acquire(self.nvml,self.handle,self.phase)
                    if self.samples and sample.host_monotonic_s<=self.samples[-1].host_monotonic_s:
                        raise ValueError('Actual acquisition timestamps must strictly increase')
                    self.samples.append(sample);self.records.append(record)
                    # No backfilled/catch-up rows if the process was delayed.
                    self.stop_event.wait(.05)
            except BaseException as error:
                self.records.append(dict(error=repr(error),failed_at_monotonic_s=time.monotonic(),valid_numeric_row=False))
                self.error=error
        self.thread=threading.Thread(target=read,daemon=True);self.thread.start()

    def check(self):
        if self.error:raise RuntimeError('Direct NVML acquisition failed: '+repr(self.error)) from self.error

    def stop(self,raise_on_error=True):
        self.stop_event.set();self.thread.join(3)
        if self.thread.is_alive():self.error=RuntimeError('Direct NVML reader failed to exit')
        if not self.saved:
            with self.raw.open('x',encoding='utf8') as f:
                for row in self.records:f.write(json.dumps(row,ensure_ascii=True)+'\n')
            self.saved=True
        self.nvml.nvmlShutdown()
        if raise_on_error:self.check()

def serve(gpu,raw):
    sampler=Sampler(gpu,raw);stopped=False;index=0;started=False
    def emit(value):print(json.dumps(value,ensure_ascii=True),flush=True)
    def flush():
        nonlocal index
        while index<len(sampler.samples):emit(dict(sample=asdict(sampler.samples[index])));index+=1
    def terminate(signum,_):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    try:
        sampler.start();started=True
        with (raw.parent/'telemetry_observer_identity.json').open('x') as f:json.dump(sampler.identity,f,indent=2)
        emit(dict(ready=sampler.identity))
        while True:
            sampler.check();flush()
            if not select.select([sys.stdin],[],[],.01)[0]:continue
            line=sys.stdin.readline()
            if not line:raise RuntimeError('Observer channel closed before explicit stop')
            command=json.loads(line);op=command['op']
            if op=='phase':sampler.phase=command['phase']
            elif op=='stop':sampler.stop();stopped=True
            elif op!='barrier':raise ValueError('Unknown original IPC command')
            flush();emit(dict(ack=command['id'],observer_monotonic_s=time.monotonic()))
            if stopped:return
    except BaseException as error:emit(dict(error=repr(error)));raise
    finally:
        if started and not stopped:sampler.stop(raise_on_error=False)

def observer_command(gpu,path):
    return [sys.executable,'-u',str(Path(__file__).resolve()),'--observer','--gpu',gpu,'--raw',str(path)]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--observer',action='store_true')
    p.add_argument('--gpu',default=GPU,choices=(GPU,));p.add_argument('--raw',type=Path)
    p.add_argument('--diagnostic',type=Path);p.add_argument('--seconds',type=float,default=30.)
    args=p.parse_args()
    if args.observer:return serve(args.gpu,args.raw)
    if args.diagnostic is None or not 0<args.seconds<=60:raise ValueError('Bounded CPU-only diagnostic required')
    args.diagnostic.mkdir(exist_ok=False);raw=args.diagnostic/'telemetry_stdout.jsonl'
    sampler=ProcessTelemetrySampler(args.gpu,raw,command=observer_command(args.gpu,raw))
    try:sampler.start();sampler.set_phase('diagnostic_CPU_NVML_no_model_no_CUDA');time.sleep(args.seconds)
    finally:rows=sampler.stop()
    save_samples(args.diagnostic/'telemetry.csv',rows)
    gaps=[b.host_monotonic_s-a.host_monotonic_s for a,b in zip(rows,rows[1:])]
    result=dict(status='complete',pid=os.getpid(),version=VERSION,samples=len(rows),max_gap_s=max(gaps),
        minimum_gap_s=min(gaps),original_support_bound_s=.25,qualified=max(gaps)<=.25,
        formal_model_measurement=False,GPU_model_started=False,CUDA_imported='torch' in sys.modules)
    with (args.diagnostic/'status.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
