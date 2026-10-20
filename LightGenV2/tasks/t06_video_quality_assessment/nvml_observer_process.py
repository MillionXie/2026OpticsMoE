"""CPU-only NVML receiver isolated from the measured model's Python process.

The query, 50-ms cadence, numeric validation and raw rows are unchanged. Receipt
timestamps originate in the observer, never in the delayed model-side reader.
No CUDA context, fake load, interpolation or modification of model execution.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import select
import subprocess
import sys
import threading
import time
from .batch_telemetry import Sample,TelemetrySampler,save_samples

VERSION='independent_process_receipt_clock_v1'


def serve(gpu,raw_path):
    sampler=TelemetrySampler(gpu,raw_path);index=0;stopped=False
    def emit(value):print(json.dumps(value),flush=True)
    def flush():
        nonlocal index
        while index<len(sampler.samples):
            emit(dict(sample=asdict(sampler.samples[index])));index+=1
    try:
        sampler.start()
        identity=dict(version=VERSION,pid=__import__('os').getpid(),gpu_uuid=gpu,
                      start_unix=time.time(),timestamp_origin='observer host receipt',
                      cuda_context_created=False,query=sampler.process.args,
                      observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (raw_path.parent/'telemetry_observer_identity.json').open('x').write(json.dumps(identity,indent=2))
        emit(dict(ready=identity))
        while True:
            sampler.check();flush()
            if not select.select([sys.stdin],[],[],.01)[0]:continue
            line=sys.stdin.readline()
            if not line:raise RuntimeError('Observer control channel closed without stop')
            command=json.loads(line);op=command['op']
            if op=='phase':sampler.set_phase(command['phase'])
            elif op=='stop':sampler.stop();stopped=True
            elif op!='barrier':raise ValueError('Unknown observer command')
            flush();emit(dict(ack=command['id'],observer_monotonic_s=time.monotonic()))
            if stopped:return
    except BaseException as error:
        emit(dict(error=repr(error)));raise
    finally:
        if not stopped:sampler.stop(raise_on_error=False)


class ProcessTelemetrySampler:
    """Same measurement surface; IPC barriers drain original observer timestamps."""
    def __init__(self,gpu_uuid,raw_path,command=None):
        self.gpu_uuid=gpu_uuid;self.raw_path=Path(raw_path);self.command=command
        self.samples=[];self.error=None;self.process=None;self.serial=0
        self.events={};self.ready=threading.Event();self.stopping=False
    def start(self):
        cmd=self.command or [sys.executable,'-u','-m',__package__+'.nvml_observer_process','--observer','--gpu',self.gpu_uuid,'--raw',str(self.raw_path)]
        self.process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()
        if not self.ready.wait(10):raise RuntimeError('Observer did not become ready')
        self.check()
    def _read(self):
        try:
            for line in self.process.stdout:
                event=json.loads(line)
                if 'sample' in event:
                    row=Sample(**event['sample'])
                    if self.samples and row.host_monotonic_s<=self.samples[-1].host_monotonic_s:
                        raise RuntimeError('Observer timestamps must strictly increase')
                    self.samples.append(row)
                if 'ready' in event:self.identity=event['ready'];self.ready.set()
                if 'ack' in event:self.events[event['ack']].set()
                if 'error' in event:raise RuntimeError(event['error'])
            if not self.stopping:raise RuntimeError('Unexpected observer EOF')
        except BaseException as error:
            self.error=error;self.ready.set()
            for event in self.events.values():event.set()
    def check(self):
        if self.error:raise RuntimeError('Observer failed: '+repr(self.error)) from self.error
    def request(self,op,**payload):
        self.check();self.serial+=1;event=threading.Event();self.events[self.serial]=event
        self.process.stdin.write(json.dumps(dict(op=op,id=self.serial,**payload))+'\n');self.process.stdin.flush()
        if not event.wait(5):raise RuntimeError('Observer IPC barrier timed out')
        self.check();del self.events[self.serial]
    def set_phase(self,phase):self.request('phase',phase=phase or 'between_calls')
    def stop(self,raise_on_error=True):
        self.stopping=True
        try:
            if self.process is not None:
                if self.process.poll() is None and self.error is None:self.request('stop')
                if self.process.poll() is None:
                    try:self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=3)
                self.thread.join(timeout=3)
                if self.thread.is_alive():raise RuntimeError('Observer reader did not stop')
                if self.process.returncode!=0:raise RuntimeError(self.process.stderr.read()[-2000:])
        except BaseException as error:self.error=error
        if raise_on_error:self.check()
        return list(self.samples)


def main():
    p=argparse.ArgumentParser();p.add_argument('--observer',action='store_true');p.add_argument('--gpu',required=True)
    p.add_argument('--raw',type=Path);p.add_argument('--diagnostic',type=Path);p.add_argument('--seconds',type=float,default=60.)
    a=p.parse_args()
    if a.observer:return serve(a.gpu,a.raw)
    if not a.diagnostic or not 0<a.seconds<=180:raise ValueError('Bounded CPU-only diagnostic')
    a.diagnostic.mkdir();sampler=ProcessTelemetrySampler(a.gpu,a.diagnostic/'telemetry_stdout.jsonl')
    try:
        sampler.start();sampler.set_phase('diagnostic_not_formal');time.sleep(a.seconds)
    finally:rows=sampler.stop()
    save_samples(a.diagnostic/'telemetry.csv',rows)
    gaps=[b.host_monotonic_s-a.host_monotonic_s for a,b in zip(rows,rows[1:])]
    result=dict(version=VERSION,samples=len(rows),max_gap_s=max(gaps),threshold_s=.25,
                qualified=max(gaps)<=.25,formal_cost=False,gpu_execution=False)
    (a.diagnostic/'report.json').open('x').write(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
