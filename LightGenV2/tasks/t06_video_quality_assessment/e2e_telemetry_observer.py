"""CPU-only E2E observer: keep raw rows in memory until acquisition stops.

The nvidia-smi query, 50-ms requested cadence, receipt clocks and numeric checks
are unchanged. Unlike the historical observer, its acquisition reader performs
no shared-filesystem writes or flushes. No rows are retimed, interpolated or
dropped. Raw evidence is written after the final measured output.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import time
from .batch_telemetry import Sample,TelemetrySampler,parse_telemetry_line,save_samples
from .nvml_observer_process import ProcessTelemetrySampler

VERSION='independent_process_memory_raw_v1'


class MemoryRawSampler(TelemetrySampler):
    def __init__(self,gpu_uuid,raw_path):
        super().__init__(gpu_uuid,raw_path)
        self.raw_records=[];self.raw_saved=False

    def start(self):
        path=self.raw_path
        if path.exists():raise FileExistsError('Preserve original raw evidence')
        self.raw_path=None
        try:super().start()
        finally:self.raw_path=path

    def _read(self):
        try:
            for line in self.process.stdout:
                record=dict(host_time_unix=time.time(),host_monotonic_s=time.monotonic(),
                    phase=self.phase,raw_line=line.rstrip('\r\n'))
                try:
                    watts,util,memory=parse_telemetry_line(line)
                    with self.lock:phase=self.phase
                    self.samples.append(Sample(time.time(),time.monotonic(),phase,watts,util,memory))
                    record['valid_numeric_row']=True
                except BaseException as error:
                    record.update(valid_numeric_row=False,error=repr(error));raise
                finally:self.raw_records.append(record)
            if not self.stopping:self.error=RuntimeError('Telemetry subprocess stdout ended')
        except BaseException as error:self.error=error

    def stop(self,raise_on_error=True):
        # Stop/join the acquisition reader BEFORE any raw-file serialization.
        rows=super().stop(raise_on_error=False)
        if not self.raw_saved:
            try:
                with self.raw_path.open('x',encoding='utf-8') as stream:
                    for record in self.raw_records:stream.write(json.dumps(record,ensure_ascii=True)+'\n')
                self.raw_saved=True
            except BaseException as error:self.error=error
        if raise_on_error:self.check()
        return rows


def serve(gpu,raw_path):
    sampler=MemoryRawSampler(gpu,raw_path);index=0;stopped=False
    def terminate(signum,_):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    def emit(value):print(json.dumps(value),flush=True)
    def flush():
        nonlocal index
        while index<len(sampler.samples):
            emit(dict(sample=asdict(sampler.samples[index])));index+=1
    try:
        sampler.start()
        identity=dict(version=VERSION,pid=os.getpid(),gpu_uuid=gpu,start_unix=time.time(),
            timestamp_origin='observer host receipt; never reconstructed from sample indices',
            cuda_context_created=False,query=sampler.process.args,query_pid=sampler.process.pid,
            raw_file_IO_in_acquisition_reader=False,raw_file_written_after_stop=True,
            observer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (raw_path.parent/'telemetry_observer_identity.json').open('x').write(json.dumps(identity,indent=2))
        emit(dict(ready=identity))
        while True:
            sampler.check();flush()
            if not select.select([__import__('sys').stdin],[],[],.01)[0]:continue
            line=__import__('sys').stdin.readline()
            if not line:raise RuntimeError('Observer control channel closed without stop')
            command=json.loads(line);op=command['op']
            if op=='phase':sampler.set_phase(command['phase'])
            elif op=='stop':sampler.stop();stopped=True
            elif op!='barrier':raise ValueError('Unknown observer command')
            flush();emit(dict(ack=command['id'],observer_monotonic_s=time.monotonic()))
            if stopped:return
    except BaseException as error:emit(dict(error=repr(error)));raise
    finally:
        if not stopped:sampler.stop(raise_on_error=False)


def observer_command(gpu,path):
    import sys
    return [sys.executable,'-u','-m',__package__+'.e2e_telemetry_observer','--observer',
        '--gpu',gpu,'--raw',str(path)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observer',action='store_true');p.add_argument('--gpu',required=True)
    p.add_argument('--raw',type=Path);p.add_argument('--diagnostic',type=Path)
    p.add_argument('--seconds',type=float,default=60.)
    args=p.parse_args()
    if args.observer:return serve(args.gpu,args.raw)
    if not args.diagnostic or not 0<args.seconds<=180:raise ValueError('Bounded CPU-only diagnostic')
    args.diagnostic.mkdir()
    raw=args.diagnostic/'telemetry_stdout.jsonl'
    sampler=ProcessTelemetrySampler(args.gpu,raw,command=observer_command(args.gpu,raw))
    try:
        sampler.start();sampler.set_phase('CPU_only_diagnostic_not_formal');time.sleep(args.seconds)
    finally:rows=sampler.stop()
    save_samples(args.diagnostic/'telemetry.csv',rows)
    gaps=[b.host_monotonic_s-a.host_monotonic_s for a,b in zip(rows,rows[1:])]
    result=dict(status='qualified' if gaps and max(gaps)<=.25 else 'unqualified',version=VERSION,
        samples=len(rows),max_gap_s=max(gaps) if gaps else None,threshold_s=.25,
        formal_cost=False,gpu_execution=False,no_retiming_or_imputation=True)
    (args.diagnostic/'report.json').open('x').write(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
