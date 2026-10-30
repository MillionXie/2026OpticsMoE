"""Single-request optical billing model with real kernel attribution and raw traces.

This does not measure a physical optical computer. Each necessary electronic
kernel is counted once, in serial request scope. CPU launch/waits are excluded;
CPU preprocessing/output WORK uses process CPU time (all process threads),
recording caller-thread and wall separately. This is a conservative work-cost
model, not a wall latency or a verified deployment DMA/SDK measurement.
"""
import collections
import gzip
import hashlib
import json
import time
from pathlib import Path
import torch

PERIOD_MS=(1/1886+.000300+.0000307)*1000
EXCLUDED={'T12/FFT_simulation','T12/electronic_residual','T12/pure_fusion'}
PURE_LAYOUT={'aten::view','aten::_unsafe_view','aten::reshape','aten::permute',
    'aten::transpose','aten::as_strided','aten::squeeze','aten::unsqueeze','aten::expand'}


def billing_group(name,parents):
    for tag in ['T12/FFT_simulation','T12/electronic_residual','T12/pure_fusion']:
        if tag in parents:return 'excluded/'+tag.rsplit('/',1)[-1]
    if name in PURE_LAYOUT:return 'excluded/pure_layout'
    return 'retained_GPU'


def summarize(events):
    """Use direct CPU->kernel links; never inclusive parent/child CUDA spans."""
    totals=collections.defaultdict(float);nodes=[]
    for event in events:
        if str(event.device_type).split('.')[-1]!='CPU':continue
        parents=[];current=event
        while current is not None:
            parents.append(current.name);current=current.cpu_parent
        durations=[float(k.duration) for k in event.kernels]
        if not durations:continue
        value=sum(durations)/1000;group=billing_group(event.name,parents)
        totals[group]+=value
        nodes.append(dict(operator=event.name,ancestors=parents,kernel_us=durations,
            kernel_names=[k.name for k in event.kernels],group=group,ms=value))
    # Preserve and bill unlinked GPU activity rather than silently dropping it.
    device=sum(float(e.time_range.elapsed_us())/1000 for e in events
        if str(e.device_type).split('.')[-1]=='CUDA')
    linked=sum(totals.values());missing=device-linked
    if missing < -max(.05,device*.001):
        raise ValueError('CPU-kernel attribution double counts device activity')
    totals['retained_unattributed_GPU']=max(0.,missing)
    return dict(groups_ms=dict(totals),device_activity_ms=device,linked_activity_ms=linked,
        retained_GPU_ms=totals['retained_GPU']+totals['retained_unattributed_GPU'],nodes=nodes)


def wrapped(original,name,cpu=None):
    def call(*args,**kwargs):
        clock=(time.process_time(),time.thread_time(),time.perf_counter()) if cpu is not None else None
        with torch.profiler.record_function(name):result=original(*args,**kwargs)
        if cpu is not None:
            cpu.append(dict(scope=name,process_CPU_ms=(time.process_time()-clock[0])*1000,
                caller_thread_CPU_ms=(time.thread_time()-clock[1])*1000,
                wall_ms=(time.perf_counter()-clock[2])*1000))
        return result
    return call


class ProcessorProxy:
    def __init__(self,processor,records):self.processor=processor;self.records=records
    def __getattr__(self,name):return getattr(self.processor,name)
    def __call__(self,*args,**kwargs):
        return wrapped(self.processor,'T12/CPU_preprocessing',self.records)(*args,**kwargs)


class Observer:
    def __init__(self,output):self.output=Path(output);self.requests=[];self.cpu=[]
    def prepare(self,qwen,processor,head,pipe):
        if qwen is None or not hasattr(qwen.model,'hybrid'):
            raise ValueError('This scope is exclusive to the real single-EG LG model')
        macros=qwen.model.hybrid
        if set(macros)!={'vision','language'} or any(m.pairs!=1 for m in macros.values()):
            raise ValueError('One router and one E/G pair per modality required')
        for macro in macros.values():
            for m in macro.electronic:m.forward=wrapped(m.forward,'T12/electronic_residual')
            for m in macro.fusions:m.forward=wrapped(m.forward,'T12/pure_fusion')
            path=macro.paths[0]
            path._t12_physical_simulate=wrapped(path._t12_physical_simulate,'T12/FFT_simulation')
            router=path.core.router
            router._t12_physical_simulate=wrapped(router._t12_physical_simulate,'T12/FFT_simulation')
        self.macros=macros
        pipe.image_processor.postprocess=wrapped(pipe.image_processor.postprocess,'T12/CPU_output',self.cpu)
        return ProcessorProxy(processor,self.cpu)
    def begin(self,row):
        self.cpu.clear();self.profiler=torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
        self.started=time.perf_counter();self.profiler.__enter__()
    def end(self,row):
        torch.cuda.synchronize();self.profiler.__exit__(None,None,None)
        elapsed=(time.perf_counter()-self.started)*1000
        counts=[dict(modality=n,pairs=m.pairs,router_calls=m.router_calls,
            optical_cycles=2*m.pairs+m.router_calls) for n,m in self.macros.items()]
        assert all(q['pairs']==q['router_calls']==1 for q in counts)
        cycles=sum(q['optical_cycles'] for q in counts);assert cycles==6
        result=summarize(self.profiler.events());assert result['device_activity_ms']>0
        # Every request executes actual preprocessing and output; no coverage by
        # another request, no padding fields or frame/batch amortization.
        assert {q['scope'] for q in self.cpu}=={'T12/CPU_preprocessing','T12/CPU_output'}
        cpu=sum(q['process_CPU_ms'] for q in self.cpu)
        serial=result['retained_GPU_ms']+cpu+cycles*PERIOD_MS
        trace=self.output/(row['sample_id']+'.trace.json');self.profiler.export_chrome_trace(str(trace))
        zipped=Path(str(trace)+'.gz')
        with trace.open('rb') as f,gzip.open(zipped,'wb') as out:
            import shutil
            shutil.copyfileobj(f,out)
        # This generated uncompressed trace is solely a recoverable duplicate.
        assert zipped.exists() and zipped.stat().st_size>0
        with gzip.open(zipped,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==hashlib.sha256(trace.read_bytes()).hexdigest()
        trace.unlink()
        nodes=self.output/(row['sample_id']+'.kernel_nodes.json')
        nodes.write_text(json.dumps(result.pop('nodes')))
        item=dict(sample_id=row['sample_id'],serial_electronic_work_plus_optical_model_ms=serial,
            optical_cycles=cycles,optical_period_ms=PERIOD_MS,optical_model_ms=cycles*PERIOD_MS,
            CPU_required_work_ms=cpu,CPU_scopes=list(self.cpu),instrumented_wall_ms=elapsed,
            actual_modal_cycles=counts,trace_gzip=zipped.name,trace_sha256=hashlib.sha256(zipped.read_bytes()).hexdigest(),
            kernel_nodes=nodes.name,kernel_nodes_sha256=hashlib.sha256(nodes.read_bytes()).hexdigest(),**result)
        self.requests.append(item)
        (self.output/'scoped_latency_progress.json').write_text(json.dumps(dict(done=len(self.requests),latest=item)))
        del self.profiler
    def finish(self):
        assert len(self.requests)==8
        report=dict(complete=True,requests=self.requests,n=8,B=1,first_request_retained=True,
            cross_request_parallelism=False,preprocessing_covered_by_other_requests=False,
            optical_time_multiplied_by_UNet_steps=False,physical_optical_measurement=False,
            verified_physical_DMA_or_SDK_time=False,
            count_scope='Retained actual GPU kernel work + all-thread required CPU work + 6 serial fixed optical cycles',
            pure_submission_waits_electronic_residual_and_fusion_excluded=True,
            CPU_work_model_not_wall=True,software_profile_wall_has_instrumentation=True,
            quality_or_speed_accepted=False)
        (self.output/'scoped_latency.json').write_text(json.dumps(report,indent=2))


def main():
    from . import scene_paired_render as render
    original=render.render
    def observed(a):
        if not a.measure_software_latency or not a.fuse_task_lora:
            raise ValueError('Use the fixed B1 timing cohort and the evaluated merged graph')
        a.request_observer=Observer(a.output);original(a);a.request_observer.finish()
    render.render=observed
    render.main()


if __name__=='__main__':main()
