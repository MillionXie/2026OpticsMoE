"""Matched complete-request wall timing and separate FFT-replacement work model.

The profiled known-work column is not a physical end-to-end measurement or a
closed deployment DAG. Unknown scheduler CPU work/communication is listed, not
silently priced at zero. Never reuse profiler wall as the uninstrumented wall.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import time


def known_work(retained_gpu_ms,cpu_work_ms,cycles,period_ms):
    if cycles not in (0,6) or min(retained_gpu_ms,cpu_work_ms,period_ms)<0:raise ValueError('Invalid work-cost components')
    return dict(retained_GPU_ms=retained_gpu_ms,required_CPU_preprocess_output_ms=cpu_work_ms,
        optical_cycles=cycles,fixed_optical_ms=cycles*period_ms,
        known_serial_work_ms=retained_gpu_ms+cpu_work_ms+cycles*period_ms,
        unknown_CPU_scheduler_and_communication_cost='unmeasured; not assumed zero',
        physical_measurement=False,closed_physical_DAG=False)


class Observer:
    def __init__(self,output,config=None,profile=False):
        self.output=Path(output);self.config=config;self.profile=profile;self.cpu=[];self.rows=[]
    def prepare(self,qwen,processor,head,pipe):
        if self.config:
            from .scene_compact_candidates import GraphObserver
            graph=GraphObserver(self.config);processor=graph.prepare(qwen,processor,head,pipe);self.graph=graph.graph
        else:self.graph=None
        self.macros=getattr(qwen.model,'hybrid',{})
        if self.macros and (set(self.macros)!={'vision','language'} or any(m.pairs!=1 for m in self.macros.values())):
            raise ValueError('One E/G per modality required')
        if self.profile:
            from .scene_scoped_latency import wrapped,ProcessorProxy
            for macro in self.macros.values():
                for layer in macro.electronic:layer.forward=wrapped(layer.forward,'T12/electronic_residual')
                for layer in macro.fusions:layer.forward=wrapped(layer.forward,'T12/pure_fusion')
                path=macro.paths[0];path._t12_physical_simulate=wrapped(path._t12_physical_simulate,'T12/FFT_simulation')
                router=path.core.router;router._t12_physical_simulate=wrapped(router._t12_physical_simulate,'T12/FFT_simulation')
            pipe.image_processor.postprocess=wrapped(pipe.image_processor.postprocess,'T12/CPU_output',self.cpu)
            processor=ProcessorProxy(processor,self.cpu)
        return processor
    def begin(self,row):
        if not self.profile:return
        import torch
        self.cpu.clear();self.profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]);self.profiler.__enter__()
    def end(self,row):
        if not self.profile:return
        import torch
        from .scene_scoped_latency import summarize,PERIOD_MS
        torch.cuda.synchronize();self.profiler.__exit__(None,None,None)
        counts=[dict(modality=name,router_calls=m.router_calls,pairs=m.pairs)for name,m in self.macros.items()]
        if counts:assert len(counts)==2 and all(r['router_calls']==r['pairs']==1 for r in counts)
        cycles=6 if counts else 0;summary=summarize(self.profiler.events());assert summary['device_activity_ms']>0
        assert {q['scope']for q in self.cpu}=={'T12/CPU_preprocessing','T12/CPU_output'}
        result=known_work(summary['retained_GPU_ms'],sum(q['process_CPU_ms']for q in self.cpu),cycles,PERIOD_MS)
        result.update(sample_id=row['sample_id'],groups_ms=summary['groups_ms'],CPU_scopes=list(self.cpu),actual_modal_cycles=counts,
            FFT_cost_removed=True,electronic_residual_and_pure_fusion_excluded=True,CPU_submission_wait_wall_not_billed=True,
            profiler_CPU_measurement_may_include_instrumentation=True,not_a_baseline_wall_divided_by_LG_partial_cost_speed_claim=True)
        trace=self.output/(row['sample_id']+'.trace.json');self.profiler.export_chrome_trace(str(trace))
        zipped=Path(str(trace)+'.gz')
        with trace.open('rb')as source,gzip.open(zipped,'wb')as dest:
            import shutil
            shutil.copyfileobj(source,dest)
        from .white_reference_scene_probe import digest,write
        # Keep the raw trace compressed, and verify compression before removing its duplicate.
        import hashlib
        with gzip.open(zipped,'rb')as stream:assert hashlib.sha256(stream.read()).hexdigest()==digest(trace)
        trace.unlink();nodes=self.output/(row['sample_id']+'.kernel_nodes.json')
        write(nodes,summary['nodes']);result.update(trace_gzip=zipped.name,trace_sha256=digest(zipped),kernel_nodes=nodes.name,kernel_nodes_sha256=digest(nodes))
        self.rows.append(result);write(self.output/'known_work_progress.json',dict(done=len(self.rows),latest=result));del self.profiler


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--compression-config',type=Path);p.add_argument('--profile-known-work',action='store_true')
    custom,remaining=p.parse_known_args();config=None
    if custom.compression_config:
        from .scene_depth_probe import checked_config
        config=checked_config(json.loads(custom.compression_config.read_text()))
    from . import scene_paired_render as renderer
    from .white_reference_scene_probe import digest,write
    import torch
    old=renderer.render
    def render(a):
        if (not a.measure_software_latency or not a.reference_run or not a.fuse_task_lora or a.scope!='dev'
            or not a.prompt_revision or a.sampler!='original' or a.lcm_adapter):
            raise ValueError('Fixed B1 natural DEV timing, merged graph and completed matching quality run required')
        if config:
            state=torch.load(a.checkpoint/'best_checkpoint.pt',map_location='cpu',weights_only=False)
            report=json.loads((a.checkpoint/'report.json').read_text());description=state.get('depth_compression',{})
            assert description.get('schema')in ['scene_compact_recovered_v1','scene_recovered_depth_v1']
            assert report['compression_training_performed'] and description['profile']==config
            assert digest(a.checkpoint/'best_checkpoint.pt')==report['best_sha256']
            quality=json.loads((a.reference_run/'report.json').read_text());assert quality['compression_training_performed']
            assert a.steps==state['sampling_contract']['steps']==quality['sampling_steps']
            a._verified_depth_compression=description
        observer=Observer(a.output,config,custom.profile_known_work);a.request_observer=observer;old(a)
        if custom.profile_known_work:
            assert len(observer.rows)==8
            write(a.output/'known_work.json',dict(complete=True,n=8,requests=observer.rows,
                same_scope_for_full_Qwen_and_LG=True,first_request_retained=True,extra_warmup=False,
                known_work_model_not_complete_latency=True,unmeasured_costs_not_assumed_zero=True,
                physical_optical_measurement=False,actual_compressed_graph=observer.graph))
            path=a.output/'software_latency.json';value=json.loads(path.read_text())
            value.update(contains_profiler_overhead=True,not_formal_software_wall=True);write(path,value)
        report=json.loads((a.output/'report.json').read_text());report.update(actual_compressed_graph=observer.graph,
            instrumented_timing_not_formal_software_wall=custom.profile_known_work,physical_optical_measurement=False)
        write(a.output/'report.json',report)
    renderer.render=render;sys.argv=[sys.argv[0],*remaining]
    try:renderer.main()
    finally:renderer.render=old


if __name__=='__main__':main()
