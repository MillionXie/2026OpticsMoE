"""Unprofiled complete-request process CPU counter, including submission overhead.

This deliberately overcounts necessary CPU computation: launch and busy-poll CPU
also remain. It bounds unknown necessary CPU work without treating it as zero.
"""
import json,time


def cpu_record(start,end):
    process=(end[0]-start[0])*1000;thread=(end[1]-start[1])*1000;wall=(end[2]-start[2])*1000
    if min(process,thread,wall)<0:raise ValueError('CPU clocks moved backwards')
    return dict(process_CPU_ms=process,caller_thread_CPU_ms=thread,wall_ms=wall,
        all_process_threads_counted=True,CPU_submission_and_busy_poll_not_subtracted=True,
        CPU_work_upper_not_CPU_latency=True,not_a_profiled_backend_envelope=True)


def main():
    import sys
    if '--profile-known-work'in sys.argv:raise ValueError('This CPU upper must be measured without the profiler')
    from . import scene_compact_latency as latency
    from .white_reference_scene_probe import write
    old=latency.Observer;instances=[]
    class Meter(old):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);assert not self.profile;self.cpu_bounds=[];instances.append(self)
        def begin(self,row):
            super().begin(row);self.cpu_clock=(time.process_time(),time.thread_time(),time.perf_counter())
        def end(self,row):
            end=(time.process_time(),time.thread_time(),time.perf_counter());record=cpu_record(self.cpu_clock,end)
            self.cpu_bounds.append(dict(record,sample_id=row['sample_id']));super().end(row)
    latency.Observer=Meter
    try:latency.main()
    finally:latency.Observer=old
    assert len(instances)==1;meter=instances[0];assert len(meter.cpu_bounds)==8
    verified=json.loads((meter.output/'software_latency.json').read_text());report=json.loads((meter.output/'report.json').read_text())
    assert verified['output_PNG_identity_vs_completed_quality_run']and not verified.get('contains_profiler_overhead')
    write(meter.output/'unprofiled_CPU_upper.json',dict(complete=True,n=8,requests=meter.cpu_bounds,
        checkpoint_sha256=report['checkpoint_sha256'],output_PNG_identity_checked=True,
        model_load_outside_counter=True,first_request_retained=True,extra_warmup=False,profiler_used=False,
        upper_includes_unnecessary_submission_CPU=True,unknown_hardware_cost_not_included=True))


if __name__=='__main__':main()
