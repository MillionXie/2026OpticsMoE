"""Conservative component bounds, never a physical end-to-end measurement."""
import argparse,json,math
from pathlib import Path


def gpu_kernel_union_ms(events):
    intervals=[]
    for event in events:
        if event.get('cat')!='kernel' or event.get('ph')!='X':continue
        start=float(event['ts']);stop=start+float(event['dur'])
        if not math.isfinite(start+stop) or stop<start:raise ValueError('Invalid GPU interval')
        intervals.append((start,stop))
    if not intervals:raise ValueError('No raw GPU kernel events; cannot invent a baseline lower bound')
    intervals.sort();total=0.;start,stop=intervals[0]
    for first,last in intervals[1:]:
        if first<=stop:stop=max(stop,last)
        else:total+=stop-start;start,stop=first,last
    return(total+stop-start)/1000


def conservative_components(baseline_gpu_union,student_gpu_serial,student_cpu_upper,period_ms=.8609226935312833):
    values=[baseline_gpu_union,student_gpu_serial,student_cpu_upper,period_ms]
    if not all(math.isfinite(v)and v>=0 for v in values)or baseline_gpu_union==0:raise ValueError('Valid measured components required')
    upper=student_gpu_serial+student_cpu_upper+6*period_ms
    return dict(baseline_GPU_occupied_lower_ms=baseline_gpu_union,student_retained_GPU_serial_ms=student_gpu_serial,
        student_unprofiled_all_process_CPU_upper_ms=student_cpu_upper,fixed_optical_ms=6*period_ms,
        modeled_student_upper_before_extra_hardware_ms=upper,
        conservative_component_ratio_before_extra_hardware=baseline_gpu_union/upper,
        additional_hardware_overhead_budget_for_4x_ms=baseline_gpu_union/4-upper,
        extra_hardware_DMA_SDK_cost='unknown H; not assumed zero',
        profiled_CPU_submission_envelope_not_used=True,physical_measurement=False,
        statement='Conditional component-model bound only; H must fit the reported budget. Not a closed physical deployment DAG.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timing-workflow',type=Path,required=True)
    p.add_argument('--cpu-runs',type=Path,nargs=3,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import gzip,statistics
    from .white_reference_scene_probe import digest,write
    ready=json.loads((a.timing_workflow/'report.json').read_text());assert ready['complete']and not a.output.exists()
    key=ready['selected_candidate']['key'];runs=json.loads((a.timing_workflow/'completed_timing_runs.json').read_text())
    profile={name:[Path(r['run'])for r in runs if r['key']==name and r['mode']=='known_work_profile']for name in ['E30',key]}
    assert all(len(v)==3 for v in profile.values());baseline=[];student=[];cpu=[];identities=[]
    expected=None
    def signature(root):
        rows=json.loads((root/'images.json').read_text())
        assert len(rows)==8
        return {r['sample_id']:(r['input_sha256'],r['prompt'],r['seed'],r['initial_latent_sha256'])for r in rows}
    for root in profile['E30']:
        sign=signature(root)
        if expected is None:expected=sign
        assert sign==expected
        records=json.loads((root/'known_work.json').read_text())['requests']
        for r in records:
            trace=root/r['trace_gzip'];assert digest(trace)==r['trace_sha256']
            with gzip.open(trace,'rt')as f:events=json.load(f)['traceEvents']
            baseline.append(gpu_kernel_union_ms(events))
    for root in profile[key]:
        assert signature(root)==expected
        report=json.loads((root/'report.json').read_text());assert report['checkpoint_sha256']==json.loads((Path(ready['selected_candidate']['checkpoint'])/'report.json').read_text())['best_sha256']
        records=json.loads((root/'known_work.json').read_text())['requests'];assert all(r['optical_cycles']==6 for r in records)
        student.extend(r['retained_GPU_ms']for r in records)
    for root in a.cpu_runs:
        assert signature(root)==expected
        record=json.loads((root/'unprofiled_CPU_upper.json').read_text());assert record['complete']and record['n']==8 and record['output_PNG_identity_checked']
        assert record['checkpoint_sha256']==json.loads((Path(ready['selected_candidate']['checkpoint'])/'report.json').read_text())['best_sha256']
        cpu.extend(r['process_CPU_ms']for r in record['requests']);identities.append(dict(run=str(root),report_sha256=digest(root/'unprofiled_CPU_upper.json')))
    assert len(baseline)==len(student)==len(cpu)==24
    a.output.mkdir();result=conservative_components(statistics.mean(baseline),statistics.mean(student),statistics.mean(cpu))
    write(a.output/'report.json',dict(result,complete=True,CPU_upper_runs=identities,timing_report_sha256=digest(a.timing_workflow/'report.json'),
        baseline_gpu_kernel_occupied_ms=baseline,student_retained_GPU_ms=student,unprofiled_CPU_upper_ms=cpu,
        independent_processes_each_scope=3,n=24,no_fastest_round_selection=True,quality_or_physical_speed_accepted=False))


if __name__=='__main__':main()
