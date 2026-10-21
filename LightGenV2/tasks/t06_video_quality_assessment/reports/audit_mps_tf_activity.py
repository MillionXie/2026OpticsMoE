"""CPU replay of both diagnostic instrumentation modes, not a paper-cost update."""
import hashlib
import json
from pathlib import Path
import subprocess
from .. import diagnose_mps_tf_activity as probe
from .. import tf_activity_clock as clock


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=probe.ROOT;points=[];source_cache={}
    for mode,folder in [('cpu_and_cuda',root),('cuda_only',root/'cuda_only01')]:
        state=read(folder/'suite_status.json')
        if state['status']!='complete' or not state['mps_released']:raise RuntimeError('Diagnosis did not finish and release')
        for name in state['completed']:
            run=folder/name;report=read(run/'report.json');identity=report['identity'];recorded=report['measurement']
            assert identity['batch']==1 and identity['explicit_warmup_forwards']==0 and identity['official_costs_unchanged']
            for path,digest in identity['sources'].items():
                key=identity['git_commit'],path
                if key not in source_cache:
                    raw=subprocess.check_output(['git','show',key[0]+':'+path],cwd=probe.REPO)
                    source_cache[key]=hashlib.sha256(raw).hexdigest()
                if source_cache[key]!=digest:raise RuntimeError('Published source mismatch: '+path)
            calls=read(run/'calls.json');anchors=read(run/'clock_anchors.json');trace=read(run/'cuda_trace.json')
            assert len(calls)==12 and len({c['sample_id'] for c in calls})==12
            allowance=None
            if mode=='cuda_only':
                errors=[clock.calibration_error(read(run/(n+'.json')),read(run/(n+'_anchors.json'))) for n in ('clock_before','clock_after')]
                assert errors==recorded['empty_cpu_clock_calibration_errors_us']
                allowance=max(errors)+10
                assert allowance==recorded['epoch_allowance_us']
                assert not any(e.get('cat')=='cpu_op' for e in trace['traceEvents'])
            replay=probe.summarize_trace(trace,calls,anchors,allowance)
            for key in ('conditional_kernel_busy_percent','lower_percent','upper_percent','kernel_union_us','tf_to_output_us'):
                if abs(replay[key]-recorded[key])>1e-7:raise RuntimeError('Activity replay differs')
            later=replay['per_video'][1:]
            later_percent=100*sum(p['kernel_busy_us'] for p in later)/sum(p['tf_to_output_us'] for p in later)
            points.append(dict(model=identity['model'],mps_cap=identity['cap'],visible_sms=identity['visible_multiprocessors'],
                trace_mode=mode,first12_conditional_kernel_busy_percent=replay['conditional_kernel_busy_percent'],
                first_sample_percent=replay['per_video'][0]['conditional_kernel_busy_percent'],
                fixed_later11_auxiliary_percent=later_percent,
                target50_reached=replay['target50_reached'],target100_reached=replay['target100_reached'],
                report=run.relative_to(probe.REPO).as_posix()+'/report.json',report_sha256=sha(run/'report.json'),
                trace_sha256=sha(run/'cuda_trace.json'),calls_sha256=sha(run/'calls.json'),
                not_uninstrumented_nvml_utilization=True,not_a_formal_cost_point=True))
    assert len(points)==24
    out=probe.REPO/'LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/recalculated_20261013'
    mps=probe.TASK/'runs/simulation/lgvq_mps_dataset_once_rental_20261021_replacement02/presentation'
    protected=read(mps/'figure_receipt.json')['files']
    protected['LGVQ_MPS_dataset_once_comparison.xlsx']='379a6808be885ee3bb19bcb9d800387cd36c89c3e11088edc60be4f7b564d957'
    for name,digest in protected.items():
        if sha(out/name)!=digest:raise RuntimeError('Protected MPS export changed: '+name)
    result=dict(status='passed',diagnostic_points=24,videos_per_point=12,points=points,
        metric='Compute-kernel interval union divided by native TF/output interval union. Not SM occupancy or NVML driver-window utilization.',
        cold_sample_policy='First sample retained in primary twelve-video result. Fixed later11 is exploratory auxiliary only, not a selected best window or replacement target.',
        limitation='Both collection modes have profiler overhead; none of these time or energy observations replaces official unprofiled costs.',
        protected_mps_exports_unchanged=True,protected_exports_sha256=protected)
    target=root/'activity_audit.json'
    with target.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(status='passed',points=24,current_mps_exports_unchanged=True)))
    for p in points:
        print(p['trace_mode'],p['model'],p['mps_cap'],round(p['first12_conditional_kernel_busy_percent'],2),round(p['fixed_later11_auxiliary_percent'],2))


if __name__=='__main__':main()
