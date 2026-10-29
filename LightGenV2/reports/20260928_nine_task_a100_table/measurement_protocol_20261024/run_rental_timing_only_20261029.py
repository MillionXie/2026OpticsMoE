"""Urgent timing-only delivery: original native512/B1/MPS, NO qualified energy.

Human2026-10-29 asked to proceed after explained telemetry query stalls.
The250ms energy gate is NOT widened or claimed passed. No energy sampler runs
inside this separately labelled timing experiment. All failed diagnostics stay.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import run_rental_fullscope_20261029 as native

ENTRY=Path(__file__).resolve();HERE=ENTRY.parent;REPO=native.REPO;base=native.base
ROOT=base.TASK/'runs/simulation/rental_timing_only_20261029_01'
PIPE='/data/tmp/rental_timing_only_20261029_01';GPU=native.GPU
PROTOCOL='rental80_TIMING_ONLY_energy_UNQUALIFIED_file_to_CPU_B1_512x3_20261029'
DEADLINE=1793278441.7376518
NAMES=None

def sources():return {n:base.sha(REPO/n) for n in NAMES}

def config():
    return dict(native.config(),protocol_id=PROTOCOL,energy_qualified=False,
        energy_policy='NOT measured/qualified in this timing-only protocol; never estimate J or energy ratios',
        authorization='Human urgent proceed after explanation of NVML latency; timing first, energy explicitly unqualified',
        telemetry_diagnostics_are_not_formal_energy=True)

def timing_cost(calls):
    if len(calls)!=512 or len({r['sample_id'] for r in calls})!=512:raise ValueError('Exactly512 distinct calls')
    previous=None
    for r in calls:
        p,a,s,e,t=(float(r[k]) for k in ('preprocess_start','model_call_start','first_block_start','gpu_output','call_return'))
        if not p<=a<=s<e<=t or (previous is not None and previous>p):raise ValueError('Original ordered full-input boundaries')
        previous=t
    elapsed=float(calls[-1]['call_return'])-float(calls[0]['preprocess_start'])
    return dict(videos=512,frames=2048,groups16=32,continuous_wall_s=elapsed,
        time16_s=elapsed/32,ms_per_video=elapsed/512*1000,energy_qualified=False,
        gpu_energy_j=None,estimated_system_energy_j=None,estimated_joules_per_video=None)

def audit_point(folder,manifest):
    folder=Path(folder);report=base.read(folder/'report.json');rows=base.input_rows(manifest)
    if (report['status']!='complete' or report['protocol_id']!=PROTOCOL or report['energy_qualified'] is not False
            or report['measured_videos']!=512 or report['batch']!=1 or report['explicit_warmup_forwards']!=0
            or report['gpu_uuid']!=GPU or report['visible_multiprocessors']!=(54 if report['cap']==50 else 108)):
        raise ValueError('Exact qualified TIMING-only contract required')
    with (folder/'calls.csv').open(newline='') as f:calls=list(csv.DictReader(f))
    if [r['sample_id'] for r in calls]!=[r['sample_id'] for r in rows]:raise ValueError('Original first512 order differs')
    value=timing_cost(calls)
    if not math.isclose(value['continuous_wall_s'],report['cost']['continuous_wall_s'],rel_tol=1e-12,abs_tol=1e-8):raise ValueError('Timing replay differs')
    _,reference=base.reference(report['model'])
    if (report['assets']!=reference['assets'] or report['checkpoint_sha256']!=reference['checkpoint_sha256']
            or report['floating_parameter_dtypes']!=reference['floating_parameter_dtypes']
            or report['quality_readout_dtypes']!=reference['quality_readout_dtypes']):
        raise ValueError('Original official model and trained head required')
    evidence=base.read(folder/'input_audit.json')
    if base.sha(folder/'input_audit.json')!=report['input_audit_sha256'] or [r['sample_id'] for r in evidence['samples']]!=[r['sample_id'] for r in rows]:raise ValueError('Input evidence differs')
    if sum(r.get('effective_input_tokens',0) for r in evidence['samples'])!=report['effective_input_tokens512']:raise ValueError('Native token numerator differs')
    if set(report['input_files'])!={r['sample_id'] for r in rows}:raise ValueError('Actual input byte hashes required')
    if [float(r['target_mos']) for r in calls]!=[float(r['temporal']) for r in rows]:raise ValueError('Original targets differ')
    from scipy.stats import spearmanr,pearsonr
    pred=[float(r['prediction']) for r in calls];target=[float(r['target_mos']) for r in calls]
    for actual,expected in [(spearmanr(target,pred).statistic,report['performance512']['srcc']),(pearsonr(target,pred).statistic,report['performance512']['plcc'])]:
        if not math.isclose(actual,expected,abs_tol=1e-6,rel_tol=1e-9):raise ValueError('Subset metric replay differs')
    for name,digest in report['sources'].items():
        blob=subprocess.check_output(['git','show',report['git_commit']+':'+name.replace('\\','/')],cwd=REPO,timeout=15)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Published source differs')
    for event in base.read(folder/'process_audit.json'):
        allowed={str(report['pid'])}|{str(p) for p in event['servers']}
        if any(str(v[0]).strip() not in allowed or str(v[1]).strip()!=GPU for v in event['processes']):raise ValueError('Foreign GPU process')
    return dict(status='passed',qualification='TIMING_ONLY_ENERGY_UNQUALIFIED',energy_qualified=False,
        report_sha256=base.sha(folder/'report.json'),calls_sha256=base.sha(folder/'calls.csv'),
        model=report['model'],cap=report['cap'],repeat=report['repeat'],measured_videos=512,cost=value)

def worker(a):
    rows=base.input_rows(a.manifest);ref_path,ref=base.reference(a.model)
    if a.output.exists():raise FileExistsError('No repeat/overwrite')
    if base.sha(a.checkpoint)!=config()['references'][a.model]['head_sha256']:raise ValueError('Original trained head')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(a.cap):raise ValueError('Exact MPS/GPU before CUDA')
    legacy=native.initialize(a);bound=sources();a.output.mkdir(parents=True)
    records=[];evidence=[];guard=None;finish=None
    identity=dict(protocol_id=PROTOCOL,model=a.model,cap=a.cap,repeat=a.repeat,batch=1,pid=os.getpid(),
        gpu_uuid=GPU,git_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'],sources=bound,energy_qualified=False,
        qualification='TIMING_ONLY_ENERGY_UNQUALIFIED',energy_sampler_not_started=True,
        failed_energy_diagnostics_preserved_separately=True,manifest_sha256=base.sha(a.manifest),
        first512_ids_sha256=base.ordered_id_sha(rows),explicit_warmup_forwards=0,first_sample_included=True,
        expected_head_sha256=ref['checkpoint_sha256'],command=sys.argv)
    def status(value,**kw):legacy.frozen.write_json(a.output/'status.json',dict(status=value,pid=os.getpid(),time=time.time(),samples=len(records),energy_qualified=False,**kw))
    class Guard(legacy.engine.MPSGuard):
        def check(self):
            if sources()!=bound:raise RuntimeError('Timing source changed')
            super().check()
    try:
        import torch
        import numpy as np
        from LightGenV2.tasks.t06_video_quality_assessment import quality_token_common as core
        guard=Guard(a);guard.check();torch.set_num_threads(8)
        cap=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
        if cap!=400:raise ValueError('Original new-card400W cap, never change it')
        identity.update(power_limit_w=cap,clock_offset_s=time.perf_counter()-time.monotonic())
        if abs(identity['clock_offset_s'])>1e-4:raise ValueError('Clock domains differ')
        metadata=base.file_identities(rows)
        identity.update(input_files_metadata_before=metadata,
            cache_protocol='No video-byte prehash, decoding preload, cache flush or dummy model forward before T0')
        status('loading')
        with torch.inference_mode():
            all_rows,prepare,transfer,forward,start,finish=legacy.runtime(a,guard,identity)
            props=torch.cuda.get_device_properties(0)
            if (torch.cuda.device_count()!=1 or 'A100' not in props.name or props.total_memory<70*1024**3
                    or props.multi_processor_count!=(54 if a.cap==50 else 108)
                    or base.ordered_id_sha(all_rows[:512])!=base.ordered_id_sha(rows)):
                raise ValueError('Original native inputs and actual MPS54/108 resources required')
            if (identity['assets']!=ref['assets'] or identity['floating_parameter_dtypes']!=ref['floating_parameter_dtypes']
                    or identity['quality_readout_dtypes']!=ref['quality_readout_dtypes']):raise ValueError('Original native math/precision identity')
            identity.update(visible_multiprocessors=props.multi_processor_count,gpu_name=props.name,
                environment=dict(python=sys.version,torch_cuda=torch.version.cuda,CPU_threads=torch.get_num_threads(),
                    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
            base.collect(rows,prepare,transfer,forward,start,torch.cuda.synchronize,guard.fast,
                progress=lambda n:status('evaluating',completed_samples=n),records=records,
                input_observer=lambda row,cpu:evidence.append(dict(sample_id=row['sample_id'],**base.cpu_input_evidence(cpu))))
            cost=timing_cost(records)
            performance=core.metrics(np.array([r['target_mos'] for r in records]),np.array([r['prediction'] for r in records]))
            with ref_path.with_name('calls.csv').open(newline='') as f:original=list(csv.DictReader(f))[:512]
            if len(original)!=512:raise ValueError('Original512 frame-selection references required')
            if any(a['sample_id']!=b['sample_id'] or json.loads(a['frame_positions'])!=json.loads(b['frame_positions']) for a,b in zip(records,original)):raise ValueError('Original four frame positions differ')
            finish();finish=None
            identity['input_files']=base.file_identities(rows,hash_content=True)
            if base.file_identities(rows)!=metadata:raise ValueError('Input changed during timing')
            legacy.frozen.write_json(a.output/'input_audit.json',dict(samples=evidence,original_video_metadata=base.input_video_metadata(rows),original_metadata_read_after_T1=True))
            identity.update(input_audit_sha256=base.sha(a.output/'input_audit.json'),effective_input_tokens512=sum(r.get('effective_input_tokens',0) for r in evidence))
            guard.check()
            legacy.frozen.write_json(a.output/'report.json',dict(status='complete',**identity,measured_videos=512,
                reference_test_videos=558,full558_reference_is_historical=True,cost=cost,performance512=performance,
                timer_start='first raw file preparation',timer_end='last synchronized CPU quality score',
                all_frontend_and_511_internal_gaps_included=True,energy_and_GPU_mean_power=None))
            status('complete',completed_samples=512)
    except BaseException as error:status('failed',error=repr(error));raise
    finally:
        try:
            if finish is not None:finish()
        finally:
            legacy.write_csv(a.output/'calls.csv',records)
            if not (a.output/'input_audit.json').exists():legacy.frozen.write_json(a.output/'input_audit_partial.json',dict(samples=evidence))
            legacy.frozen.write_json(a.output/'identity.json',identity)
            if guard is not None:guard.close(a.output/'process_audit.json')

def configure():
    global NAMES
    native.configure();NAMES=set(native.sources())|{ENTRY.relative_to(REPO).as_posix()}
    base.CANONICAL=ROOT;base.source_map=sources;base.config=config;base.audit_point=audit_point
    base.initialize_profile=native.initialize
    base.suite.__globals__.update(vars(base));base.suite.__globals__.update(ENTRY=ENTRY,NEW_PROTOCOL=PROTOCOL)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('suite','worker','audit'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--suite-root',type=Path);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--profile',choices=('rental',),default='rental');p.add_argument('--gpu-uuid',choices=(GPU,),default=GPU)
    p.add_argument('--deadline',type=float);p.add_argument('--pipe',default=PIPE);p.add_argument('--model',choices=('kimi','deepseek'))
    p.add_argument('--cap',type=int,choices=(50,100));p.add_argument('--repeat',type=int,choices=(1,2,3));p.add_argument('--model-path',type=Path);p.add_argument('--checkpoint',type=Path)
    a=p.parse_args();a.output=a.output.resolve();a.suite_root=(a.suite_root or a.output).resolve()
    if a.suite_root!=ROOT or a.pipe!=PIPE or (a.phase=='suite' and a.output!=ROOT) or (a.phase!='suite' and a.output.parent!=ROOT):raise ValueError('Single new TIMING-only root required')
    if a.phase!='audit' and (a.deadline!=DEADLINE or time.time()>=DEADLINE-120):raise ValueError('Original overall cutoff, no extension')
    a.models=['kimi','deepseek'];a.caps=[50,100];a.repeats=[1,2,3];configure()
    if a.phase=='audit':print(json.dumps(audit_point(a.output,a.manifest),ensure_ascii=True));return
    native.identity.verify(os.environ['MEASUREMENT_PUBLISHED_COMMIT'],sources(),REPO)
    if a.phase=='suite':native.verified_admission_predecessor();native.prepare_pipe_parent(PIPE)
    (base.suite if a.phase=='suite' else worker)(a)

if __name__=='__main__':main()
