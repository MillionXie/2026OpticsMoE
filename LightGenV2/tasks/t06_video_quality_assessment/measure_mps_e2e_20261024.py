"""Frozen B1 MPS measurements: first file preparation to last CPU quality score.

The old native arithmetic is reused, never its legacy cost normalization. No
warmup/Graph/training, no historical Qwen cost override, no change to old runs.
The first 200 distinct TEST videos are processed in three fresh processes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
CONFIG = TASK/'configs/baselines/mps_e2e_20261024.json'
CANONICAL = TASK/'runs/simulation/lgvq_mps_e2e_a100_20261024'
PROTOCOL = 'file_to_cpu_result_e2e_200x3_b1_20261024_v1'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def config():
    c=read(CONFIG)
    if (c['protocol_id'],c['batch'],c['videos_per_repeat'],c['frames_per_video'],c['repeats'],
            c['resource_caps'],c['explicit_warmup_forwards'],c['host_proxy_w']) != (
            PROTOCOL,1,200,4,[1,2,3],[100,50],0,338.2):
        raise ValueError('Approved E2E contract differs')
    return c


def ordered_id_sha(rows):
    return hashlib.sha256('\n'.join(r['sample_id'] for r in rows).encode('utf-8')).hexdigest()


def input_rows(manifest):
    c=config()
    if sha(manifest)!=c['manifest_sha256']:raise ValueError('Frozen manifest SHA mismatch')
    with Path(manifest).open(newline='',encoding='utf-8-sig') as f:
        rows=[r for r in csv.DictReader(f) if r['split']=='test']
    selected=rows[:200]
    if len(rows)!=558 or len({r['sample_id'] for r in selected})!=200 or ordered_id_sha(selected)!=c['first200_ids_sha256']:
        raise ValueError('Require original ordered 200 distinct TEST videos')
    return selected


def reference(model):
    c=config()['references'][model]
    path=TASK/'runs/simulation'/c['report']
    if sha(path)!=c['report_sha256']:raise ValueError('Reference report changed')
    report=read(path)
    if report['status']!='complete' or report['checkpoint_sha256']!=c['head_sha256'] or report['batch']!=1:
        raise ValueError('Wrong frozen reference identity')
    return path,report


def file_identities(rows,hash_content=False):
    result={}
    for r in rows:
        path=Path(r['video_path']);st=path.stat()
        item=dict(path=str(path),size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,
            ctime_ns=st.st_ctime_ns,inode=st.st_ino)
        if hash_content:item['sha256']=sha(path)
        result[r['sample_id']]=item
    return result


def plan(profile,models=None,caps=None,repeats=None):
    c=config();allowed=c['profiles'][profile]['models']
    models=list(models or allowed);caps=list(caps or c['resource_caps']);repeats=list(repeats or c['repeats'])
    if (not models or len(set(models))!=len(models) or any(m not in allowed for m in models)
        or len(set(caps))!=len(caps) or any(v not in (100,50) for v in caps)
        or len(set(repeats))!=len(repeats) or any(v not in (1,2,3) for v in repeats)):
        raise ValueError('Finite approved model/cap/repeat plan required')
    return [(m,c,r) for m in models for c in caps for r in repeats]


def validate_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(CANONICAL.resolve()):raise ValueError('Only canonical T06 run paths')
    if path.exists():raise FileExistsError('Preserve existing complete or failed evidence: '+str(path))
    return path


def summarize(calls,samples):
    """CPU-only continuous wall and matching energy; includes ALL preparation."""
    from .strict_tf_window_20261020 import interval_energy
    if len(calls)!=200 or len({r['sample_id'] for r in calls})!=200:raise ValueError('200 unique calls required')
    parts=dict(prepare_s=0.,h2d_s=0.,model_front_s=0.,first_block_s=0.,cpu_return_s=0.,internal_gap_s=0.)
    keys=('preprocess_start','prepare_end','model_call_start','first_block_start','gpu_output','call_return')
    previous=None
    for i,r in enumerate(calls):
        p,d,a,s,e,z=[float(r[k]) for k in keys]
        if not all(math.isfinite(v) for v in (p,d,a,s,e,z)) or not p<=d<=a<=s<e<=z:
            raise ValueError('Invalid measured boundary order')
        if int(r['sample_index'])!=i:raise ValueError('Wrong ordered sample index')
        if previous is not None:
            if p<previous:raise ValueError('Overlapping serial B1 requests')
            parts['internal_gap_s']+=p-previous
        for k,v in zip(tuple(parts)[:5],(d-p,a-d,s-a,e-s,z-e)):parts[k]+=v
        previous=z
    start=float(calls[0]['preprocess_start']);end=float(calls[-1]['call_return'])
    elapsed=end-start
    if not math.isclose(sum(parts.values()),elapsed,abs_tol=1e-6):raise ValueError('Partition versus wall mismatch')
    if samples[0].host_monotonic_s>start or samples[-1].host_monotonic_s<end:
        raise ValueError('Power must bracket the entire continuous input-output window')
    energy=interval_energy([(start,end)],samples,config()['host_proxy_w'])
    return dict(protocol_id=PROTOCOL,start_monotonic_s=start,end_monotonic_s=end,
        continuous_wall_s=elapsed,charged_intervals=[[start,end]],videos=200,frames=800,
        ms_per_video=elapsed*1000/200,equivalent_ms_per_input_frame=elapsed*1000/800,
        estimated_joules_per_video=energy['energy_j']/200,
        estimated_joules_per_input_frame=energy['energy_j']/800,
        estimated_system_energy_j=energy['energy_j'],
        gpu_energy_j=energy['energy_j']-338.2*elapsed,host_proxy_w=338.2,
        components=parts,power_integration=energy,
        cuda_events_are_diagnostic_only=True,no_historical_cost_override=True)


def cpu_input_evidence(inputs):
    """Read CPU-only shape/small token metadata; never execute a GPU audit op."""
    from collections.abc import Mapping
    shapes={};small={}
    def visit(value,path):
        if isinstance(value,Mapping):
            for key,item in value.items():visit(item,str(key) if not path else path+'.'+str(key))
        elif hasattr(value,'shape') and hasattr(value,'dtype'):
            device=getattr(value,'device',None)
            if device is not None and getattr(device,'type',str(device))!='cpu':
                raise ValueError('Input evidence must not touch CUDA tensors')
            shapes[path or 'input']=dict(shape=[int(v) for v in value.shape],dtype=str(value.dtype))
            if path.split('.')[-1] in ('input_ids','attention_mask','image_grid_thw','video_grid_thw','images_spatial_crop'):
                small[path]=value.tolist()
        elif isinstance(value,(tuple,list)):
            for index,item in enumerate(value):visit(item,path+'.'+str(index))
    visit(inputs,'')
    result=dict(tensors=shapes,small_cpu_values=small,
        small_values_sha256=hashlib.sha256(json.dumps(small,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    ids=small.get('input_ids');mask=small.get('attention_mask')
    if ids is not None:
        if len(ids)!=1:raise ValueError('B1 token evidence required')
        if mask is not None and (len(mask)!=1 or len(mask[0])!=len(ids[0])):
            raise ValueError('Input IDs and attention mask differ')
        effective=ids[0] if mask is None else [v for v,active in zip(ids[0],mask[0]) if active]
        result.update(effective_input_tokens=len(effective),effective_input_ids_sha256=
            hashlib.sha256(json.dumps(effective,separators=(',',':')).encode()).hexdigest())
    return result


def input_video_metadata(rows):
    """Original file dimensions read AFTER T1; processor shapes come from CPU inputs."""
    import cv2
    result={}
    for row in rows:
        video=cv2.VideoCapture(row['video_path'])
        try:
            if not video.isOpened():raise ValueError('Video metadata could not be opened')
            width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if min(width,height)<=0:raise ValueError('Invalid original video dimensions')
            result[row['sample_id']]=dict(width=width,height=height,
                reported_frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:video.release()
    return result


def collect(rows,prepare,transfer,forward,first_start,synchronize,check,clock=time.perf_counter,progress=None,records=None,input_observer=None):
    """No warmup, synthetic waits, prepared-input reuse or conditional sampling."""
    if records is None:records=[]
    for index,row in enumerate(rows):
        check()
        p=clock();cpu,positions=prepare([row]);d=clock()
        inputs=transfer(cpu);synchronize();a=clock()
        values,legacy_wall_ms,cuda_ms=forward(inputs);synchronize();z=clock()
        s=float(first_start());e=s+float(legacy_wall_ms)/1000
        if len(values)!=1 or not math.isfinite(float(values[0])):raise ValueError('Finite CPU quality score required')
        records.append(dict(sample_index=index,sample_id=row['sample_id'],target_mos=float(row['temporal']),
            prediction=float(values[0]),preprocess_start=p,prepare_end=d,model_call_start=a,
            first_block_start=s,gpu_output=e,call_return=z,legacy_host_ms=float(legacy_wall_ms),
            diagnostic_first_block_cuda_ms=float(cuda_ms),frame_positions=json.dumps(positions[0])))
        if not p<=d<=a<=s<e<=z or len(positions[0])!=4:
            raise ValueError('Invalid single-video timing or original four-frame selection')
        if input_observer is not None:input_observer(row,cpu)
        del inputs,cpu
        if progress is not None and (index==0 or (index+1)%25==0):progress(index+1)
    return records


def source_map():
    from . import measure_mps_dataset_once as legacy
    extra=[Path(__file__),CONFIG,REPO/'LightGenV2/common/baseline_measurement.py',
        *[TASK/name for name in ('quality_token_common.py','quality_token_batch_benchmark_200.py',
            'visual_backbone_quality.py','evaluate_temporal_baseline_batches.py','nvml_observer_process.py','batch_telemetry.py')]]
    return {**legacy.sources(),**{str(path.relative_to(REPO)):sha(path) for path in extra}}


def initialize_profile(args):
    from . import measure_mps_dataset_once as legacy
    legacy.select_profile('rentalb' if args.profile=='rental' else 'lab')
    legacy.frozen.A100_UUID=args.gpu_uuid
    legacy.engine.GPU=args.gpu_uuid;legacy.engine.PIPE=args.pipe;legacy.engine.ROOT=args.suite_root
    return legacy


def worker(args):
    # All path/manifest/reference/environment checks precede loading a CUDA model.
    output=validate_output(args.output);rows=input_rows(args.manifest);ref_path,ref=reference(args.model)
    if sha(args.checkpoint)!=config()['references'][args.model]['head_sha256']:raise ValueError('Wrong quality head')
    if os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')!=str(args.cap) or os.environ.get('CUDA_VISIBLE_DEVICES')!=args.gpu_uuid:
        raise ValueError('Exact MPS cap and GPU UUID must precede CUDA initialization')
    legacy=initialize_profile(args);bound=source_map()
    output.mkdir(parents=True);records=[];input_evidence=[];sampler=None;finish=None;guard=None
    identity=dict(protocol_id=PROTOCOL,model=args.model,cap=args.cap,repeat=args.repeat,batch=1,
        gpu_uuid=args.gpu_uuid,pid=os.getpid(),sources=bound,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        manifest_sha256=sha(args.manifest),first200_ids_sha256=ordered_id_sha(rows),
        reference_report=str(ref_path.relative_to(REPO)),reference_report_sha256=sha(ref_path),
        explicit_warmup_forwards=0,first_sample_included=True,
        timer_start='before first raw video file preparation',timer_end='200th synchronized CPU quality score',
        expected_head_sha256=ref['checkpoint_sha256'],command=sys.argv)
    def status(state,**kw):
        legacy.frozen.write_json(output/'status.json',dict(status=state,time=time.time(),pid=os.getpid(),samples=len(records),**kw))
    class Guard(legacy.engine.MPSGuard):
        def check(self):
            if source_map()!=bound:raise RuntimeError('Pinned E2E source changed')
            super().check()
    try:
        import torch
        from . import quality_token_common as core
        guard=Guard(args);guard.check();torch.set_num_threads(8)
        limit=float(subprocess.check_output(['nvidia-smi','--id='+args.gpu_uuid,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
        if abs(limit-config()['profiles'][args.profile]['power_limit_w'])>.1:raise RuntimeError('Unexpected power cap; do not change it')
        identity.update(power_limit_w=limit,clock_offset_s=time.perf_counter()-time.monotonic())
        if abs(identity['clock_offset_s'])>1e-4:raise RuntimeError('Host and NVML monotonic clocks differ')
        # Do not hash video bytes before T0: that would explicitly warm the input cache.
        input_metadata=file_identities(rows);identity['input_files_metadata_before']=input_metadata
        identity['cache_protocol']='No video-byte prehash, decode preloading, cache flush or dummy forward before T0; content SHA after T1'
        status('loading')
        with torch.inference_mode():
            all_rows,prepare,transfer,forward,start,finish=legacy.runtime(args,guard,identity)
            if ordered_id_sha(all_rows[:200])!=ordered_id_sha(rows):raise RuntimeError('Runtime input order differs')
            visible=torch.cuda.get_device_properties(0).multi_processor_count
            props=torch.cuda.get_device_properties(0)
            if torch.cuda.device_count()!=1 or 'A100' not in props.name:
                raise RuntimeError('Only one approved A100 may be visible')
            if args.profile=='rental' and props.total_memory<70*1024**3:
                raise RuntimeError('Full DS/Kimi rental requires a single 80GB A100')
            if visible!=(108 if args.cap==100 else 54):raise RuntimeError('MPS resource limit not effective')
            if (identity['assets']!=ref['assets'] or identity['floating_parameter_dtypes']!=ref['floating_parameter_dtypes']
                    or identity['quality_readout_dtypes']!=ref['quality_readout_dtypes']):
                raise RuntimeError('Official assets or deployment precision differ')
            identity.update(visible_multiprocessors=visible,gpu_name=torch.cuda.get_device_name(0),
                original_full558_reference=ref['performance'],reference_is_not_new200_accuracy=True)
            import importlib.metadata
            versions={}
            for package in ('torch','transformers','numpy','opencv-python','ultralytics','xformers'):
                try:versions[package]=importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:versions[package]=None
            identity['environment']=dict(python=sys.version,packages=versions,torch_cuda=torch.version.cuda,
                cpu_threads=torch.get_num_threads(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
            legacy.frozen.write_json(output/'identity.json',identity)
            sampler=legacy.ProcessTelemetrySampler(args.gpu_uuid,output/'telemetry_stdout.jsonl')
            sampler.start();sampler.set_phase('e2e_no_warmup');time.sleep(.15)  # CPU observer startup, no inference.
            def check():guard.fast();sampler.check()
            # All input, embedding, compute and actual inter-request work is charged.
            collect(rows,prepare,transfer,forward,start,torch.cuda.synchronize,check,
                progress=lambda n:status('evaluating',completed_samples=n),records=records,
                input_observer=lambda row,cpu:input_evidence.append(dict(sample_id=row['sample_id'],**cpu_input_evidence(cpu))))
            sampler.request('barrier');time.sleep(.12)  # Outside T1: bracket final output with an observed power sample.
            samples=sampler.stop();sampler=None;legacy.telemetry.save_samples(output/'telemetry.csv',samples)
            cost=summarize(records,samples)
            predictions=[r['prediction'] for r in records]
            import numpy as np
            performance=core.metrics(np.array([r['target_mos'] for r in records]),np.array(predictions))
            with ref_path.with_name('calls.csv').open(newline='',encoding='utf-8-sig') as f:
                original=list(csv.DictReader(f))[:200]
            if any(a['sample_id']!=b['sample_id'] or json.loads(a['frame_positions'])!=json.loads(b['frame_positions'])
                    for a,b in zip(records,original)):
                raise RuntimeError('Original per-video frame selection changed')
            finish();finish=None
            input_hashes=file_identities(rows,hash_content=True)
            if file_identities(rows)!=input_metadata:raise RuntimeError('Raw input metadata changed during measurement')
            identity['input_files']=input_hashes
            legacy.frozen.write_json(output/'input_audit.json',dict(samples=input_evidence,
                original_video_metadata=input_video_metadata(rows),
                original_metadata_read_after_T1=True,processor_metadata_source='actual CPU prepare output',
                cpu_metadata_bookkeeping_in_inter_request_gaps=True,no_extra_GPU_audit_operators=True))
            identity['input_audit_sha256']=sha(output/'input_audit.json')
            identity['effective_input_tokens200']=sum(v.get('effective_input_tokens',0) for v in input_evidence)
            guard.check()
            report=dict(status='complete',**identity,cost=cost,performance200=performance,
                measured_videos=200,reference_test_videos=558,full558_reference_is_historical=True,
                quality_policy='Measured200 separately; archived full558 MPS100 reference not relabelled as a new measurement',
                same_run_shared_qwen_temporal_mps100=args.model=='qwen' and args.cap==100,
                power_is_board_measurement_plus_fixed_host_estimate=True)
            legacy.frozen.write_json(output/'report.json',report);status('complete',completed_samples=200)
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        if sampler is not None:
            legacy.telemetry.save_samples(output/'telemetry.csv',sampler.stop(raise_on_error=False))
        try:
            if finish is not None:finish()
        finally:
            legacy.write_csv(output/'calls.csv',records)
            if not (output/'input_audit.json').exists():
                legacy.frozen.write_json(output/'input_audit_partial.json',dict(samples=input_evidence))
            legacy.frozen.write_json(output/'identity.json',identity)
            if guard is not None:guard.close(output/'process_audit.json')


def suite(args):
    import fcntl
    points=plan(args.profile,args.models,args.caps,args.repeats)
    root=validate_output(args.output);rows=input_rows(args.manifest)
    for model in {m for m,c,r in points}:reference(model)
    legacy=initialize_profile(args)
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        root.mkdir(parents=True);active=False
        state=dict(status='admission',pid=os.getpid(),protocol=config(),sources=source_map(),
            deadline=args.deadline,plan=points,completed=[],first200_ids_sha256=ordered_id_sha(rows))
        def save():legacy.frozen.write_json(root/'suite_status.json',state)
        save()
        try:
            for _ in range(3):
                if legacy.lab.shared.guard_records(args.gpu_uuid):raise RuntimeError('Foreign GPU process; no automatic retry')
                legacy.validate_idle_memory(args.gpu_uuid)
                if time.time()>=args.deadline:raise TimeoutError('Finite suite deadline')
                time.sleep(10)
            legacy.engine.daemon(True);active=True
            for model,cap,repeat in points:
                if time.time()>args.deadline-120:raise TimeoutError('No time for another worker')
                if source_map()!=state['sources']:raise RuntimeError('Suite source changed')
                if args.profile=='lab':mp,pt,python,receipt=legacy.lab.locations(model)
                else:
                    mp=Path('/data/models')/model;pt=legacy.engine.base.old.RUN/model/'train/best_checkpoint.pt'
                    python=f'/data/venvs/{model}/bin/python';receipt=None
                name=f'{model}_cap{cap}_repeat{repeat:02d}';output=root/name
                if output.exists():raise FileExistsError('Preserve prior worker; do not retry blindly')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,CUDA_MPS_PIPE_DIRECTORY=args.pipe,
                    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap),OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                command=[python,'-u','-m',__package__+'.measure_mps_e2e_20261024','--phase','worker',
                    '--profile',args.profile,'--model',model,'--cap',str(cap),'--repeat',str(repeat),
                    '--model-path',str(mp),'--checkpoint',str(pt),'--manifest',str(args.manifest),
                    '--output',str(output),'--suite-root',str(root),'--pipe',args.pipe,
                    '--gpu-uuid',args.gpu_uuid,'--deadline',str(args.deadline)]
                if receipt:command.extend(['--reproduction-receipt',str(receipt)])
                legacy.frozen.write_json(root/(name+'.command.json'),command)
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='running',phase=name,child_pid=child.pid);save()
                    try:code=child.wait(timeout=min(1800,max(1,args.deadline-time.time())))
                    except BaseException:
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise
                if code:raise RuntimeError(f'{name} exit{code}; preserve partial evidence and stop')
                report=read(output/'report.json')
                if report['status']!='complete' or report['cost']['videos']!=200:raise RuntimeError('Incomplete child report')
                audit=audit_point(output,args.manifest)
                legacy.frozen.write_json(output/'cpu_audit.json',audit)
                state['completed'].append(name);save()
            state.update(status='complete',finished=time.time())
        except BaseException as error:state.update(status='stopped',error=repr(error));raise
        finally:
            if active:
                try:legacy.engine.daemon(False);state['mps_released']=True
                except BaseException as error:state['release_error']=repr(error)
            save()


def audit_point(folder,manifest):
    """Independent timestamp/power/identity replay; no CUDA or historical anchor."""
    from .batch_telemetry import Sample
    from .reports.strict_tf_resource_audit_20261020 import observed_energy
    from scipy.stats import spearmanr,pearsonr
    folder=Path(folder);report=read(folder/'report.json');state=read(folder/'status.json')
    expected=input_rows(manifest);_,ref=reference(report['model'])
    if report['status']!='complete' or state['status']!='complete' or state['samples']!=200:
        raise ValueError('Completed original 200-video run required')
    with (folder/'calls.csv').open(newline='',encoding='utf-8-sig') as f:calls=list(csv.DictReader(f))
    with (folder/'telemetry.csv').open(newline='',encoding='utf-8-sig') as f:raw=list(csv.DictReader(f))
    if [r['sample_id'] for r in calls]!=[r['sample_id'] for r in expected]:raise ValueError('Sample IDs changed')
    if report['checkpoint_sha256']!=config()['references'][report['model']]['head_sha256'] or report['assets']!=ref['assets']:
        raise ValueError('Wrong model identity')
    if (report['manifest_sha256']!=config()['manifest_sha256'] or report['explicit_warmup_forwards']!=0
            or report['batch']!=1 or report['visible_multiprocessors']!=(108 if report['cap']==100 else 54)):
        raise ValueError('Wrong MPS or sampling contract')
    if set(report['input_files'])!={r['sample_id'] for r in expected} or any(len(v['sha256'])!=64 for v in report['input_files'].values()):
        raise ValueError('Raw input content identities incomplete')
    evidence=read(folder/'input_audit.json')
    if sha(folder/'input_audit.json')!=report['input_audit_sha256'] or [v['sample_id'] for v in evidence['samples']]!=[r['sample_id'] for r in expected]:
        raise ValueError('Input metadata audit changed or incomplete')
    if sum(v.get('effective_input_tokens',0) for v in evidence['samples'])!=report['effective_input_tokens200']:
        raise ValueError('Shared token numerator mismatch')
    samples=[Sample(float(r['host_time_unix']),float(r['host_monotonic_s']),r['phase'],float(r['watts']),
        float(r['utilization_percent']),float(r['memory_used_mib'])) for r in raw]
    value=summarize(calls,samples)
    def close(a,b):
        if not math.isclose(float(a),float(b),abs_tol=1e-6,rel_tol=1e-9):raise ValueError('Independent replay mismatch')
    for key in ('continuous_wall_s','ms_per_video','equivalent_ms_per_input_frame','estimated_joules_per_video',
                'estimated_joules_per_input_frame','gpu_energy_j'):
        close(value[key],report['cost'][key])
    board=observed_energy([(float(calls[0]['preprocess_start']),float(calls[-1]['call_return']))],raw)
    close(board,value['gpu_energy_j'])
    targets=[float(r['target_mos']) for r in calls];predictions=[float(r['prediction']) for r in calls]
    if targets!=[float(r['temporal']) for r in expected]:raise ValueError('TEST targets changed')
    close(spearmanr(targets,predictions).statistic,report['performance200']['srcc'])
    close(pearsonr(targets,predictions).statistic,report['performance200']['plcc'])
    for name,digest in report['sources'].items():
        blob=subprocess.check_output(['git','show',report['git_commit']+':'+name.replace('\\','/')],cwd=REPO)
        if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('Source not bound to published Git')
    for event in read(folder/'process_audit.json'):
        allowed={str(report['pid'])}|{str(v) for v in event['servers']}
        if any(str(v[0]).strip() not in allowed or str(v[1]).strip()!=report['gpu_uuid'] for v in event['processes']):
            raise ValueError('Foreign GPU process in measurement evidence')
    return dict(status='passed',protocol_id=PROTOCOL,model=report['model'],cap=report['cap'],repeat=report['repeat'],
        report_sha256=sha(folder/'report.json'),calls_sha256=sha(folder/'calls.csv'),power_sha256=sha(folder/'telemetry.csv'),
        first200_ids_sha256=ordered_id_sha(expected),measured_videos=200,measured_frames=800,
        independent_whole_window_energy=True,all_frontend_and_internal_gaps_included=True,
        no_historical_cost_override=True,cost=value)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('validate','suite','worker','audit'),required=True)
    p.add_argument('--profile',choices=('lab','rental'),default='lab')
    p.add_argument('--models',nargs='+');p.add_argument('--caps',type=int,nargs='+');p.add_argument('--repeats',type=int,nargs='+')
    p.add_argument('--model',choices=tuple(config()['references']));p.add_argument('--cap',type=int,choices=(100,50));p.add_argument('--repeat',type=int,choices=(1,2,3))
    p.add_argument('--model-path',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--reproduction-receipt',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,default=CANONICAL)
    p.add_argument('--suite-root',type=Path,default=CANONICAL);p.add_argument('--pipe');p.add_argument('--gpu-uuid')
    p.add_argument('--deadline',type=float);a=p.parse_args()
    a.output=a.output.resolve();a.suite_root=a.suite_root.resolve()
    if a.phase=='suite':a.suite_root=a.output
    c=config();a.gpu_uuid=a.gpu_uuid or c['profiles'][a.profile]['gpu_uuid']
    if a.profile=='lab' and a.gpu_uuid!=c['profiles']['lab']['gpu_uuid']:raise ValueError('Only approved physical lab A100')
    if a.profile=='rental' and not a.gpu_uuid and a.phase not in ('validate','audit'):
        raise ValueError('New explicitly authorized rental GPU UUID required')
    a.pipe=a.pipe or ('/DATA/DATA1/guest3/.cache/t06_mps_e2e_20261024' if a.profile=='lab' else '/data/tmp/t06_mps_e2e_20261024')
    if a.phase=='validate':
        rows=input_rows(a.manifest);points=plan(a.profile,a.models,a.caps,a.repeats)
        for m in {m for m,c,r in points}:reference(m)
        print(json.dumps(dict(status='CPU_contract_validated',points=len(points),videos=200,frames=800,
            first200_ids_sha256=ordered_id_sha(rows),gpu_initialized=False)));return
    if a.phase=='audit':
        print(json.dumps(audit_point(a.output,a.manifest),indent=2));return
    if a.deadline is None or not 0<a.deadline-time.time()<=28800:raise ValueError('Finite deadline, at most eight hours')
    if a.phase=='worker' and (a.model not in c['profiles'][a.profile]['models'] or a.cap is None or a.repeat is None):
        raise ValueError('Exact approved worker point required')
    (suite if a.phase=='suite' else worker)(a)


if __name__=='__main__':main()
