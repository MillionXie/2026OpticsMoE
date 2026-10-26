"""One fresh 2048-call LG async round, only after the lab baseline truly exits.

No CLI invocation is made by importing this file. The suite waits CPU-only for
the explicit completed baseline/PID/private-MPS release before taking the shared
lock, then retains the original 12-idle/384-MiB admission and exclusive guard.
No automatic retry, source deployment, baseline mutation or foreign signal.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
SIM=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation'
ENTRY=Path(__file__).resolve()
PYTHON='/home/guest3/miniconda3/envs/xml/bin/python'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def validate_args(args):
    args.root=args.root.resolve()
    if args.root.parent!=SIM or not re.fullmatch(r'a100_lgvq_async_20261026_\d{2}',args.root.name):
        raise ValueError('New canonical LG-only suite root required')
    if not math.isfinite(args.deadline) or not 0<args.deadline-time.time()<=36000:
        raise ValueError('Positive finite safety deadline, at most ten hours, required')
    args.pipe='/DATA/DATA1/guest3/.cache/'+args.root.name
    if args.phase=='suite':
        if args.after_suite is None:raise ValueError('Explicit baseline predecessor required')
        args.after_suite=args.after_suite.resolve()
        if args.after_suite.parent!=SIM or not args.after_suite.name.startswith('a100_lab_corrected_energy_20261026_'):
            raise ValueError('Explicit canonical baseline predecessor required')
    else:
        if args.output is None:raise ValueError('Exact worker leaf required')
        args.output=args.output.resolve()
        if args.output!=args.root/'lg_repeat01':raise ValueError('Exact single fresh LG leaf required')


def predecessor_complete(state):
    if state.get('status')!='complete':return False
    plan=state.get('plan',[]);completed=state.get('completed',[]);retained=state.get('retained',[])
    retained_names=[item['name'] for item in retained]
    try:planned={f'{model}_cap{cap}_repeat{repeat:02d}' for model,cap,repeat in plan}
    except (TypeError,ValueError):return False
    expected={f'{model}_cap{cap}_repeat{repeat:02d}' for model in ('qwen','glm','clip','yolo')
              for cap in (100,50) for repeat in (1,2,3)}
    return (set(completed)==planned and len(completed)==len(plan)==len(planned)
            and set(retained_names)|set(completed)==expected
            and len(set(retained_names))==len(retained_names)
            and not set(retained_names)&set(completed))


def original_pid_alive(pid,root):
    process=Path('/proc')/str(int(pid))
    try:
        state=(process/'stat').read_text().rsplit(')',1)[1].split()[0]
        command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        return state!='Z' and str(root) in command
    except FileNotFoundError:return False


def source_map():
    import lgvq_async_worker_20261026 as candidate
    result=candidate.source_map()
    paths=[ENTRY,HERE/'run_corrected_boundaries_20261026.py',HERE/'qwen_shared_serial_20261026.py',
           HERE/'qwen_strict_serial_20261026.py',HERE/'run_temporal_remainder.py',
           HERE/'first_tf_supplement_20261025.py',HERE/'lab_corrected_energy_triples_20261026.py',
           REPO/'LightGenV2/tasks/t06_video_quality_assessment/rental_busy_resource_20261020.py']
    result.update({p.relative_to(REPO).as_posix():sha(p) for p in paths})
    return result


def verify_published_candidates(commit,sources):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Exact published commit required')
    for name,digest in sources.items():
        if name==ENTRY.relative_to(REPO).as_posix() or Path(name).name.startswith('lgvq_async_'):
            data=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO,timeout=10)
            if hashlib.sha256(data).hexdigest()!=digest:
                raise ValueError('Candidate does not match its published Git blob: '+name)


def control(pipe,command):
    return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',text=True,capture_output=True,
                          env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=pipe),timeout=5)


def suite_command(argv):
    return (any(Path(arg).name==ENTRY.name for arg in argv) and '--phase' in argv
            and argv.index('--phase')+1<len(argv) and argv[argv.index('--phase')+1]=='suite')


def assert_unique_controller():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.geteuid():continue
            argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if suite_command(argv):raise RuntimeError('Existing LG async controller; inspect it instead of creating a duplicate')
        except (FileNotFoundError,ProcessLookupError):continue


def child_identity_matches(argv,uid,pgid,pid,leaf,own_uid):
    def value(flag):return argv[argv.index(flag)+1] if flag in argv and argv.index(flag)+1<len(argv) else None
    return (uid==own_uid and pgid==pid and str(ENTRY) in argv
            and value('--phase')=='worker' and value('--output')==str(leaf))


def accepted_report(report):
    return (report.get('status')=='complete' and report.get('qualified_for_final_hybrid_table') is True
            and report.get('video_calls')==2048 and report.get('fields')==128
            and report.get('all_predictions_bitwise_equal') is True
            and report.get('physical_hardware_time_measured') is False
            and report.get('fixed_optics_projection',{}).get('status')=='COUNTERFACTUAL_FIXED_OPTICS_PROJECTION')


def checked_stop_child(child,leaf):
    if child is None or child.poll() is not None:return
    proc=Path('/proc')/str(child.pid)
    argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
    if not child_identity_matches(argv,proc.stat().st_uid,os.getpgid(child.pid),child.pid,leaf,os.geteuid()):
        raise RuntimeError('Refuse signal to an unverified worker')
    os.killpg(child.pid,signal.SIGINT);child.wait(timeout=60)


def worker(args):
    import torch
    import run_corrected_boundaries_20261026 as original
    import qwen_shared_serial_20261026 as shared
    import lgvq_async_worker_20261026 as candidate
    receipt=args.root/'source_receipt.json'
    if sha(receipt)!=args.source_receipt_sha256:raise ValueError('Source receipt changed')
    bound=json.loads(receipt.read_text())
    if source_map()!=bound['sources']:raise ValueError('Source changed before loading')
    verify_published_candidates(bound['published_commit'],bound['sources'])
    if os.environ.get('MEASUREMENT_PUBLISHED_COMMIT')!=bound['published_commit']:
        raise ValueError('Published identity differs')
    original.base.GPU=candidate.GPU;original.base.PIPE=args.pipe
    if args.output.exists():raise FileExistsError('Preserve prior worker leaf')
    args.output.mkdir();guard=None
    write(args.output/'identity.json',dict(published_commit=bound['published_commit'],sources=bound['sources'],
        pid=os.getpid(),GPU=candidate.GPU,cap=100,video_calls=2048,repeat=1,
        command=sys.argv,private_MPS_pipe=args.pipe))
    def status(value,**details):write(args.output/'status.json',dict(status=value,pid=os.getpid(),unix=time.time(),**details))
    class Guard(original.ExclusiveGuard):
        def check(self):
            try:
                super().check()
                if source_map()!=bound['sources']:raise RuntimeError('Frozen source changed')
            except BaseException as error:
                write(args.output/'guard_rejection_snapshot.json',dict(error=repr(error),
                      snapshot=self.audit[-1] if self.audit else None))
                raise
    try:
        status('loading_original_LG_candidate')
        guard=Guard(args.deadline,100);torch.set_num_threads(8)
        report=candidate.worker(args,guard,2048)
        if not accepted_report(report):
            raise RuntimeError('Raw software completion is not an accepted fixed-optics report')
        guard.check();status('complete',time16_ms=report['time16_ms'],energy16_j=report['energy16_j'])
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        if guard is not None:guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    import lgvq_async_worker_20261026 as candidate
    from first_tf_supplement_20261025 import serialize_mps_commands
    import lab_corrected_energy_triples_20261026 as admission
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    import qwen_shared_serial_20261026 as shared
    serialize_mps_commands()
    assert_unique_controller()
    if args.root.exists():raise FileExistsError('Preserve existing suite; inspect its actual PID instead')
    sources=source_map();commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT']
    verify_published_candidates(commit,sources)
    args.root.mkdir()
    write(args.root/'source_receipt.json',dict(published_commit=commit,sources=sources))
    receipt_sha=sha(args.root/'source_receipt.json')
    state=dict(status='waiting_previous_baseline',pid=os.getpid(),deadline=args.deadline,
               after_suite=str(args.after_suite),plan=['lg_repeat01'],completed=[],sources=sources)
    child=None;active=False;leaf=args.root/'lg_repeat01';lock=None
    save=lambda:write(args.root/'suite_status.json',state)
    def bounded():
        if time.time()>=args.deadline:raise TimeoutError('Finite LG safety deadline; preserve all existing data')
        if source_map()!=sources:raise RuntimeError('Frozen source changed while queued/running')
    save()
    try:
        while True:
            bounded()
            previous=json.loads((args.after_suite/'suite_status.json').read_text())
            if predecessor_complete(previous) and not original_pid_alive(previous['pid'],args.after_suite):
                pipe='/DATA/DATA1/guest3/.cache/'+args.after_suite.name
                if control(pipe,'get_server_list').returncode!=0:break
            state.update(status='waiting_previous_baseline',previous_status=previous['status']);save();time.sleep(5)
        lock=(SIM/'.temporal_telemetry_a100.lock').open('a')
        while True:
            bounded()
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                state.update(status='waiting_shared_lock');save();time.sleep(5)
        idle=0
        while idle<12:
            bounded();snapshot=admission.admission_snapshot(args.pipe)
            memory=float(subprocess.check_output(['nvidia-smi','-i',candidate.GPU,
                '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=5))
            if not math.isfinite(memory) or memory<0:raise ValueError('Invalid original GPU memory observation')
            if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients'] or memory>=384:
                idle=0;state.update(status='waiting_foreign_or_memory',admission_observation=snapshot,memory_mib=memory)
                save();time.sleep(5);continue
            idle+=1;time.sleep(.25)
        if control(args.pipe,'get_server_list').returncode==0:raise RuntimeError('Unexpected existing private MPS daemon')
        engine.GPU=candidate.GPU;engine.PIPE=args.pipe;engine.ROOT=args.root
        shared.base.GPU=candidate.GPU;shared.base.PIPE=args.pipe
        engine.daemon(True);active=True
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=candidate.GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                 CUDA_MPS_PIPE_DIRECTORY=args.pipe,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
                 OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        command=[PYTHON,'-u',str(ENTRY),'--phase','worker','--root',str(args.root),'--output',str(leaf),
                 '--deadline',str(args.deadline),'--source-receipt-sha256',receipt_sha]
        if args.optimize_runtime:
            env['CUDA_MODULE_LOADING']='EAGER'
            command.append('--optimize-runtime')
        write(args.root/'lg_repeat01.command.json',command)
        with (args.root/'lg_repeat01.log').open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',phase='lg_repeat01',child=child.pid);save()
            while child.poll() is None:bounded();time.sleep(1)
        if child.returncode!=0:raise RuntimeError('Preserve failed leaf; no automatic retry')
        report=json.loads((leaf/'cost_report.json').read_text())
        if not accepted_report(report):raise RuntimeError('Projected DAG report is not accepted')
        state.update(status='complete',completed=['lg_repeat01'],finished=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error))
        checked_stop_child(child,leaf)
        raise
    finally:
        try:
            if active:
                shared.stop_private_daemon()
                state['mps_released']=control(args.pipe,'get_server_list').returncode!=0
                if not state['mps_released']:raise RuntimeError('Own private MPS daemon did not release')
        except BaseException as error:
            state.update(status='stopped',cleanup_error=repr(error),mps_released=False)
            raise
        finally:
            save()
            if lock is not None:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['suite','worker'],required=True)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path)
    parser.add_argument('--after-suite',type=Path);parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--source-receipt-sha256')
    parser.add_argument('--optimize-runtime',action='store_true',
                        help='Fresh candidate: EAGER module loading, no forced GC/cache purge, detailed electronic phases')
    args=parser.parse_args();validate_args(args)
    (suite if args.phase=='suite' else worker)(args)


if __name__=='__main__':main()
