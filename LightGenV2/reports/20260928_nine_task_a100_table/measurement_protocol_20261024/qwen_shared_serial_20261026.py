"""Human-authorized NON-EXCLUSIVE urgent Qwen50 measurement.

Reuse strict one-video CPU/H2D/GPU/CPU-return unchanged. Other GPU applications
are OBSERVED, not killed or used as a reason to discard this exploratory run.
These times must not be presented as uncontended baseline performance or as
evidence of a hardware speedup relative to an uncontended LG measurement.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import qwen_strict_serial_20261026 as strict

base=strict.base
STRICT_FILE=Path(strict.__file__)
previous_sources=base.source_map
base.__file__=__file__
base.PIPE='/DATA/DATA1/guest3/.cache/qwen_shared_serial_20261026'


def sources():
    result=previous_sources();result[STRICT_FILE.relative_to(base.REPO).as_posix()]=base.sha(STRICT_FILE)
    return result


base.source_map=sources


def validate_shared_snapshot(snapshot,worker):
    # Only our own private MPS clients may use this private pipe. Ordinary
    # applications on the same physical GPU are allowed by the new user scope.
    if set(snapshot['clients'])-{worker}:raise RuntimeError('Unexpected client in own private MPS')
    return snapshot['foreign_gpu_processes']


class SharedGuard:
    def __init__(self,deadline,cap):
        self.deadline=deadline;self.cap=cap;self.audit=[];self.error=None
        self.stop=threading.Event();self.thread=None;self.pid=os.getpid();self.lock=threading.Lock();self.bound=False
        self.check()
    def check(self):
        from run_temporal_remainder import process_snapshot
        if time.time()>=self.deadline:raise TimeoutError('Finite non-exclusive worker')
        with self.lock:
            row=process_snapshot(base.GPU,self.pid,base.PIPE)
            validate_shared_snapshot(row,self.pid)
            if self.bound and self.pid not in row['clients']:raise RuntimeError('Own private MPS client disappeared')
            raw=subprocess.check_output(['nvidia-smi','-i',base.GPU,'--query-gpu=uuid,name,power.limit,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
            uuid,name,watts,util,memory=[x.strip() for x in raw.strip().split(',')]
            assert uuid==base.GPU and name=='NVIDIA A100-PCIE-40GB' and float(watts)==250
            row.update(platform=dict(uuid=uuid,name=name,power_limit_w=float(watts),requested_cap=self.cap),
                global_GPU_utilization_percent=float(util),global_memory_used_mib=float(memory),
                human_authorized_nonexclusive=True)
            self.audit.append(row)
    def bind(self):
        self.bound=True;self.check()
        def watch():
            while not self.stop.wait(.5):
                try:self.check()
                except BaseException as e:self.error=e;return
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
    def fast(self):
        if self.error:raise RuntimeError('Shared-run identity/private-MPS guard failed') from self.error
        if time.time()>=self.deadline:raise TimeoutError('Finite shared-run deadline')
    def close(self,path):
        self.stop.set()
        if self.thread:self.thread.join(10)
        base.write(path,self.audit)


base.model_guard=SharedGuard


def worker(args):
    strict.worker(args)
    p=args.output/'identity.json';identity=json.loads(p.read_text())
    identity.update(profile='Qwen50_strict_single_video_NONEXCLUSIVE_512x4_20261026_v1',
        nonexclusive_authorized=True,foreign_GPU_processes_observed_not_rejected=True,
        not_uncontended_baseline_performance=True,
        private_mps_pipe=base.PIPE,
        GPU_50_percent_is_requested_MPS_cap_not_guaranteed_physical_share=True)
    base.write(p,identity)


def control(command):
    return subprocess.run(['nvidia-cuda-mps-control'],input=command+'\n',text=True,capture_output=True,
        env=dict(os.environ,CUDA_MPS_PIPE_DIRECTORY=base.PIPE),timeout=5)


def stop_private_daemon():
    from LightGenV2.tasks.t06_video_quality_assessment.rental_busy_resource_20261020 import numeric_lines
    state=control('get_server_list')
    if state.returncode:return
    for server in numeric_lines(state.stdout):
        clients=control('get_client_list '+str(server));assert clients.returncode==0
        if numeric_lines(clients.stdout):raise RuntimeError('Refuse shutdown with active private client')
    ended=control('quit');assert ended.returncode==0,ended.stderr
    # Do NOT wait for other users' GPU applications to exit.
    for _ in range(10):
        if control('get_server_list').returncode:return
        time.sleep(.2)
    raise RuntimeError('Own private MPS daemon did not release')


def suite(args):
    import fcntl
    from first_tf_supplement_20261025 import serialize_mps_commands
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from run_temporal_remainder import process_snapshot
    serialize_mps_commands();engine.GPU=base.GPU;engine.PIPE=base.PIPE;engine.ROOT=args.output
    assert not args.output.exists() and args.output.is_relative_to(base.ROOT) and 0<args.deadline-time.time()<=10800
    args.output.mkdir();state=dict(status='admission_nonexclusive',pid=os.getpid(),deadline=args.deadline,completed=[],
        human_authorized_nonexclusive=True,original_Qwen_single_video_execution_unchanged=True)
    active=False;child=None
    def save():base.write(args.output/'suite_status.json',state)
    def bounded():
        if time.time()>args.deadline-60:raise TimeoutError('Finite non-exclusive measurement')
    save()
    try:
        with (base.ROOT/'.temporal_telemetry_a100.lock').open('a') as lock:
            while True:
                bounded()
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(2)
            prior=json.loads((args.previous/'suite_status.json').read_text())
            assert prior['status']=='stopped' and prior['completed']==[] and not Path('/proc',str(prior['pid'])).exists()
            state['stopped_exclusive_predecessor']=str(args.previous);save()
            for rep in (1,2,3):
                bounded()
                # No idle-utilization wait: real free memory only, no dummy load.
                free=float(subprocess.check_output(['nvidia-smi','-i',base.GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                if free<8192:raise RuntimeError('Insufficient free memory for safe native Qwen load')
                assert control('get_server_list').returncode!=0,'Private MPS already active'
                engine.daemon(True);active=True;leaf=args.output/f'qwen_repeat{rep:02d}'
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_MPS_PIPE_DIRECTORY=base.PIPE,
                    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='50',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
                command=['/home/guest3/miniconda3/envs/xml/bin/python','-u',str(Path(__file__)),'--phase','worker',
                    '--system','qwen','--repeat',str(rep),'--output',str(leaf),'--deadline',str(args.deadline)]
                observations=[]
                try:
                    with leaf.with_suffix('.log').open('x') as log:
                        child=subprocess.Popen(command,cwd=base.REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        state.update(status='running_nonexclusive',repeat=rep,child=child.pid,free_memory_mib_before_loading=free);save()
                        while child.poll() is None:
                            bounded();snap=process_snapshot(base.GPU,child.pid,base.PIPE)
                            validate_shared_snapshot(snap,child.pid);observations.append(snap);time.sleep(2)
                except BaseException:
                    if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT);child.wait(timeout=45)
                    raise
                finally:base.write(args.output/f'qwen_repeat{rep:02d}_loading_and_process_guard.json',observations)
                stop_private_daemon();active=False
                if child.returncode:raise RuntimeError('Worker failed; preserve evidence, do not blindly restart')
                state['completed'].append(leaf.name);save()
            state.update(status='complete',finished=time.time())
    except BaseException as e:state.update(status='stopped',error=repr(e));raise
    finally:
        if active:
            try:stop_private_daemon();state['mps_released']=True
            except BaseException as e:state['mps_release_error']=repr(e)
        else:state['mps_released']=True
        save()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['suite','worker'],required=True)
    p.add_argument('--system',default='qwen');p.add_argument('--repeat',type=int);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--previous',type=Path)
    a=p.parse_args();a.output=a.output.resolve();(suite if a.phase=='suite' else worker)(a)
