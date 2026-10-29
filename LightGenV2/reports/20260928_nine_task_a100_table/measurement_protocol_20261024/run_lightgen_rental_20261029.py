"""Same native LightGen-II resize-inclusive computation on the new A10080.

One original512-video trace for the requested approximate software runtime.
Report actual FFT software wall/board energy separately from the fixed-optics
schedule estimate. No physical optical hardware or measured hybrid power.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import run_lgvq_resize_norm_20261029 as entry
import rental_power_observer_20261029 as power
import git_blob_identity_20261029 as identity
import lightgen_rental_front_20261029 as frontend_storage

runtime=entry.runtime;REPO=runtime.REPO;SIM=runtime.SIM
ENTRY=Path(__file__).resolve();ROOT=SIM/'lightgen_rental_20261029_01'
GPU=power.GPU;PIPE='/data/tmp/lightgen_rental_20261029_01'
PYTHON='/root/data/lgvq_20261029/venvs/lightgen/bin/python'
NATIVE_SOURCES=entry.sources;NATIVE_WRITE=runtime.write

def validate_hardware(actual):
    fields=[s.strip() for s in actual.strip().split(',')]
    if len(fields)!=3 or fields[:2]!=[GPU,'NVIDIA A100-SXM4-80GB'] or float(fields[2])!=400:
        raise ValueError('Exact original rental GPU400W required')
    return dict(gpu_uuid=fields[0],name=fields[1],power_limit_w=400)

def sources():
    return {**NATIVE_SOURCES(),**{p.relative_to(REPO).as_posix():runtime.sha(p)
        for p in (ENTRY,Path(power.__file__),Path(frontend_storage.__file__))}}

def write(path,value):
    if path.name=='compute_identity.json':
        value=dict(value,profile='LG_RENTAL80_SAME_NATIVE_RESIZE_NORMALIZATION_SOFTWARE_TRACE',
                   gpu_uuid=GPU,power_limit_W=400,CPU_resize_and_normalization_included=True,
                   physical_optical_hardware_measured=False,hybrid_deployment_energy_qualified=False)
    entry.write(path,value)

class RentalGuard:
    def __init__(self,deadline,cap):
        from first_tf_supplement_20261025 import serialize_mps_commands
        from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
        serialize_mps_commands();engine.GPU=GPU;engine.PIPE=PIPE;engine.ROOT=ROOT
        self.guard=engine.MPSGuard(SimpleNamespace(deadline=deadline));self.audit=self.guard.audit
        if cap!=100:raise ValueError('LightGen rental uses actual108SM only')
        actual=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=uuid,name,power.limit','--format=csv,noheader,nounits'],text=True,timeout=5)
        validate_hardware(actual)
        self.guard.check()
    def bind(self):return self.guard.bind()
    def fast(self):return self.guard.fast()
    def check(self):return self.guard.check()
    def close(self,path):return self.guard.close(path)

def configure():
    import cv2
    cv2.setNumThreads(1)
    import run_corrected_boundaries_20261026 as original
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as observer
    runtime.GPU=GPU;runtime.PYTHON=PYTHON;runtime.ENTRY=ENTRY;runtime.sources=sources;runtime.write=write
    runtime.verify=lambda c,s:identity.verify(c,s,REPO)
    entry.scope.prepare_cpu_worker=entry.preprocessing.prepare_cpu_worker
    entry.scope.prepare_video=entry.preprocessing.prepare_video
    entry.scope.patch_video=entry.preprocessing.patch_video
    entry.scope.summarize_compute=entry.preprocessing.summarize_compute
    original.ExclusiveGuard=RentalGuard;observer.observer_command=power.observer_command
    if os.environ.get('LIGHTGEN_FRONT_COMPONENT_RECEIPT_SHA256'):frontend_storage.install()

def suite(a):
    import fcntl
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as old
    predecessor=SIM/'rental_energy_remeasure_20261029_01'
    state=json.loads((predecessor/'suite_status.json').read_text())
    if state['status']!='complete' or len(state['completed'])!=12 or not state.get('mps_released'):
        raise ValueError('All twelve new time+energy points must complete first')
    if runtime.alive(state['pid']):raise ValueError('Predecessor controller still alive')
    if ROOT.exists():raise FileExistsError('Preserve original LightGen attempt')
    frozen=sources();commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'];runtime.verify(commit,frozen)
    lock=(SIM/'.temporal_telemetry_a100.lock').open('a');active=False;child=None
    record=dict(status='admission',pid=os.getpid(),deadline=a.deadline,completed=[],plan=['lg_repeat01'],gpu_uuid=GPU,
                purpose='one complete512-video approximate software runtime, not three-round inference statistics')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ROOT.mkdir();runtime.write(ROOT/'source_receipt.json',dict(published_commit=commit,sources=frozen))
        source_sha=runtime.sha(ROOT/'source_receipt.json')
        for _ in range(3):
            if old.initialize_profile(SimpleNamespace(profile='rental',gpu_uuid=GPU,pipe=PIPE,suite_root=ROOT)).engine.base.guard_records(GPU):
                raise RuntimeError('Foreign GPU process, no retry')
            time.sleep(2)
        engine.GPU=GPU;engine.PIPE=PIPE;engine.ROOT=ROOT
        Path(PIPE).parent.mkdir(parents=True,exist_ok=True);engine.daemon(True);active=True
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,CUDA_MPS_PIPE_DIRECTORY=PIPE,
                 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
        for name in record['plan']:
            if time.time()>=a.deadline-120:raise TimeoutError('Same new overall deadline, no extension')
            command=[PYTHON,'-u',str(ENTRY),'--phase','worker','--root',str(ROOT),'--output',str(ROOT/name),
                     '--deadline',str(a.deadline),'--source-receipt-sha256',source_sha]
            runtime.write(ROOT/(name+'.command.json'),command)
            process_start=time.perf_counter()
            with (ROOT/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record.update(status='running',phase=name,child_pid=child.pid);runtime.write(ROOT/'suite_status.json',record)
                try:code=child.wait(timeout=max(1,a.deadline-time.time()))
                except BaseException:
                    os.killpg(child.pid,15);child.wait(timeout=20);raise
            if code:raise RuntimeError('LightGen failed, preserve this attempt without retry')
            value=json.loads((ROOT/name/'compute_report.json').read_text())
            if not value['all_predictions_bitwise_equal'] or value['actual_FFT_software_wall_s']<=0:raise ValueError('Native software result failed')
            runtime.write(ROOT/name/'rental_wall_summary.json',dict(
                full_fresh_process_wall_s=time.perf_counter()-process_start,
                actual_FFT_software_window_s=value['actual_FFT_software_wall_s'],
                measured_videos=512,formal_fields=32,extra_warm_fields=3,
                actual_FFT_s_per_16_videos=value['actual_FFT_software_wall_s']/32,
                resize_inclusive_fixed_optics_schedule_s_per_16=value['time16_s'],
                physical_photonic_hardware_measured=False,software_and_modelled_schedule_not_interchangeable=True))
            record['completed'].append(name);runtime.write(ROOT/'suite_status.json',record)
        record.update(status='complete',finished=time.time())
    except BaseException as e:record.update(status='stopped',error=repr(e));raise
    finally:
        if active:
            engine.daemon(False);record['mps_released']=True
        if ROOT.exists():runtime.write(ROOT/'suite_status.json',record)
        lock.close()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('suite','worker'),required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--deadline',type=float,required=True);p.add_argument('--source-receipt-sha256')
    a=p.parse_args();a.root=a.root.resolve();a.pipe=PIPE
    if a.root!=ROOT or not 0<a.deadline-time.time()<=9000:raise ValueError('Exact new rental root and bounded deadline')
    if a.phase=='worker' and (a.output is None or a.output.resolve().parent!=ROOT or a.output.name not in ('lg_repeat01','lg_repeat02','lg_repeat03')):
        raise ValueError('Exact fresh LightGen leaf required')
    configure()
    (suite if a.phase=='suite' else runtime.worker)(a)

if __name__=='__main__':main()
