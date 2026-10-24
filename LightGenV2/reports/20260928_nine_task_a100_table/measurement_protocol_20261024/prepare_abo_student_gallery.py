"""Untimed original .8600 B1 student image gallery in an isolated process.

Call the original encode_student_samples on all2400 TEST images, not cached
teacher features or an FP16 saved approximation of the original FP32 gallery.
No query/title inference is performed here. This is not a timing/energy trial.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(REPO))
from measure_abo_text_qwen import TASK,GPU,sha,write
from run_abo_student_contract import source_hashes,check_result,MODEL,BODY


def imported_sources():
    paths={Path(__file__),HERE/'abo_student_graph.py'}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file:
            # torch.classes declares a virtual relative _classes.py. Never
            # resolve that synthetic name against the repository working dir.
            # Missing ABSOLUTE repository sources still fail rather than hide.
            if not Path(file).is_absolute():continue
            p=Path(file).resolve()
            if p.suffix=='.py' and p.is_relative_to(REPO):paths.add(p)
    return {p.relative_to(REPO).as_posix():sha(p) for p in paths}


def generated_module_files():
    return sorted({str(getattr(m,'__file__')) for m in list(sys.modules.values())
        if getattr(m,'__file__',None) and not Path(getattr(m,'__file__')).is_absolute()})


def validate_reused_image_contract(path,title_contract):
    value=json.loads(Path(path).read_text())
    if (value.get('status')!='passed' or value.get('no_gpu') is not True
            or value.get('title_contract_sha256')!=sha(title_contract)
            or value.get('identity',{}).get('body_sha256')!='cc977b83286a8e90398ebd30064428886bc3c06f1eb0557e470060f7ff5c1cae'
            or [c.get('test_index') for c in value.get('checks',[])]!=[200,2399]
            or not all(c.get('bitwise_equal') is True for c in value['checks'])):
        raise ValueError('Only the already passed exact original gallery-image CPU contract may be reused')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','data','contract'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--cpu-contract',action='store_true')
    p.add_argument('--image-contract',type=Path,help='Explicit already-qualified CPU image-contract DATA; never repeat it')
    args=p.parse_args();output=args.output.resolve()
    if output.exists() or not output.is_relative_to(TASK/'runs/simulation') or not 0<args.deadline-time.time()<=7200:
        raise ValueError('Fresh finite task run required')
    contract=json.loads(args.contract.read_text());check_result(contract)
    queue=json.loads((args.contract.parent.parent/'suite_status.json').read_text())
    if queue['status']!='complete' or queue['sources']!=source_hashes():
        raise ValueError('Passed real-title CPU contract source binding changed')
    if args.cpu_contract:
        cpu_contract(args)
        return
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.measure_mps_dataset_once import validate_idle_memory
    import run_temporal_remainder as loading
    engine.GPU=GPU;engine.PIPE='/DATA/DATA1/guest3/.cache/nine_task_input_e2e_20261024';engine.ROOT=output
    lockpath=REPO/'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock'
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if engine.base.guard_records(GPU):raise RuntimeError('Foreign A100 compute; no mixed gallery generation')
        validate_idle_memory(GPU);output.mkdir(parents=True)
        state=dict(status='admission',controller_pid=os.getpid(),command=sys.argv,no_formal_timing=True,
            original_contract=str(args.contract),original_contract_sha256=sha(args.contract),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        write(output/'controller_status.json',state)
        active=False;child=None
        try:
            # Gallery vision was not exercised by the pure-title contract.
            # Verify actual held-out images first in a separate CPU process.
            cpu=output/'original_gallery_image_cpu_contract'
            if args.image_contract:
                validate_reused_image_contract(args.image_contract,args.contract)
                cpu.mkdir()
                with (cpu/'contract.json').open('xb') as target:target.write(args.image_contract.read_bytes())
                write(output/'reused_image_contract_receipt.json',dict(original=str(args.image_contract),
                    original_sha256=sha(args.image_contract),copy_sha256=sha(cpu/'contract.json'),CPU_or_GPU_reinference_calls=0))
            else:
                command=[sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(cpu),
                    '--data',str(args.data),'--contract',str(args.contract),'--deadline',str(args.deadline),'--cpu-contract']
                write(output/'cpu_contract.command.json',command)
                with (output/'cpu_contract.log').open('x') as log:
                    own=subprocess.Popen(command,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    state.update(status='real_gallery_image_cpu_contract',cpu_worker_pid=own.pid)
                    write(output/'controller_status.json',state)
                    try:
                        while own.poll() is None:
                            if time.time()>args.deadline-120:raise TimeoutError('No remaining gallery budget')
                            if engine.base.guard_records(GPU):raise RuntimeError('Foreign GPU acquired during CPU gallery contract')
                            time.sleep(2)
                        if own.returncode:raise RuntimeError('Actual gallery image CPU contract failed')
                    finally:
                        if own.poll() is None:own.terminate();own.wait(timeout=15)
            if json.loads((cpu/'contract.json').read_text())['status']!='passed':raise ValueError('Missing original image contract')
            for _ in range(3):
                if engine.base.guard_records(GPU):raise RuntimeError('Foreign A100 during gallery admission')
                validate_idle_memory(GPU);time.sleep(10)
            engine.daemon(True);active=True
            os.environ.update(CUDA_VISIBLE_DEVICES=GPU,CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_MPS_PIPE_DIRECTORY=engine.PIPE,CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
            # CPU parent never imports Torch/CUDA and releases MPS only after
            # its sole GPU child has fully exited.
            child=os.fork()
            if child:
                observations=[]
                while True:
                    pid,status=os.waitpid(child,os.WNOHANG)
                    if pid:break
                    if time.time()>args.deadline:raise TimeoutError('Gallery worker deadline')
                    snapshot=loading.process_snapshot(GPU,child,engine.PIPE);observations.append(snapshot)
                    if snapshot['foreign_gpu_processes'] or snapshot['foreign_mps_clients']:
                        write(output/'foreign_loading_or_encoding_interruption.json',snapshot)
                        raise RuntimeError('Foreign process during gallery load/encoding')
                    time.sleep(1)
                write(output/'parent_process_audit.json',observations)
                code=os.waitstatus_to_exitcode(status)
                if code:raise RuntimeError('Original gallery worker exit '+str(code))
                if json.loads((output/'status.json').read_text())['status']!='complete':raise ValueError('Missing complete bank')
                state.update(status='complete',worker_pid=child,worker_exit=code,finished_unix=time.time())
            else:
                try:worker(args,contract,engine);os._exit(0)
                except BaseException as error:
                    import traceback
                    traceback.print_exc();write(output/'failure.json',dict(error=repr(error)));os._exit(1)
        except BaseException as error:
            state.update(status='stopped',error=repr(error))
            if child:
                try:
                    pid,_=os.waitpid(child,os.WNOHANG)
                    if not pid:os.kill(child,signal.SIGTERM);os.waitpid(child,0)
                except ChildProcessError:pass
            raise
        finally:
            if child!=0:
                if active:
                    try:engine.daemon(False);state['mps_released']=True
                    except BaseException as error:state['release_error']=repr(error)
                write(output/'controller_status.json',state)


def cpu_contract(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from collections import Counter
    from abo_student_graph import load,patch
    from measure_salicon import shared
    from LightGenV2.tasks.t08_abo_image_text_retrieval import reverse_runtime as original
    torch.set_num_threads(8);args.output.mkdir(parents=True)
    write(args.output/'status.json',dict(status='loading',no_gpu=True,pid=os.getpid()))
    replacement=None;meter=shared.meter_lib.StageMeter()
    try:
        runtime=SimpleNamespace(model=MODEL,data=args.data,checkpoint=BODY,output=args.output)
        loaded,replacement,readout,settings,data,prompts,identity=load(runtime,'cpu')
        checks=[]
        with torch.inference_mode():
            for index in (200,2399):
                sample=data.test[index];meter.close()
                expected=original.encode_student_samples(loaded,replacement,readout,[sample],settings)
                patch(loaded,replacement,readout,meter);meter.sample_index=index
                actual=original.encode_student_samples(loaded,replacement,readout,[sample],settings)
                if not torch.equal(expected,actual):raise ValueError('Actual original gallery image CPU forward is not bitwise equal')
                counts=Counter(r['stage'] for r in meter.rows if r['sample_index']==index)
                for modality in ('vision','language'):
                    if counts[modality+'.router_simulation']!=1 or counts[modality+'.expert_global_simulation']!=2:
                        raise ValueError('Original six optical gallery passes were not observed')
                checks.append(dict(test_index=index,sample_id=sample.sample_id,product_id=sample.sku_name,
                    image_file_sha256=sha(sample.image_path),bitwise_equal=True,shape=list(actual.shape),
                    module_call_counts=dict(counts)))
        if torch.cuda.is_initialized():raise ValueError('CPU image contract created CUDA')
        write(args.output/'contract.json',dict(status='passed',no_gpu=True,checks=checks,identity=identity,
            stage_ledger=meter.ledger,title_contract_sha256=sha(args.contract),not_a_formal_timing_or_energy_run=True))
        write(args.output/'status.json',dict(status='complete',no_gpu=True,real_images=2,pid=os.getpid()))
    except BaseException as error:
        write(args.output/'status.json',dict(status='failed',no_gpu=True,error=repr(error),pid=os.getpid()));raise
    finally:
        meter.close()
        if replacement is not None:replacement.close()


def worker(args,contract,engine):
    guard=engine.MPSGuard(SimpleNamespace(deadline=args.deadline));guard.check()
    import numpy as np
    import torch
    from abo_student_graph import load
    from LightGenV2.tasks.t08_abo_image_text_retrieval import reverse_runtime as original
    torch.set_num_threads(8);replacement=None
    output=args.output.resolve()
    write(output/'status.json',dict(status='loading',pid=os.getpid(),images=0,no_formal_timing=True))
    try:
        props=torch.cuda.get_device_properties(0)
        cap=float(subprocess.check_output(['nvidia-smi','--id='+GPU,'--query-gpu=power.limit','--format=csv,noheader,nounits'],text=True))
        if props.name!='NVIDIA A100-PCIE-40GB' or props.multi_processor_count!=108 or abs(cap-250)>.1:
            raise ValueError('Incorrect physical A100/cap')
        runtime=SimpleNamespace(model=MODEL,data=args.data,checkpoint=BODY,output=output)
        loaded,replacement,readout,settings,data,prompts,info=load(runtime,'cuda:0')
        if info['model_assets']!=contract['identity']['model_assets']:raise ValueError('Original CPU-bound native assets changed')
        if settings.inference_batch_size!=1 or len(data.test)!=2400:raise ValueError('Original full gallery/B1 required')
        sources=imported_sources();virtual=generated_module_files();guard.bind()
        write(output/'status.json',dict(status='encoding_original_gallery2400',pid=os.getpid(),images=0,no_formal_timing=True))
        with torch.inference_mode():
            vectors=original.encode_student_samples(loaded,replacement,readout,data.test,settings)
        guard.check()
        if vectors.shape!=(2400,64) or not torch.isfinite(vectors).all():raise ValueError('Original student gallery output shape/content')
        if imported_sources()!=sources:raise ValueError('Original source module identities changed')
        np.savez_compressed(output/'gallery.npz',vectors=vectors.float().numpy(),
            sample_ids=np.asarray([s.sample_id for s in data.test]),
            product_ids=np.asarray([s.sku_name for s in data.test]),labels=np.asarray([s.sku_index for s in data.test]))
        write(output/'receipt.json',dict(status='complete',identity=info,sources=sources,generated_module_files_not_Git_sources=virtual,images=2400,batch=1,
            original_function='reverse_runtime.encode_student_samples, original image instruction/settings unchanged',
            gallery_sha256=sha(output/'gallery.npz'),original_output_dtype=str(vectors.dtype),saved_dtype='float32',
            contract_sha256=sha(args.contract),gallery_image_cpu_contract_sha256=sha(output/'original_gallery_image_cpu_contract/contract.json'),
            dataset_sha256=data.sha256,image_query_only=True,title_or_dummy_inference_calls=0,
            online_title_warmup=False,not_a_formal_timing_or_energy_run=True,gpu_uuid=GPU,mps_percent=100,power_limit_w=cap))
        write(output/'status.json',dict(status='complete',pid=os.getpid(),images=2400,no_formal_timing=True))
        print(json.dumps(dict(status='complete',images=2400,gallery_sha256=sha(output/'gallery.npz'))),flush=True)
    finally:
        if replacement is not None:replacement.close()
        guard.close(output/'worker_process_audit.json')
        write(output/'files_manifest.json',[dict(file=f.name,bytes=f.stat().st_size,sha256=sha(f))
            for f in sorted(output.iterdir()) if f.is_file() and f.name not in ('files_manifest.json','controller_status.json')])


if __name__=='__main__':main()
