"""New authorized 80GB Qwen anchor; reuse the published E2E worker unchanged.

Only the hardware/path binding is different from the qualified 40GB worker.
This is not a replacement of the shared 40GB Qwen triple, nor a full TEST rerun.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
GPU = 'GPU-6d1916d1-fbd3-524b-3d98-7fb38512c388'


def bind_platform(entry):
    """Bind only GPU/cap profile; preserve model, arithmetic, input and cost code."""
    original_config = entry.config
    original_sources = entry.source_map
    def config():
        value = copy.deepcopy(original_config())
        value['profiles']['lab'].update(gpu_uuid=GPU, power_limit_w=400)
        return value
    entry.config = config
    entry.source_map = lambda: {
        **original_sources(), str(Path(__file__).relative_to(REPO)): entry.sha(__file__)}


def predecessor_ready(path):
    try:
        state = json.loads((Path(path)/'suite_status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if state['status'] not in ('complete', 'stopped'):
        return False
    if not state.get('mps_released'):
        raise RuntimeError('Predecessor has not released its private MPS')
    return True


def worker(entry, args):
    point = SimpleNamespace(
        profile='lab', model='qwen', cap=100, repeat=args.repeat,
        gpu_uuid=GPU, pipe=args.pipe, suite_root=args.suite_root,
        output=args.output, manifest=args.manifest, deadline=args.deadline,
        model_path=Path('/data/models/qwen3vl2b-89644892'),
        checkpoint=entry.TASK/'runs/simulation/lgvq_full_deepseek_kimi_rental_20261017'
        /'qwen_device_control_20261017/original_best_checkpoint.pt',
        reproduction_receipt=None)
    entry.worker(point)
    # Keep the untouched generic worker report. Explicitly override its broad
    # same_run_shared_qwen_temporal_mps100 flag in the separate role receipt.
    entry.frozen.write_json(args.output/'rental_anchor_role.json', dict(
        role='supplementary_same_80GB_GPU_Qwen_anchor_only',
        not_the_shared_40GB_table_anchor=True,
        generic_worker_shared_anchor_flag_not_applicable=True,
        gpu_uuid=GPU, power_limit_w=400, model_and_cost_math_unchanged=True,
        report_sha256=entry.sha(args.output/'report.json')))


def suite(entry, args):
    import fcntl
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_dataset_once as legacy
    root = entry.validate_output(args.output)
    root.mkdir(parents=True)
    state = dict(status='waiting_previous_queue', pid=os.getpid(),
        previous=str(args.previous), deadline=args.deadline, completed=[],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        sources=entry.source_map(), gpu_uuid=GPU, power_limit_w=400,
        role='supplementary_same_80GB_GPU_anchor_not_shared_40GB_anchor')
    save = lambda: entry.frozen.write_json(root/'suite_status.json', state)
    save()
    active = False
    try:
        while not predecessor_ready(args.previous):
            if time.time() > args.deadline-180:
                raise TimeoutError('Preserve rental backup time')
            time.sleep(10)
        with (entry.TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            legacy.select_profile('lab')
            legacy.engine.GPU=GPU; legacy.engine.PIPE=args.pipe; legacy.engine.ROOT=root
            state.update(status='admission'); save()
            for _ in range(3):
                if legacy.lab.shared.guard_records(GPU):
                    raise RuntimeError('Foreign GPU process; no mixed anchor')
                legacy.validate_idle_memory(GPU)
                time.sleep(10)
            legacy.engine.daemon(True); active=True
            for repeat in (1,2,3):
                if time.time() > args.deadline-120:
                    raise TimeoutError('No time for new anchor worker')
                if entry.source_map()!=state['sources']:
                    raise RuntimeError('Frozen source changed')
                point=root/f'qwen_cap100_repeat{repeat:02d}'
                command=[sys.executable,'-u',str(Path(__file__)), '--phase','worker',
                    '--output',str(point),'--suite-root',str(root), '--repeat',str(repeat),
                    '--manifest',str(args.manifest),'--deadline',str(args.deadline),
                    '--pipe',args.pipe]
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=GPU,
                    CUDA_MPS_PIPE_DIRECTORY=args.pipe, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='8')
                entry.frozen.write_json(root/(point.name+'.command.json'),command)
                with (root/(point.name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,
                        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    state.update(status='running',phase=point.name,child_pid=child.pid);save()
                    try:
                        code=child.wait(timeout=max(1,args.deadline-time.time()))
                    except BaseException:
                        # This is exactly the child created above, never any
                        # foreign PID or a process found by a broad name match.
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            child.wait(timeout=30)
                        raise
                if code:
                    raise RuntimeError(f'Anchor worker exit{code}; preserve original data')
                entry.frozen.write_json(point/'cpu_audit.json',entry.audit_point(point,args.manifest))
                state['completed'].append(point.name);save()
            state.update(status='complete',finished_unix=time.time())
    except BaseException as error:
        state.update(status='stopped',error=repr(error));raise
    finally:
        if active:
            try:legacy.engine.daemon(False);state['mps_released']=True
            except BaseException as error:state['release_error']=repr(error)
        save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('suite','worker','audit'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path)
    parser.add_argument('--suite-root',type=Path)
    parser.add_argument('--repeat',type=int,choices=(1,2,3))
    parser.add_argument('--manifest',type=Path,default=Path('/data/lgvq/original_manifest.csv'))
    parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--pipe',default='/data/tmp/t06_rental_qwen_anchor_20261025')
    args=parser.parse_args()
    if not REPO.is_relative_to('/data') or not 0<args.deadline-time.time()<=10800:
        raise ValueError('Authorized new rental /data and finite deadline only')
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry
    from LightGenV2.tasks.t06_video_quality_assessment import evaluate_temporal_baseline_batches
    entry.frozen=evaluate_temporal_baseline_batches
    bind_platform(entry)
    args.output=args.output.resolve()
    if args.phase=='suite':suite(entry,args)
    elif args.phase=='worker':worker(entry,args)
    else:print(json.dumps(entry.audit_point(args.output,args.manifest),indent=2))


if __name__=='__main__':main()
