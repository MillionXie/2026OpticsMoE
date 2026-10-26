"""Three fresh native B1 rounds, corrected scope plus matched observed energy.

Reuse the already qualified inference worker unchanged. Only add the existing
independent CPU-only power observer and data-only accounting. Never infer new
energy from the earlier timing-only round or replace missing power observations.
"""
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
import rental_corrected_serial_20261026 as original


def write(path, value):
    original.write(Path(path), value)


def retained_intervals(rows):
    """Complement image preparation/H2D; retain tokenizer unions and all gaps."""
    if not rows:
        raise ValueError('No original records')
    result = []
    previous = rows[0]['CPU_start']
    end = rows[-1]['GPU_ready_host']
    for row in rows:
        a, c = row['CPU_start'], row['H2D_end']
        if not previous <= a <= row['CPU_end'] <= c <= row['GPU_ready_host'] <= end:
            raise ValueError('Nonserial or invalid timestamps')
        if previous < a:
            result.append((previous, a))
        merged = []
        for token in sorted(row['tokenizer_intervals'], key=lambda t: t['start']):
            x, y = token['start'], token['end']
            if not a <= x <= y <= row['CPU_end']:
                raise ValueError('Tokenizer outside original CPU preparation')
            if merged and x <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(y, merged[-1][1]))
            elif x < y:
                merged.append((x, y))
        result.extend(merged)
        previous = c
    if previous < end:
        result.append((previous, end))
    return result


def summarize(timing, samples, common_tokens):
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    rows = timing['records']
    intervals = retained_intervals(rows)
    cost = interval_energy(intervals, samples, 338.2)
    if not math.isclose(cost['elapsed_s'] * 1000, timing['corrected_GPU_ready_ms'], abs_tol=1e-4):
        raise ValueError('Energy and original timing boundaries differ')
    count = len(rows)
    native_tokens = sum(sum(bool(x) for x in r['attention_mask'][0]) for r in rows)
    return dict(status='complete', protocol='corrected_serial_with_matched_energy_20261026_v1',
        model=timing['model'], video_calls=count, unique_videos=timing['unique_videos'],
        cap=50, fresh_process_rounds=1, **cost,
        GPU_energy_j=cost['energy_j']-338.2*cost['elapsed_s'],
        observed_timeweighted_GPU_w=(cost['energy_j']/cost['elapsed_s'])-338.2,
        time16_ms=timing['time16_ms'], equivalent_frame_ms=timing['equivalent_frame_ms'],
        energy16_j=cost['energy_j']*16/count, energy_per_video_j=cost['energy_j']/count,
        energy_per_frame_j=cost['energy_j']/(count*4),
        native_effective_tokens=native_tokens, native_tokens_per_s=native_tokens/cost['elapsed_s'],
        common_Qwen_effective_tokens=common_tokens,
        common_tokens_per_s=common_tokens/cost['elapsed_s'] if common_tokens else None,
        cost_intervals=intervals, energy_is_board_NVML_plus_fixed_host_proxy_not_wall_meter=True,
        no_idle_subtraction=True, no_preparation_energy_added_back=True,
        every_video_serial=True, no_graph=True, no_prefetch=True)


def bind(root):
    original.ROOT = root
    original.PIPE = '/data/tmp/' + root.name


def worker(args):
    from LightGenV2.tasks.t06_video_quality_assessment import rental_precision_recheck as native
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as observer
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    old_runtime, old_sources = native.runtime, original.sources
    tracked = [Path(__file__), Path(observer.__file__),
        REPO/'LightGenV2/tasks/t06_video_quality_assessment/nvml_observer_process.py',
        REPO/'LightGenV2/tasks/t06_video_quality_assessment/batch_telemetry.py',
        REPO/'LightGenV2/tasks/t06_video_quality_assessment/strict_tf_window_20261020.py']
    extra = {p.relative_to(REPO).as_posix(): original.sha(p) for p in tracked}
    original.sources = lambda: dict(old_sources(), **extra)
    sampler = None
    def runtime(*positional, **keywords):
        nonlocal sampler
        result = old_runtime(*positional, **keywords)
        raw = args.output/'telemetry_stdout.jsonl'
        sampler = ProcessTelemetrySampler(original.GPU, raw,
            command=observer.observer_command(original.GPU, raw))
        sampler.start()
        sampler.set_phase('original_serial_no_warmup')
        until = time.monotonic()+3
        while len(sampler.samples) < 2:
            sampler.check()
            if time.monotonic() > until:
                raise RuntimeError('Power observer did not produce initial readings')
            time.sleep(.02)
        return result
    native.runtime = runtime
    try:
        original.worker(args)
        sampler.request('barrier')
        rows = sampler.stop()
        save_samples(args.output/'telemetry.csv', rows)
        timing = json.loads((args.output/'timing.json').read_text())
        report = summarize(timing, rows, args.common_tokens)
        report.update(round=args.repeat, timing_sha256=original.sha(args.output/'timing.json'),
            telemetry_sha256=original.sha(args.output/'telemetry.csv'), observer_sources=extra)
        write(args.output/'cost_report.json', report)
        write(args.output/'energy_status.json', dict(status='complete', repeat=args.repeat,
            pid=os.getpid(), time16_ms=report['time16_ms'], energy16_j=report['energy16_j']))
    except BaseException as error:
        if args.output.exists():
            write(args.output/'energy_status.json', dict(status='failed', error=repr(error), pid=os.getpid()))
        raise
    finally:
        if sampler is not None and not sampler.stopping:
            rows = sampler.stop(raise_on_error=False)
            if not (args.output/'telemetry.csv').exists():
                save_samples(args.output/'telemetry.csv', rows)
        native.runtime, original.sources = old_runtime, old_sources


def suite(args):
    import fcntl
    root = args.root
    if root.exists():
        raise FileExistsError('Preserve existing immutable run')
    root.mkdir()
    child = None
    active = False
    plan = [(m, r) for r in (1, 2, 3) for m in ('kimi', 'deepseek')]
    state = dict(status='admission', pid=os.getpid(), deadline=args.deadline,
        plan=plan, completed=[], cap=50, N=2048, common_tokens=args.common_tokens,
        source_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'],
        source_sha256=original.sha(__file__), sources=original.sources())
    def save():
        write(root/'suite_status.json', state)
    save()
    try:
        with (original.TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            for _ in range(12):
                if original.gpu_processes():
                    raise RuntimeError('Foreign GPU process')
                time.sleep(.25)
            if original.control('get_server_list').returncode == 0:
                raise RuntimeError('Existing private MPS daemon')
            Path(original.PIPE).mkdir(exist_ok=False)
            (root/'mps_logs').mkdir()
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=original.GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                CUDA_MPS_PIPE_DIRECTORY=original.PIPE, CUDA_MPS_LOG_DIRECTORY=str(root/'mps_logs'),
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='50', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
            subprocess.run(['nvidia-cuda-mps-control', '-d'], env=env, check=True)
            active = True
            for model, repeat in plan:
                if time.time() >= args.deadline-600:
                    raise TimeoutError('Preserve final backup margin; do not start another point')
                leaf = root/f'{model}50_repeat{repeat:02d}'
                command = ['/data/venvs/'+model+'/bin/python', '-u', str(Path(__file__)),
                    '--phase', 'worker', '--root', str(root), '--output', str(leaf), '--model', model,
                    '--repeat', str(repeat), '--deadline', str(args.deadline),
                    '--common-tokens', str(args.common_tokens)]
                write(root/(leaf.name+'.command.json'), command)
                with (root/(leaf.name+'.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    state.update(status='running', phase=leaf.name, child=child.pid)
                    save()
                    while child.poll() is None:
                        if time.time() >= args.deadline:
                            raise TimeoutError('Finite continuation deadline')
                        time.sleep(1)
                if child.returncode != 0 or json.loads((leaf/'energy_status.json').read_text())['status'] != 'complete':
                    raise RuntimeError('Preserve failed point; no blind retry')
                state['completed'].append(leaf.name)
                save()
            state.update(status='complete', finished=time.time())
    except BaseException as error:
        state.update(status='stopped', error=repr(error))
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            child.wait(timeout=60)
        raise
    finally:
        if active:
            result = original.control('quit')
            state.update(mps_release_returncode=result.returncode,
                mps_release_stdout=result.stdout, mps_release_stderr=result.stderr)
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['suite', 'worker'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--model', choices=['kimi', 'deepseek'])
    p.add_argument('--repeat', type=int, choices=[1, 2, 3])
    p.add_argument('--common-tokens', type=int, required=True)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    args.root = args.root.resolve()
    if REPO != Path('/data/2026OpticsMoE') or args.root.parent != original.TASK/'runs/simulation':
        raise ValueError('Only authorized new rental run directory')
    if not args.root.name.startswith('a10080_energy_triples_20261026_'):
        raise ValueError('New task-specific run name required')
    if not 0 < args.deadline-time.time() <= 36000 or args.common_tokens <= 0:
        raise ValueError('Finite bound and original shared token numerator required')
    bind(args.root)
    if args.phase == 'suite':
        suite(args)
    else:
        args.output = args.output.resolve()
        if args.output.parent != args.root:
            raise ValueError('Wrong leaf')
        worker(args)


if __name__ == '__main__':
    main()
