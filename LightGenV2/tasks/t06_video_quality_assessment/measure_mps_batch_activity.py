"""Bounded batch selection by conditional kernel activity, then unprofiled costs.

Frozen native arithmetic, original precision and timer. Instrumented activity is
only a declared batch-selection diagnostic; never substituted for NVML or for
unprofiled cost. Keep first cold batch, every original frame, and all tail videos.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import diagnose_mps_tf_activity as activity
from . import measure_mps_dataset_once as original
from . import tf_activity_clock as clock

TASK, REPO = original.TASK, original.REPO
CONFIG = TASK/'configs/baselines/mps_batch_activity_20261021.json'
ROOT = TASK/'runs/simulation/lgvq_mps_batch_activity_a100_20261021'
PIPE = '/DATA/DATA1/guest3/.cache/t06_mps_batch_activity_20261021'


def configuration():
    c = json.loads(CONFIG.read_text())
    assert c['models'] == ['clip', 'yolo', 'qwen', 'glm']
    assert (c['selection_cap'], c['paired_caps'], c['pilot_videos']) == (100, [100, 50], 32)
    assert (c['test_videos'], c['cost_videos'], c['repeats'], c['explicit_warmup_forwards']) == (558, 200, 3, 0)
    assert c['maximum_suite_s'] == 7200 and c['minimum_profiled_busy_percent'] == 50
    for batches in c['candidate_batches'].values():
        assert batches == sorted(set(batches)) and batches[0] == 1 and batches[-1] <= 16
    return c


def sources():
    return {**original.sources(), **{str(p.relative_to(REPO)): original.frozen.sha256(p)
        for p in (Path(__file__), CONFIG, Path(activity.__file__), Path(clock.__file__))}}


def batches(batch, pilot=False):
    if not isinstance(batch, int) or batch < 1 or batch > 16:
        raise ValueError('Finite positive batch <=16 required')
    return ([(i, min(i+batch, 32)) for i in range(0, 32, batch)] if pilot
            else original.costs.plan(batch))


def qualifies(reports):
    return len(reports) == 3 and all(r['status'] == 'complete' and
        r['measurement']['lower_percent'] >= 50 for r in reports)


def selected_batch(attempts, candidates):
    for batch in candidates:
        if qualifies(attempts.get(batch, [])):
            return batch
    return None


def worker(args):
    import torch
    import numpy as np
    from . import quality_token_common as core
    c = configuration()
    engine, frozen = original.engine, original.frozen
    engine.GPU, engine.PIPE, engine.ROOT = frozen.A100_UUID, PIPE, ROOT
    if args.output.exists():
        raise RuntimeError('Preserve all existing evidence')
    if os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE') != str(args.cap):
        raise RuntimeError('Cap must precede CUDA initialization')
    if args.batch not in c['candidate_batches'][args.model]:
        raise ValueError('Batch outside finite approved plan')
    args.output.mkdir()
    bound, calls, predictions, anchors = sources(), [], [], []
    sampler = finish = None
    class Guard(engine.MPSGuard):
        def check(self):
            if sources() != bound:
                raise RuntimeError('Bound measurement source changed')
            super().check()
    guard = Guard(args)
    identity = dict(model=args.model, cap=args.cap, batch=args.batch, repeat=args.repeat,
        mode=args.mode, pid=os.getpid(), command=sys.argv, sources=bound,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        checkpoint_sha256=frozen.sha256(args.checkpoint), manifest_sha256=frozen.sha256(args.manifest),
        gpu_uuid=frozen.A100_UUID, explicit_warmup_forwards=0, protocol=c,
        historical_Qwen_B1_unchanged=True, official_paper_exports_unchanged=True)
    def status(value, **extra):
        frozen.write_json(args.output/'status.json', dict(status=value, unix=time.time(), **extra))
    def evaluate(rows, prepare, transfer, forward, start):
        for index, (begin, end) in enumerate(batches(args.batch, args.mode == 'pilot')):
            guard.fast()
            sampler.check()
            p = time.perf_counter()
            cpu, positions = prepare(rows[begin:end])
            decoded = time.perf_counter()
            inputs = transfer(cpu)
            torch.cuda.synchronize()
            a = time.perf_counter()
            if args.mode == 'pilot':
                anchors.append(clock.snapshot())
            values, wall, cuda = forward(inputs)
            returned, s = time.perf_counter(), start()
            e = s + wall/1000
            if args.mode == 'pilot':
                anchors.append(clock.snapshot())
            if len(values) != end-begin or not all(map(math.isfinite, values)) or not p <= a <= s < e <= returned+1e-5:
                raise RuntimeError('Invalid native batched result or boundary')
            calls.append(dict(sample_index=begin, sample_id=rows[begin]['sample_id'],
                batch_index=index, videos=end-begin, sample_indices=list(range(begin,end)),
                sample_ids=[r['sample_id'] for r in rows[begin:end]], preprocess_start=p,
                model_call_start=a, first_block_start=s, gpu_output=e, call_return=returned,
                external_preprocess_end=a, decode_processor_s=decoded-p, h2d_sync_s=a-decoded,
                excluded_model_preblock_s=s-a, legacy_host_ms=wall, cuda_ms=cuda,
                frame_positions=positions, in_cost_subset=end <= 200))
            predictions.extend(float(x) for x in values)
            del inputs, cpu
            status('profiling' if args.mode == 'pilot' else 'evaluating', samples=end,
                   total=32 if args.mode == 'pilot' else 558)
            print(f'{args.model} B{args.batch} cap{args.cap} {args.mode} {end} native_ms={wall:.3f}', flush=True)
    try:
        guard.check()
        torch.set_num_threads(8)
        if identity['manifest_sha256'] != frozen.MANIFEST_SHA:
            raise RuntimeError('Original split required')
        limit = float(subprocess.check_output(['nvidia-smi','--id='+frozen.A100_UUID,
            '--query-gpu=power.limit','--format=csv,noheader,nounits'], text=True))
        if abs(limit-c['power_limit_w']) > .1:
            raise RuntimeError('Unchanged250W required')
        status('loading')
        with torch.inference_mode():
            rows, prepare, transfer, forward, start, finish = original.runtime(args, guard, identity)
            visible = torch.cuda.get_device_properties(0).multi_processor_count
            if visible != (108 if args.cap == 100 else 54):
                raise RuntimeError('MPS resource cap not effective')
            identity.update(visible_multiprocessors=visible, power_limit_w=limit,
                profiler_overhead_present=args.mode == 'pilot')
            frozen.write_json(args.output/'identity.json', identity)
            sampler = original.ProcessTelemetrySampler(frozen.A100_UUID, args.output/'telemetry_stdout.jsonl')
            sampler.start()
            sampler.set_phase(args.mode)
            time.sleep(.3)  # CPU-only observation setup; no warmup forward.
            if args.mode == 'pilot':
                errors = [clock.empty_cpu_calibration(args.output, 'clock_before')]
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    evaluate(rows, prepare, transfer, forward, start)
                target = args.output/'cuda_trace.json'
                prof.export_chrome_trace(str(target))
                if target.stat().st_size > c['maximum_trace_bytes']:
                    raise RuntimeError('Finite trace-storage ceiling exceeded')
                errors.append(clock.empty_cpu_calibration(args.output, 'clock_after'))
                trace = json.loads(target.read_text())
                if any(e.get('cat') == 'cpu_op' for e in trace['traceEvents']):
                    raise RuntimeError('CPU operator tracing unexpectedly enabled')
                measurement = activity.summarize_trace(trace, calls, anchors, max(errors)+10)
                measurement.update(empty_cpu_clock_calibration_errors_us=errors,
                    first_cold_batch_included=True, all_tail_videos_retained=True,
                    minimum50_reached=measurement['lower_percent'] >= 50)
                payload = dict(measurement=measurement, activity_is_instrumented=True,
                    not_SM_activity=True, not_uninstrumented_NVML=True, no_formal_cost=True)
            else:
                evaluate(rows, prepare, transfer, forward, start)
                payload = {}
            sampler.request('barrier')
            samples = sampler.stop()
            sampler = None
            original.telemetry.save_samples(args.output/'telemetry.csv', samples)
            if args.mode == 'cost':
                charged = [x for x in calls if x['in_cost_subset']]
                if sum(x['videos'] for x in charged) != 200 or len(predictions) != 558:
                    raise RuntimeError('Missing first200 or full558 / tail batch')
                intervals, compute, gap = original.charged_intervals(charged)
                result = original.energy.interval_energy(intervals, samples, c['host_proxy_w'])
                payload = dict(test_videos=558, cost_videos=200,
                    performance=core.metrics(np.array([float(r['temporal']) for r in rows]),np.array(predictions)),
                    cost=dict(time16_ms=result['elapsed_s']*80,energy16_j=result['energy_j']*.08,
                        compute_s=compute,nonpreprocessing_gap_s=gap,power_integration=result),
                    no_profiling_during_cost=True, throughput_normalized_to16=True)
            finish()
            finish = None
            guard.check()
            frozen.write_json(args.output/'report.json', dict(status='complete', **identity, **payload))
            status('complete', samples=len(predictions))
    except BaseException as error:
        status('failed', error=repr(error), samples=len(predictions))
        raise
    finally:
        if sampler is not None:
            original.telemetry.save_samples(args.output/'telemetry.csv', sampler.stop(raise_on_error=False))
        if finish is not None:
            finish()
        frozen.write_json(args.output/'calls.json', calls)
        frozen.write_json(args.output/'clock_anchors.json', anchors)
        original.write_csv(args.output/'predictions.csv', [dict(sample_index=i,
            sample_id=rows[i]['sample_id'],target_mos=rows[i]['temporal'],prediction=x)
            for i,x in enumerate(predictions)])
        guard.close(args.output/'process_audit.json')


def suite(args):
    import fcntl
    c = configuration()
    engine = original.engine
    engine.GPU, engine.PIPE, engine.ROOT = original.frozen.A100_UUID, PIPE, ROOT
    with (TASK/'runs/simulation/.temporal_telemetry_a100.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        if ROOT.exists():
            raise RuntimeError('Do not overwrite or automatically resume any run')
        ROOT.mkdir()
        state = dict(status='admission',pid=os.getpid(),deadline=args.deadline,protocol=c,completed=[],selections={})
        active = False
        def save():
            original.frozen.write_json(ROOT/'suite_status.json',state)
        def run(model,batch,cap,mode,repeat):
            name = f'{model}_b{batch:02d}_cap{cap}_{mode}_r{repeat:02d}'
            mp,pt,python,receipt = original.lab.locations(model)
            command = [python,'-u','-m',__package__+'.measure_mps_batch_activity','--phase','worker',
                '--mode',mode,'--model',model,'--batch',str(batch),'--cap',str(cap),'--repeat',str(repeat),
                '--model-path',str(mp),'--checkpoint',str(pt),'--manifest',str(args.manifest),
                '--output',str(ROOT/name),'--deadline',str(args.deadline)]
            if receipt:
                command += ['--reproduction-receipt',str(receipt)]
            if args.deadline-time.time() < 120:
                raise TimeoutError('No time for another fresh worker')
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=engine.GPU,CUDA_MPS_PIPE_DIRECTORY=PIPE,
                CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(cap),OMP_NUM_THREADS='8',HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',HF_HOME='/data/cache')
            with (ROOT/(name+'.log')).open('x') as log:
                child = subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='running',phase=name,child_pid=child.pid)
                save()
                try:
                    code = child.wait(timeout=max(1,args.deadline-time.time()))
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=15)
                    raise
            if code:
                raise RuntimeError(f'{name} exit{code}; retain evidence, no automatic retry')
            state['completed'].append(name)
            save()
            return json.loads((ROOT/name/'report.json').read_text())
        save()
        try:
            for _ in range(3):
                if original.lab.shared.guard_records(engine.GPU):
                    raise RuntimeError('Foreign A100 process: do not occupy or wait indefinitely')
                original.validate_idle_memory(engine.GPU)
                if time.time() >= args.deadline:
                    raise TimeoutError('Finite admission deadline')
                time.sleep(10)
            engine.daemon(True)
            active = True
            for model in c['models']:
                attempts = {}
                for batch in c['candidate_batches'][model]:
                    reports = [run(model,batch,100,'pilot',1)]
                    if reports[0]['measurement']['lower_percent'] >= 50:
                        reports += [run(model,batch,100,'pilot',r) for r in (2,3)]
                    attempts[batch] = reports
                    chosen = selected_batch(attempts,c['candidate_batches'][model])
                    if chosen is not None:
                        break
                else:
                    state['selections'][model] = dict(status='no_qualified_batch_within_finite_plan')
                    save()
                    continue
                state['selections'][model] = dict(batch=chosen,selection_cap=100,
                    selection_metric=c['selection_metric'],status='selected_before_cost')
                save()
                paired = [run(model,chosen,50,'pilot',r) for r in (1,2,3)]
                state['selections'][model]['paired_cap50_minimum50_reached'] = qualifies(paired)
                save()
                for cap in (100,50):
                    for repeat in (1,2,3):
                        run(model,chosen,cap,'cost',repeat)
                state['selections'][model]['status'] = 'paired_cost_complete'
                save()
            state.update(status='complete',finished=time.time())
        except BaseException as error:
            state.update(status='stopped',error=repr(error))
            raise
        finally:
            if active:
                try:
                    engine.daemon(False)
                    state['mps_released'] = True
                except BaseException as error:
                    state['release_error'] = repr(error)
            save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('suite','worker'),required=True)
    p.add_argument('--mode',choices=('pilot','cost'))
    p.add_argument('--model',choices=('clip','yolo','qwen','glm'))
    p.add_argument('--batch',type=int)
    p.add_argument('--cap',type=int,choices=(100,50))
    p.add_argument('--repeat',type=int,choices=(1,2,3))
    p.add_argument('--model-path',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--reproduction-receipt',type=Path)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--deadline',type=float,required=True)
    a = p.parse_args()
    if not 0 < a.deadline-time.time() <= configuration()['maximum_suite_s']:
        raise ValueError('Finite two-hour maximum required')
    suite(a) if a.phase == 'suite' else worker(a)


if __name__ == '__main__':
    main()
