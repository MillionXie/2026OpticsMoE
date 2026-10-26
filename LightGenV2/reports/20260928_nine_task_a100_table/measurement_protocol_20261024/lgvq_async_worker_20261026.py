"""Real LG async worker integration, NOT YET admitted as a formal measurement.

Never import this to start GPU work. A future exclusive controller must supply
the original guard after the baseline suite actually exits/releases its lock.
This worker retains the actual software wall, all 128 real outputs, the first
three eager fields, both real capture fields and original NVML raw evidence.
Original raw events are admitted through the witnessed DAG before fixed-optics
and component-union energy projection. No CLI or automatic dispatch is provided;
the external controller must still enforce exclusive original GPU admission.
"""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'
REFERENCE_RELATIVE = 'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/a100_lgvq_graph_pipeline_20261026_03/lg_repeat01/graph/graph_outputs.npz'
REFERENCE_SHA256 = '6fc71216c3deaa4975532256c3d50367203143e352268372da39ad89ea216a4e'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_map():
    names = ['lgvq_async_worker_20261026.py', 'lgvq_async_capture_20261026.py',
             'lgvq_async_replay_20261026.py', 'lgvq_async_trace_contract_20261026.py',
             'lgvq_async_component_energy_20261026.py', 'lgvq_async_observed_dag_20261026.py',
             'lgvq_async_raw_adapter_20261026.py', 'lgvq_async_optimization_20261026.py',
             'lgvq_scoreonly_20261026.py', 'lgvq_lowlevel_capture_20261026.py',
             'lgvq_physical_projection_20261026.py', 'lgvq_graph_pipeline_20261026.py',
             'lgvq_hoisted_pipeline_20261026.py', 'lgvq_minimal_pipeline_20261026.py',
             'lgvq_gpu_pipeline_20261026.py', 'lgvq_live_frontend.py',
             'lgvq_pipeline_demo_20261025.py']
    from lgvq_live_frontend import sources
    result = sources()
    for path in [HERE/name for name in names] + [
        REPO/'LightGenV2/tasks/t06_video_quality_assessment'/name for name in (
            'nvml_observer_process.py', 'e2e_telemetry_observer.py',
            'batch_telemetry.py', 'strict_tf_window_20261020.py')]:
        result[path.relative_to(REPO).as_posix()] = sha(path)
    return result


def validate_predictions(array, expected):
    import numpy as np
    if array.shape != (128, 16) or expected.shape != (128, 16):
        raise ValueError('Exactly 2048 real scores in 128 ordered fields required')
    if (array.dtype != np.float32 or expected.dtype != np.float32
            or not np.isfinite(array).all() or array.tobytes() != expected.tobytes()):
        raise ValueError('Every score must be bitwise equal to the sealed reference')
    if not all(array[:32].tobytes()==array[i:i+32].tobytes() for i in (32, 64, 96)):
        raise ValueError('Four original 512-video cycles differ')


def event_record(origin, event):
    return {k: v for k, v in event.items() if k not in ('start', 'end')} | dict(
        cuda_start_ms=origin.elapsed_time(event['start']),
        cuda_end_ms=origin.elapsed_time(event['end']))


def serialize_field(origin, record):
    row = {k: v for k, v in record.items()
           if k not in ('events', 'original_H2D_events', 'score_ready', 'body_envelope')}
    row['events'] = [event_record(origin, e) for e in record['events']]
    a, b = record['original_H2D_events']
    row['original_H2D_cuda_ms'] = [origin.elapsed_time(a), origin.elapsed_time(b)]
    row['score_ready_cuda_ms'] = origin.elapsed_time(record['score_ready'])
    if 'body_envelope' in record:
        a, b = record['body_envelope']
        row['body_envelope_cuda_ms'] = [origin.elapsed_time(a), origin.elapsed_time(b)]
    return row


def worker(args, guard, N):
    """Execute only when called by an exclusive, separately approved controller.

    Raw software completion is saved separately. A successful projected report
    requires every observed DAG node start, complete scoring and telemetry gates.
    """
    if N != 2048:
        raise ValueError('No probe/fast-prefix/round selection in this candidate')
    import os
    import subprocess
    import numpy as np
    import torch
    import lgvq_gpu_pipeline_20261026 as base
    saved = {k: getattr(base, k) for k in ('__file__', 'PIPE', 'source_map', 'model_guard')}
    import lgvq_graph_pipeline_20261026 as approved
    for key, value in saved.items():
        setattr(base, key, value)
    import lgvq_scoreonly_20261026 as score
    import lgvq_lowlevel_capture_20261026 as lowlevel
    import lgvq_async_capture_20261026 as capture
    import lgvq_async_replay_20261026 as replay
    import lgvq_async_optimization_20261026 as optimization
    import lgvq_live_frontend as live
    from lgvq_pipeline_demo_20261025 import PreparedFront
    from LightGenV2.tasks.t06_video_quality_assessment import lab_runtime as native
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.data import read_manifest
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as text
    from LightGenV2.tasks.t06_video_quality_assessment import e2e_telemetry_observer as observer
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.strict_tf_window_20261020 import interval_energy
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE') != '100':
        raise ValueError('Original exclusive lab GPU / actual 108-SM allocation required')
    guard.check()
    output = args.output
    optimized = bool(getattr(args, 'optimize_runtime', False))
    if optimized and os.environ.get('CUDA_MODULE_LOADING') != 'EAGER':
        raise ValueError('Optimized runtime must select EAGER before worker CUDA initialization')
    frozen = source_map()
    reference = REPO/REFERENCE_RELATIVE
    if sha(reference) != REFERENCE_SHA256:
        raise ValueError('Sealed validation reference identity changed')
    checkpoint = live.TASK/'runs/simulation/multivideo16x4_rank_s163/best_checkpoint.pt'
    if sha(checkpoint) != live.PINS['temporal'][1]:
        raise ValueError('Original model checkpoint identity changed')
    base.UNIQUE = 512; base.CYCLES = 4; base.TOTAL = N
    front = live.LiveFront('cuda:0')
    model, _ = native.load_model('temporal', checkpoint, 'cuda:0')
    model.requires_grad_(False)
    guard.bind()
    if torch.cuda.get_device_properties(0).multi_processor_count != 108:
        raise ValueError('Actual MPS allocation differs')
    power_limit = float(subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=power.limit', '--format=csv,noheader,nounits'], text=True, timeout=5))
    if power_limit != 250:
        raise ValueError('Original 250-W hardware limit required; never change it')
    fixed = PreparedFront(front)
    position = fixed.position4.repeat(16, 1).to(dtype=torch.bfloat16)
    raw, files = base.prep_lg(front, [r for r in read_manifest(live.MANIFEST) if r.split == 'test'], output, guard)
    base.write(output/'input_files.json', [dict(**r, bytes=Path(r['video_path']).stat().st_size,
        sha256=sha(r['video_path'])) for r in files])
    restore_constants, constants = approved.prepare_graph_constants(model)
    prefix_fn, tail_fn, partition = capture.scoring_partition(model, score, approved.hoisted)
    base.write(output/'async_identity.json', dict(status='prepared_for_actual_execution', sources=frozen,
        gpu_uuid=GPU, actual_visible_multiprocessors=108, power_limit_w=power_limit,
        published_commit=os.environ['MEASUREMENT_PUBLISHED_COMMIT'],
        source_HEAD=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        checkpoint_sha256=live.PINS['temporal'][1], reference_sha256=REFERENCE_SHA256,
        unique_videos=512, video_calls=N, fields=128, cycles=4, constants=constants,
        scoring_partition=partition, first_three_eager_real_fields_retained=True,
        two_independent_real_capture_fields_retained=True, dummy_forwards=0,
        optimized_runtime=optimized, detailed_electronics_phases=optimized,
        CUDA_MODULE_LOADING=os.environ.get('CUDA_MODULE_LOADING','not_explicitly_set'),
        original_full_field_coherent_six_passes=True, fixed_optics_projection_pending=True))
    fs, bs = torch.cuda.Stream(), torch.cuda.Stream()
    meter = approved.hoisted.minimal.NoInnerTimer()
    outputs = []; records = []; slots = []; sampler = None; restore_eager = None
    shared_preparation = [None]
    class OncePreparedTape(lowlevel.LowLevelTape):
        def __init__(self, torch, stream):
            super().__init__(torch, stream)
            if shared_preparation[0] is not None:
                self.preparation = shared_preparation[0]
        def start(self, kind, label):
            if optimized and self.preparation is None:
                optimization.prepare_without_cache_purge(self)
            super().start(kind, label)
            if shared_preparation[0] is None:
                shared_preparation[0] = self.preparation
    class ObservedEagerTape(lowlevel.projection.Tape):
        @contextmanager
        def excluded(self, label, kind):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter(); a.record()
            try:
                yield
            finally:
                b.record(); ended = time.perf_counter()
                self.eager.append((kind, label, a, b))
                self.eager_host.append((started, ended))
    def tokenize(field):
        a = time.perf_counter()
        rendered = text.render_prompt(front.tokenizer, target_name='temporal', prompt=text.TARGET_PROMPTS['temporal'])
        token = front.tokenizer([rendered], padding=True, return_tensors='pt', add_special_tokens=False)
        b = time.perf_counter()
        return token['input_ids'], token['attention_mask'], dict(field=field, start=a, end=b,
            input_ids=token['input_ids'].tolist(), attention_mask=token['attention_mask'].tolist())
    def prefix(current):
        batch = base.native_lg_front(front, position, current, meter, fs)
        return prefix_fn(model, **batch, optical_enabled=True)
    def tail(prepared):
        return tail_fn(model, prepared, optical_enabled=True)
    def install(model, tape):
        original = lowlevel.projection.install_simulation_boundaries
        return optimization.detailed_boundaries(model, tape, original) if optimized else original(model, tape)
    gc_policy = optimization.runtime_gc_policy(optimized)
    gc_policy.__enter__()
    try:
        sampler = ProcessTelemetrySampler(GPU, output/'telemetry_stdout.jsonl',
            command=observer.observer_command(GPU, output/'telemetry_stdout.jsonl'))
        sampler.start(); sampler.set_phase('LG_actual_async_software_including_FFT')
        until = time.monotonic()+6
        while len(sampler.samples) < 4:
            guard.fast(); sampler.check()
            if time.monotonic() > until:
                raise RuntimeError('No initial original NVML samples')
            time.sleep(.01)
        torch.cuda.synchronize()
        origin = torch.cuda.Event(enable_timing=True)
        clock_before = time.perf_counter(); origin.record(); origin.synchronize(); clock_after = time.perf_counter()
        mono_before = time.monotonic(); host_start = time.perf_counter(); mono_after = time.monotonic()
        eager_tape = ObservedEagerTape(torch, bs)
        restore_eager, eager_ledger = install(model, eager_tape)
        with torch.inference_mode():
            for field in range(3):
                guard.fast(); sampler.check()
                ids, mask, token_record = tokenize(field)
                pin_start = time.perf_counter()
                ids = ids.pin_memory(); mask = mask.bool().pin_memory()
                pin_end = time.perf_counter()
                with torch.cuda.stream(fs):
                    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
                    h2d_host_start = time.perf_counter()
                    a.record(); current = dict(raw[field], ids=ids.to('cuda', non_blocking=True),
                                               mask=mask.to('cuda', non_blocking=True)); b.record()
                    h2d_host_end = time.perf_counter()
                    x = torch.cuda.Event(enable_timing=True); y = torch.cuda.Event(enable_timing=True)
                    prefix_host_start = time.perf_counter()
                    x.record(); prepared = prefix(current); y.record()
                    prefix_host_end = time.perf_counter()
                eager_tape.eager = []; eager_tape.eager_host = []
                with torch.cuda.stream(bs):
                    bs.wait_event(y)
                    body_start = torch.cuda.Event(enable_timing=True); body_end = torch.cuda.Event(enable_timing=True)
                    body_host_start = time.perf_counter()
                    body_start.record(); result = tail(prepared); prediction = result['prediction'].clone(); body_end.record()
                    body_host_end = time.perf_counter()
                score_sync_start = time.perf_counter(); body_end.synchronize(); score_sync_end = time.perf_counter()
                events = [dict(kind='frontend', label='original_prefix', start=x, end=y, stream='frontend',
                               host_submit_start=prefix_host_start, host_submit_end=prefix_host_end)]
                events += [dict(kind=kind, label=label, start=x, end=y, stream='body',
                                host_submit_start=h[0], host_submit_end=h[1])
                           for (kind, label, x, y), h in zip(eager_tape.eager, eager_tape.eager_host)]
                if sum(e['kind'] == 'optical_simulation' for e in events) != 6:
                    raise RuntimeError('Original eager field lost a coherent optical pass')
                outputs.append(prediction)
                records.append(dict(field=field, mode='eager_real', tokenizer=token_record,
                    events=events, original_H2D_events=(a, b), body_envelope=(body_start, body_end),
                    body_host_start=body_host_start, body_host_end=body_host_end,
                    pin_host_start=pin_start, pin_host_end=pin_end,
                    H2D_host_submit_start=h2d_host_start, H2D_host_submit_end=h2d_host_end,
                    synchronization_observations=[dict(kind='score_ready',start=score_sync_start,end=score_sync_end)],
                    score_ready=body_end, score_ready_host=time.perf_counter()))
            restore_eager(); restore_eager = None
            for field in (3, 4):
                guard.fast(); sampler.check()
                ids, mask, token_record = tokenize(field)
                slot, prediction, record = capture.capture_real_slot(torch, field=field,
                    raw=raw[field], cpu_ids=ids, cpu_mask=mask, model=model,
                    front_stream=fs, body_stream=bs, prepare_prefix=prefix, forward_tail=tail,
                    tape_factory=OncePreparedTape, install_boundaries=install, phase_detail=optimized)
                record.update(mode='capture_real', slot=field-3, tokenizer=token_record)
                slots.append(slot); outputs.append(prediction); records.append(record)
            online = replay.replay_fields(torch, slots, raw, tokenize, guard, fs, bs, first_field=5, total_fields=128)
            outputs.extend(online['outputs']); records.extend(online['records'])
            host_end = online['last_score_ready_host']; monotonic_end = time.monotonic()
        if [slot['tape'].preparation_calls for slot in slots] != [1, 0]:
            raise RuntimeError('Exactly one original initial capture preparation required')
        serialized = [serialize_field(origin, row) for row in records]
        wall_ms = (host_end-host_start)*1000
        # Preserve original event clocks and ALL scores even if power/identity
        # acceptance fails afterwards. This leaf remains unqualified on failure.
        base.write(output/'async_event_records.json', dict(status='acquired_not_yet_qualified',
            measured_software_wall_ms=wall_ms, host_start=host_start, host_end=host_end,
            origin_host_clock_bracket=[clock_before, clock_after],
            perf_monotonic_alignment_bracket=[mono_before, mono_after], records=serialized))
        array = torch.cat(outputs).float().cpu().numpy()
        np.savez_compressed(output/'outputs.npz', prediction=array)
        sampler.request('barrier'); samples = sampler.stop(); sampler = None
        save_samples(output/'telemetry.csv', samples)
        with np.load(reference, allow_pickle=False) as sealed:
            expected = sealed['prediction']
        validate_predictions(array, expected)
        # Actual board-only FFT diagnostic, NOT the hybrid optical energy model.
        offset = (mono_before+mono_after)/2-host_start
        energy = interval_energy([(host_start+offset, host_end+offset)], samples, host_w=0.)
        timing = dict(status='actual_software_complete_projection_pending', video_calls=N,
            unique_videos=512, fields=128, cycles=4, actual_async_software_wall_ms=wall_ms,
            detailed_electronics_phases=optimized, optimized_runtime=optimized,
            actual_async_software_time16_ms=wall_ms/128,
            common_effective_tokens=913408, actual_software_common_tokens_s=913408/(wall_ms/1000),
            origin_host_clock_bracket=[clock_before, clock_after],
            perf_monotonic_alignment_bracket=[mono_before, mono_after],
            host_start=host_start, host_end=host_end, monotonic_end_observation=monotonic_end,
            records=serialized, eager_boundary_ledger=eager_ledger,
            all_predictions_bitwise_equal=True, reference_sha256=REFERENCE_SHA256,
            startup_and_both_actual_captures_included=True, explicit_warmup_forwards=0,
            initial_capture_preparation=shared_preparation[0], capture_preparation_calls=1,
            original_image_preparation_and_bulk_H2D_before_window=True,
            online_tokenizer_and_token_H2D_retained_in_raw_software_wall=True,
            raw_board_energy=energy, raw_energy_is_software_FFT_diagnostic_not_hybrid_projection=True,
            fixed_optics_projection=dict(status='PENDING_DEPENDENCY_DAG_INTEGRATION'),
            phase_energy_projection=dict(status='PENDING_UNION_ACCOUNTING_NO_BASELINE_HOST_PROXY'),
            qualified_for_final_hybrid_table=False, physical_hardware_time_measured=False)
        base.write(output/'async_raw_timing.json', timing)
        if source_map() != frozen or sha(checkpoint) != live.PINS['temporal'][1] or sha(reference) != REFERENCE_SHA256:
            raise RuntimeError('Source/weights/reference changed during candidate')
        guard.check()
        from lgvq_async_raw_adapter_20261026 import adapt
        trace, report = adapt(timing)
        if source_map()!=frozen:raise RuntimeError('Source changed during data-only DAG validation')
        guard.check()
        base.write(output/'async_causal_trace.json',trace)
        report.update(raw_timing_sha256=sha(output/'async_raw_timing.json'),
                      event_records_sha256=sha(output/'async_event_records.json'),
                      outputs_sha256=sha(output/'outputs.npz'),telemetry_sha256=sha(output/'telemetry.csv'))
        base.write(output/'cost_report.json',report)
        return report
    except BaseException as error:
        # Do not try to synchronize failed CUDA contexts just to format evidence.
        # Preserve completed host/token/capture records and original observer raw.
        partial = [{k: v for k, v in row.items() if k not in (
            'events', 'original_H2D_events', 'score_ready', 'body_envelope')}
            for row in records]
        base.write(output/'async_partial_host_records.json', dict(status='failed',
            error=repr(error), completed_fields_retained=len(outputs), records=partial,
            qualified_for_final_hybrid_table=False))
        raise
    finally:
        try:
            if restore_eager is not None:
                restore_eager()
        finally:
            restore_constants()
            if sampler is not None:
                save_samples(output/'telemetry.csv', sampler.stop(raise_on_error=False))
            gc_policy.__exit__(None, None, None)
