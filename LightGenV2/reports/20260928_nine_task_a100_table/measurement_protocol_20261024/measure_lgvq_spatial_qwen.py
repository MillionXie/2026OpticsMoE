"""Original spatial quality-token baseline, input-inclusive 200-call protocol.

The frozen .6907726 full-558 reference binds head9155ec7a. The earlier .72324
report is its 200-subset, not a different desired full-TEST model. Reuse the
original processor/forward (including full vocabulary projection) unchanged.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
TASK = REPO / 'LightGenV2/tasks/t06_video_quality_assessment'
PIN = '9155ec7a45a19e14ac87661021b44599317d5efb16475df77707d70e866ad913'
REFERENCE_SHA = '9858954de368171e4736607cd8f2340b871191799e10d3068b3576b73cdd20e6'
REFERENCE = REPO / 'LightGenV2/reports/20260914_latest_qwen_baselines_a100_FIRSTBLOCK_FINAL/evidence/T06_spatial/report.json'
GPU = 'GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55'


def validate_reference(sha):
    if sha(REFERENCE) != REFERENCE_SHA:
        raise ValueError('Full558 spatial baseline reference changed')
    reference = json.loads(REFERENCE.read_text())
    if (reference['checkpoint_sha256'] != PIN or reference['target'] != 'spatial'
            or reference['test_videos'] != 558 or reference['frame_count'] != 4
            or abs(reference['performance']['srcc'] - .6907725734336255) > 1e-12):
        raise ValueError('Wrong spatial model reference')
    return reference


def collect(rows, prepare, transfer, forward, first_start, sync, check,
            observe, progress, records, clock=time.perf_counter):
    """Continuous serial input-output calls; every genuine gap is retained."""
    for index, row in enumerate(rows):
        check()
        p = clock()
        cpu, positions = prepare([row])
        d = clock()
        inputs = transfer(cpu)
        sync()
        a = clock()
        predictions, host_ms, cuda_ms = forward(inputs)
        sync()
        returned = clock()
        started = float(first_start())
        ended = started + float(host_ms) / 1000
        if (len(predictions) != 1 or not math.isfinite(float(predictions[0]))
                or not p <= d <= a <= started < ended <= returned
                or len(positions[0]) != 4):
            raise ValueError('Invalid original four-frame scalar forward')
        records.append(dict(sample_index=index, sample_id=row['sample_id'],
            target_mos=float(row['spatial']), prediction=float(predictions[0]),
            preprocess_start=p, prepare_end=d, model_call_start=a,
            first_block_start=started, gpu_output=ended, call_return=returned,
            legacy_host_ms=float(host_ms), diagnostic_first_block_cuda_ms=float(cuda_ms),
            frame_positions=json.dumps(positions[0])))
        observe(row, cpu)
        del inputs, cpu
        if index == 0 or (index + 1) % 25 == 0:
            progress(index + 1)


def source_map():
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry
    return {**entry.source_map(), str(Path(__file__).relative_to(REPO)):entry.sha(__file__),
        str((TASK/'quality_token_dataset_once.py').relative_to(REPO)):entry.sha(TASK/'quality_token_dataset_once.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'checkpoint', 'manifest', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--repeat', type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(TASK/'runs/simulation'):
        raise ValueError('New original-task run required')
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != GPU
            or os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE') != '100'):
        raise ValueError('Exact A100 UUID and private MPS100 required before CUDA')
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_e2e_20261024 as entry
    from LightGenV2.tasks.t06_video_quality_assessment import measure_mps_dataset_once as old
    from LightGenV2.tasks.t06_video_quality_assessment import rental_busy_resource_20261020 as engine
    from LightGenV2.tasks.t06_video_quality_assessment.batch_telemetry import save_samples
    from LightGenV2.tasks.t06_video_quality_assessment.e2e_telemetry_observer import observer_command
    from LightGenV2.tasks.t06_video_quality_assessment.nvml_observer_process import ProcessTelemetrySampler
    rows = entry.input_rows(args.manifest)
    reference = validate_reference(entry.sha)
    if entry.sha(args.checkpoint) != PIN:
        raise ValueError('Wrong spatial quality-token head')
    engine.GPU, engine.PIPE = GPU, os.environ['CUDA_MPS_PIPE_DIRECTORY']
    bound = source_map()
    class Guard(engine.MPSGuard):
        def check(self):
            if source_map() != bound:
                raise ValueError('Bound spatial inference source changed')
            super().check()
    guard = Guard(SimpleNamespace(deadline=time.time()+1800))
    guard.check()
    import numpy as np
    import torch
    from LightGenV2.tasks.t06_video_quality_assessment import quality_token_common as core
    from LightGenV2.tasks.t06_video_quality_assessment import quality_token_batch_benchmark_200 as original
    props = torch.cuda.get_device_properties(0)
    cap = float(subprocess.check_output(['nvidia-smi', '--id='+GPU, '--query-gpu=power.limit', '--format=csv,noheader,nounits'], text=True))
    if props.name != 'NVIDIA A100-PCIE-40GB' or props.multi_processor_count != 108 or abs(cap-250) > .1:
        raise ValueError('Wrong physical GPU or resource contract')
    torch.set_num_threads(8)
    args.output.mkdir(parents=True)
    write = old.frozen.write_json
    identity = dict(status='loading', task='LGVQ spatial', system='qwen', pid=os.getpid(),
        protocol_id='lightgen_qwen_input_inclusive_200x3_20261024_v1', repeat=args.repeat,
        batch=1, mps_percent=100, gpu_uuid=GPU, visible_multiprocessors=108, power_limit_w=cap,
        sources=bound, reference=str(REFERENCE), reference_sha256=REFERENCE_SHA,
        checkpoint=str(args.checkpoint), checkpoint_sha256=PIN, manifest_sha256=entry.sha(args.manifest),
        first200_ids_sha256=entry.ordered_id_sha(rows), explicit_warmup_forwards=0,
        first_sample_retained=True, full558_reference=reference['performance'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        command=sys.argv, cpu_threads=8, torch=torch.__version__, cuda=torch.version.cuda)
    write(args.output/'status.json',identity)
    calls, evidence, telemetry = [], [], []
    sampler = timer = None
    def status(state, **extra):
        write(args.output/'status.json',dict(status=state,pid=os.getpid(),samples=len(calls),**extra))
    try:
        input_metadata = entry.file_identities(rows)
        with torch.inference_mode():
            assets = old.costs.model_hashes(args.model)
            _, temporal_ref = entry.reference('qwen')
            if assets != temporal_ref['assets']:
                raise ValueError('Frozen Qwen backbone differs from shared temporal anchor')
            processor, model, head, scores, all_rows, _ = original.load_runtime(
                SimpleNamespace(model=args.model,checkpoint=args.checkpoint,manifest=args.manifest,image_size=448))
            if entry.ordered_id_sha(all_rows[:200]) != entry.ordered_id_sha(rows):
                raise ValueError('Original TEST order differs')
            prompt = core.render_prompt(processor, 'spatial')
            if core.PROMPTS['spatial'] != reference['prompt']:
                raise ValueError('Original spatial prompt changed')
            model.requires_grad_(False)
            head.requires_grad_(False)
            identity.update(assets=assets, prompt=reference['prompt'], processed_frame_size=[448,448],
                floating_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
                quality_readout_dtypes=sorted({str(p.dtype) for p in head.parameters() if p.is_floating_point()}),
                original_full_vocabulary_projection=True,
                input_files_metadata_before=input_metadata,
                timer='first video file preparation through200th CPU score, all199 internal gaps')
            write(args.output/'identity.json',identity)
            timer = core.BoundaryTimer(model)
            def prepare(selected):
                return original.prepare_batch(selected,processor,prompt,torch.device('cpu'),448)
            def transfer(cpu):
                return {key:value.to('cuda:0') for key,value in cpu.items()}
            def forward(inputs):
                _, predictions, timing = original.forward_batch(inputs,model,head,scores,timer)
                return predictions,timing['vision_first_block_to_score_host_ms'],timing['vision_first_block_to_score_cuda_ms']
            guard.bind()
            raw = args.output/'telemetry_stdout.jsonl'
            sampler = ProcessTelemetrySampler(GPU,raw,command=observer_command(GPU,raw))
            sampler.start();sampler.set_phase('continuous200_spatial_no_warmup')
            time.sleep(.15)  # Observer bracketing, no model forward.
            collect(rows,prepare,transfer,forward,lambda:timer.state['vision_host_started'],
                torch.cuda.synchronize,lambda:(guard.fast(),sampler.check()),
                lambda row,cpu:evidence.append(dict(sample_id=row['sample_id'],**entry.cpu_input_evidence(cpu))),
                lambda n:status('evaluating',completed_samples=n),calls)
            sampler.request('barrier');time.sleep(.12)
            telemetry=sampler.stop();sampler=None
            cost=entry.summarize(calls,telemetry)
            performance=core.metrics(np.array([r['target_mos'] for r in calls]),np.array([r['prediction'] for r in calls]))
            if entry.file_identities(rows) != input_metadata:
                raise ValueError('Input file metadata changed')
            identity['input_files']=entry.file_identities(rows,hash_content=True)
            audit=dict(samples=evidence,original_video_metadata=entry.input_video_metadata(rows),
                metadata_and_content_hashes_read_after_T1=True,no_predecoded_video_cache=True)
            write(args.output/'input_audit.json',audit)
            identity['input_audit_sha256']=entry.sha(args.output/'input_audit.json')
            identity['effective_input_tokens200']=sum(r['effective_input_tokens'] for r in evidence)
            if old.costs.model_hashes(args.model)!=assets:
                raise ValueError('Backbone assets changed')
            guard.check()
            report=dict(status='complete',**{k:v for k,v in identity.items() if k!='status'},
                measured_videos=200,cost=cost,performance200=performance,
                mean_board_power_w=cost['gpu_energy_j']/cost['continuous_wall_s'],
                full558_reference_is_historical=True,power_is_fixed_host_estimate_plus_full_window_board_integral=True)
            write(args.output/'report.json',report);status('complete',completed_samples=200)
            print(json.dumps({k:report[k] for k in ('status','performance200','cost','mean_board_power_w')}))
    except BaseException as error:
        status('failed',error=repr(error));raise
    finally:
        if timer is not None:timer.close()
        if sampler is not None:telemetry=sampler.stop(raise_on_error=False)
        if telemetry:save_samples(args.output/'telemetry.csv',telemetry)
        old.write_csv(args.output/'calls.csv',calls)
        if not (args.output/'input_audit.json').exists():write(args.output/'input_audit_partial.json',dict(samples=evidence))
        write(args.output/'identity.json',identity)
        guard.close(args.output/'process_audit.json')
        write(args.output/'files_manifest.json',[dict(file=p.name,sha256=entry.sha(p),bytes=p.stat().st_size)
            for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='files_manifest.json'])


if __name__=='__main__':main()
