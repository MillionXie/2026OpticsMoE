"""CPU-only 1/16-worker LGVQ diagnostic, not a new LG latency result.

Use original first16 video bytes, decode/crop before timing, and re-execute
the accepted resize/normalization/BF16 patch math. No model or CUDA calls.
Warm one group per mode, then five groups; validation follows each group.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


def sha(data):
    return hashlib.sha256(data).hexdigest()


def summarize(groups):
    result = {}
    for workers in (1, 16):
        selected = [g for g in groups if g['workers'] == workers]
        result[str(workers)] = {
            'rounds': len(selected),
            'group_compute_wall_median_ms': statistics.median(g['compute_wall_ms'] for g in selected),
            'group_compute_wall_range_ms': [min(g['compute_wall_ms'] for g in selected), max(g['compute_wall_ms'] for g in selected)],
            'mean_thread_cpu_per_video_ms': statistics.mean(r['thread_cpu_ms'] for g in selected for r in g['rows']),
            'stage_cpu_per_video_ms': {k: statistics.mean(r[k] for g in selected for r in g['rows']) for k in ('resize_ms', 'normalization_ms', 'patch_ms')},
            'mean_process_cpu_per_group_ms': statistics.mean(g['process_cpu_ms'] for g in selected),
        }
    return result


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must be hidden before imports')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU diagnostic300s limit')))
    signal.alarm(300)
    protocol = Path('LightGenV2/reports/20260928_nine_task_a100_table/measurement_protocol_20261024')
    sys.path[:0] = [str(args.repo), str(args.repo / protocol)]
    names = [protocol / x for x in ('lgvq_resize_norm_20261029.py', 'lgvq_compute_scope_20261029.py', 'lgvq_patch_layout_20261029.py')]
    names.append(Path('experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/cache_qwen_front.py'))
    source_sha = {}
    for name in names:
        original = subprocess.check_output(['git', 'show', args.commit + ':' + name.as_posix()], cwd=args.repo, timeout=10)
        if sha(original) != sha((args.repo / name).read_bytes()):
            raise ValueError('Installed dependency differs from pinned Git: ' + str(name))
        source_sha[name.as_posix()] = sha(original)
    self_name = protocol / 'benchmark_lgvq_cpu_pair_20261031.py'
    source_sha[self_name.as_posix()] = sha(subprocess.check_output(['git', 'show', args.commit + ':' + self_name.as_posix()], cwd=args.repo, timeout=10))
    # Same cooperative lock as temporal measurement: do not contend with it.
    lock = (args.repo / 'LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/.temporal_telemetry_a100.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(exist_ok=False)
    def save(name, value):
        with (args.output / name).open('x') as f:
            json.dump(value, f, ensure_ascii=True, indent=2, allow_nan=False)
    save('started.json', dict(pid=os.getpid(), unix=time.time(), command=sys.argv, commit=args.commit))
    import lgvq_resize_norm_20261029 as prep
    import lgvq_patch_layout_20261029 as layout
    import torch
    import cv2
    from PIL import Image
    prep.prepare_cpu_worker(str(args.model))
    original = json.loads(args.preparation.read_text())['rows'][:16]
    if len(original) != 16 or len({r['sample_id'] for r in original}) != 16:
        raise ValueError('Exactly16 original distinct videos')
    inputs = []
    for row in original:
        if sha(Path(row['video_path']).read_bytes()) != row['sha256']:
            raise ValueError('Original video bytes changed')
        item = prep.prepare_video(row)
        if item['original_flat_sha256'] != row['original_flat_sha256']:
            raise ValueError('Original processor pixels changed')
        inputs.append(item)
    receipts = [dict(sample_id=x['sample_id'], raw_crops_sha256=[sha(a.tobytes()) for a in x['raw_rgb_crops']], raw_shapes=[list(a.shape) for a in x['raw_rgb_crops']], original_flat_sha256=x['original_flat_sha256']) for x in inputs]
    def calculate(slot):
        item = inputs[slot]
        proc = prep.processor(str(args.model))
        prep.LOCAL.processor_resize_cpu_s = 0.
        wall = time.perf_counter(); start = time.thread_time()
        resized = [cv2.resize(a, (448, 448), interpolation=cv2.INTER_AREA) for a in item['raw_rgb_crops']]
        images = [Image.fromarray(a) for a in resized]
        resize_end = time.thread_time()
        values = proc(images=images, return_tensors='pt')
        norm_end = time.thread_time()
        if len(values) != 1:
            raise ValueError('Original one four-frame chunk')
        value, h, w, t, p, m = values[0]
        output = layout.strided_patch(value, h, w, t, p, m)
        end = time.thread_time(); wall_end = time.perf_counter()
        cpu_resize = resize_end - start + prep.LOCAL.processor_resize_cpu_s
        norm = norm_end - resize_end - prep.LOCAL.processor_resize_cpu_s
        return output, dict(slot=slot, thread=threading.get_ident(), start=wall, end=wall_end,
            thread_cpu_ms=(end-start)*1000, resize_ms=cpu_resize*1000,
            normalization_ms=norm*1000, patch_ms=(end-norm_end)*1000)
    # Establish reference bytes outside every measurement.
    reference = []
    for item in inputs:
        expected = item['expected_normalized'][0]
        # Native FP32 layout is already checked by prepare_video; also verify
        # the accepted optimized BF16 output against native patch extraction.
        output, _ = calculate(len(reference))
        proc = prep.processor(str(args.model));prep.LOCAL.processor_resize_cpu_s = 0.
        ims = [Image.fromarray(cv2.resize(a, (448,448), interpolation=cv2.INTER_AREA)) for a in item['raw_rgb_crops']]
        value,h,w,t,p,m = proc(images=ims,return_tensors='pt')[0]
        if layout.digest_tensor(value) != expected['sha256']:
            raise ValueError('Normalized pixel identity failed')
        _, patch_fn, _ = prep.scope.processor_adapter(type(prep.scope.CPU_PROCESSOR.image_processor)._preprocess, lambda *_: None)
        native = patch_fn(value,h,w,t,p,m).reshape(-1,3*t*p*p)
        if layout.digest_tensor(native) != item['original_flat_sha256'] or not torch.equal(output.view(torch.int16), native.to(torch.bfloat16).view(torch.int16)):
            raise ValueError('Exact BF16 layout identity failed')
        reference.append(layout.digest_tensor(output))
    before = dict(load=list(os.getloadavg()), affinity=sorted(os.sched_getaffinity(0)))
    groups = []
    # Alternate modes to reduce directional temperature/load drift.
    for repeat in range(6):
        for workers in ((1,16) if repeat % 2 == 0 else (16,1)):
            process_stamp = {}
            # Last arrival stamps after every thread-local initializer. Earlier
            # b595 raw process_cpu_ms included those initializers; its valid
            # thread CPU / compute-span clocks are audited separately.
            barrier = threading.Barrier(workers + 1, action=lambda: process_stamp.update(start=time.process_time()))
            def job(slots):
                prep.processor(str(args.model))  # thread-local initialization untimed
                barrier.wait(timeout=60)
                return [calculate(slot) for slot in slots]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(job, list(range(i,16,workers))) for i in range(workers)]
                barrier.wait(timeout=60)
                process_start = process_stamp['start']
                values = [v for f in futures for v in f.result(timeout=60)]
                process_end = time.process_time()
            values.sort(key=lambda x:x[1]['slot'])
            for output, row in values:
                if layout.digest_tensor(output) != reference[row['slot']]:
                    raise ValueError('Measured output changed')
            rows = [r for _,r in values]
            if workers == 16 and len({r['thread'] for r in rows}) != 16:
                raise ValueError('16 actual concurrent workers required')
            if repeat:
                groups.append(dict(workers=workers,repeat=repeat,compute_wall_ms=(max(r['end'] for r in rows)-min(r['start'] for r in rows))*1000,process_cpu_ms=(process_end-process_start)*1000,rows=rows))
            del values
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA unexpectedly initialized')
    packages = {x: importlib.metadata.version(x) for x in ('torch','transformers','numpy','pillow')}
    packages['opencv_runtime'] = cv2.__version__
    cgroup = {x:(Path('/sys/fs/cgroup')/x).read_text() if (Path('/sys/fs/cgroup')/x).exists() else None for x in ('cpu.max','cpu.stat','cpuset.cpus.effective')}
    result = dict(status='complete',commit=args.commit,sources=source_sha,input_preparation_sha256=sha(args.preparation.read_bytes()),inputs=receipts,
        cpu_model=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),
        packages=packages,python=sys.version,before=before,after_load=list(os.getloadavg()),cgroup=cgroup,
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),cv2_threads=cv2.getNumThreads(),
        one_warm_group_per_mode=True,formal_groups_per_mode=5,raw_groups=groups,summary=summarize(groups),
        cuda_initialized=False,model_loaded=False,scope='same in-memory real four-frame crops; resize/PIL, native normalization, accepted BF16 patch; no decoding/GPU/optics',
        group_wall_includes_actual_GIL_OS_resource_wait=True,thread_cpu_excludes_off_CPU_wait=True,
        no_LG_table_or_physical_latency_replacement=True,unix=time.time())
    save('cpu_benchmark.json',result)
    print(json.dumps(dict(status=result['status'],cpu=result['cpu_model'],packages=packages,summary=result['summary'],output=str(args.output)),ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
