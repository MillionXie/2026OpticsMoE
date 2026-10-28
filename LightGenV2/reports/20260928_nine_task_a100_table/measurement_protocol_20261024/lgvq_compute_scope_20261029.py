"""Three-warm-field operator trace for the new LG compute-only scope.

This records the unchanged real model. The hardware schedule derived from
kernel durations is explicitly a resource-constrained calculation, NOT measured
photonic latency. FFT-context NVML energy is diagnostic, never deployment power.
"""
import ast
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import threading
import time

OPTICAL_S = 1 / 1400 + .000300 + .0000307
PROFILE = 'LG_THREE_WARM_FIELDS_OPERATOR_COMPUTE_SCOPE_20261029'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def processor_adapter(original, capture):
    """Preserve native preprocessing; expose normalized raw pixels and patch AST."""
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source); function = tree.body[0]
    loops = [n for n in ast.walk(function) if isinstance(n, ast.For)
             and any(isinstance(x, ast.Assign) and isinstance(x.value, ast.Call)
                     and ast.unparse(x.value.func) == 'self.rescale_and_normalize' for x in n.body)]
    if len(loops) != 1:
        raise ValueError('Pinned native normalization/patch loop missing')
    loop = loops[0]
    start = next(i + 1 for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'self.rescale_and_normalize')
    stop = next(i + 1 for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'flatten_patches' for t in n.targets))
    suffix = loop.body[start:stop]
    patch = ast.parse('def patch_only(patches,resized_height,resized_width,temporal_patch_size,patch_size,merge_size):\n    pass').body[0]
    patch.body = suffix + ast.parse('return flatten_patches').body
    patch_module = ast.fix_missing_locations(ast.Module(body=[patch], type_ignores=[]))
    namespace = dict(original.__globals__)
    exec(compile(patch_module, '<native_patch_only>', 'exec'), namespace)
    observe = ast.parse('_record_raw_normalized(patches,resized_height,resized_width,temporal_patch_size,patch_size,merge_size)').body[0]
    loop.body.insert(start, observe)
    namespace['_record_raw_normalized'] = capture
    exec(compile(ast.fix_missing_locations(tree), '<observed_native_processor>', 'exec'), namespace)
    return namespace[function.name], namespace['patch_only'], dict(
        original_processor_source_sha256=digest(source.encode()),
        patch_AST_sha256=digest(ast.dump(ast.Module(body=suffix, type_ignores=[])).encode()),
        native_patch_statements_preserved=True)


def prepare_cpu_worker(model_path):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch, cv2
    from transformers import AutoProcessor
    from types import MethodType
    torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    global CPU_PROCESSOR, NORMALIZED, PROCESSOR_RECEIPT
    CPU_PROCESSOR = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    NORMALIZED = []
    def capture(value, h, w, temporal, patch, merge):
        NORMALIZED.append((value.numpy().copy(), h, w, temporal, patch, merge))
    original = type(CPU_PROCESSOR.image_processor)._preprocess
    observed, _, PROCESSOR_RECEIPT = processor_adapter(original, capture)
    CPU_PROCESSOR.image_processor._preprocess = MethodType(observed, CPU_PROCESSOR.image_processor)


def prepare_video(row):
    import torch
    from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 import cache_qwen_front as native
    NORMALIZED.clear(); start = time.perf_counter()
    images = native.decode_frames(Path(row['video_path']), 4); decoded = time.perf_counter()
    inputs = CPU_PROCESSOR(text=['x'] * 4, images=images, return_tensors='pt'); processed = time.perf_counter()
    quality = native.quality_tokens_from_images(images, video_count=1, frame_count=4, output_grid=7)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preparation initialized CUDA')
    if inputs['image_grid_thw'].tolist() != [[1, 28, 28]] * 4:
        raise ValueError('Original four-frame geometry changed')
    pixels = inputs['pixel_values'].contiguous().numpy()
    return dict(sample_id=row['sample_id'], normalized=list(NORMALIZED), quality=quality.numpy(),
                original_flat_sha256=digest(pixels.tobytes()), original_flat_shape=list(pixels.shape),
                processor_receipt=PROCESSOR_RECEIPT, external_preparation=dict(
                    start=start, decode_end=decoded, processor_end=processed, end=time.perf_counter()))


def patch_video(prepared, field, slot, patch_function):
    import torch
    inputs = [(torch.from_numpy(a), h, w, t, p, m) for a, h, w, t, p, m in prepared['normalized']]
    start = time.perf_counter(); cpu_start = time.thread_time()
    chunks = [patch_function(*values) for values in inputs]
    pixels = torch.cat([x for chunk in chunks for x in chunk], dim=0)
    cpu_end = time.thread_time(); end = time.perf_counter()
    if list(pixels.shape) != prepared['original_flat_shape'] or digest(pixels.numpy().tobytes()) != prepared['original_flat_sha256']:
        raise ValueError('Native CPU patch bytes differ')
    return pixels, dict(field=field, slot=slot, worker_thread=threading.get_ident(),
                        host_start=start, host_end=end, thread_cpu_s=cpu_end-cpu_start,
                        raw_patch_bytes_equal=True, sample_id=prepared['sample_id'])


class ScopeMeter:
    def __init__(self, torch):
        self.torch = torch; self.field = -1; self.events = []; self.phase = 'adapter_visual'
        self.active = None; self.pass_count = 0

    @contextmanager
    def scope(self, name, *args, **kwargs):
        phase = {'frontend.text_embedding': 'emb_text', 'frontend.conv_embedding': 'emb_conv3d',
                 'frontend.image_position': 'emb_position', 'frontend.mean_pool': 'adapter_visual_pool'}.get(name, name)
        a = self.torch.cuda.Event(enable_timing=True); b = self.torch.cuda.Event(enable_timing=True)
        start = time.perf_counter(); a.record()
        with self.torch.profiler.record_function(f'LG|{self.field}|{phase}'):
            yield
        b.record(); end = time.perf_counter()
        self.events.append(dict(field=self.field, phase=phase, start=a, end=b,
                                host_start=start, host_end=end))

    def close_electronics(self):
        if self.active is not None:
            self.active.__exit__(None, None, None); self.active = None

    def open_electronics(self):
        self.active = self.scope(self.phase); self.active.__enter__()

    @contextmanager
    def simulation(self, label):
        self.close_electronics(); self.pass_count += 1
        if self.pass_count > 6:
            raise ValueError('More than six real optical passes')
        with self.scope(f'optics{self.pass_count}'):
            yield
        self.phase = f'oeo{self.pass_count}'; self.open_electronics()

    @contextmanager
    def excluded(self, label, kind):
        self.close_electronics()
        with self.scope('excluded_' + kind):
            yield
        self.open_electronics()


def annotate_tail(function, meter, source):
    """Only wrap original merger and decoder statements, preserving their AST."""
    tree = ast.parse(textwrap.dedent(source))
    fn = tree.body[0]
    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    merger_start = next(i for i, n in enumerate(fn.body) if assigned(n, 'image_tokens'))
    merger_stop = next(i for i, n in enumerate(fn.body) if assigned(n, 'fields3'))
    decoder_start = next(i for i, n in enumerate(fn.body) if assigned(n, 'normalized'))
    def wrapped(nodes, phase):
        expression = ast.parse(f'_compute_meter.scope({phase!r})', mode='eval').body
        return ast.With(items=[ast.withitem(context_expr=expression)], body=nodes)
    original_sha = digest(ast.dump(tree).encode())
    fn.body = (fn.body[:merger_start] + [wrapped(fn.body[merger_start:merger_stop], 'merger')]
               + [wrapped([fn.body[merger_stop]], 'adapter_language')]
               + fn.body[merger_stop+1:decoder_start] + [wrapped(fn.body[decoder_start:], 'decoder')])
    namespace = dict(function.__globals__, _compute_meter=meter)
    exec(compile(ast.fix_missing_locations(tree), '<observed_original_tail>', 'exec'), namespace)
    return namespace[fn.name], original_sha


def traced_kernels(trace):
    """Map each GPU kernel through its original external id to a user range."""
    events = trace['traceEvents']; cpu = defaultdict(list); scopes = defaultdict(list)
    for e in events:
        if e.get('ph') != 'X':
            continue
        if e.get('name', '').startswith('LG|'):
            scopes[(e.get('pid'), e.get('tid'))].append(e)
        external = e.get('args', {}).get('External id')
        if external is not None and e.get('cat') in ('cpu_op', 'user_annotation'):
            cpu[external].append(e)
    for values in scopes.values(): values.sort(key=lambda e:e['ts'])
    starts = {key:[e['ts'] for e in values] for key,values in scopes.items()}
    kernels = []; unmapped = []
    for e in events:
        if e.get('cat') != 'kernel' or e.get('ph') != 'X':
            continue
        candidates = []
        for op in cpu.get(e.get('args', {}).get('External id'), []):
            key = (op.get('pid'), op.get('tid'))
            values = scopes.get(key, [])
            index = bisect_right(starts.get(key, []),op['ts'])
            for scope in reversed(values[:index]):
                if scope['ts'] <= op['ts'] and op['ts'] + op.get('dur', 0) <= scope['ts'] + scope['dur'] + .01:
                    candidates.append(scope); break
        if not candidates:
            unmapped.append(e); continue
        scope = min(candidates, key=lambda x: x['dur'])
        _, field, phase = scope['name'].split('|', 2)
        kernels.append(dict(field=int(field), phase=phase, name=e['name'],
                            start_us=e['ts'], end_us=e['ts']+e['dur'], stream=e.get('args', {}).get('stream'),
                            external_id=e.get('args', {}).get('External id')))
    if unmapped:
        raise ValueError(f'{len(unmapped)} GPU kernels have no witnessed phase; preserve raw trace')
    if not kernels:
        raise ValueError('No device kernel trace; no pure-compute time may be claimed')
    return kernels


def union_seconds(intervals):
    total = 0.; end = None
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError('Negative interval')
        total += b-max(a, end) if end is not None and b > end else (b-a if end is None else 0.)
        end = max(end, b) if end is not None else b
    return total / 1e6


def compute_schedule(nodes):
    """Deterministic earliest-ready nonpreemptive resource schedule, no CPU gaps."""
    pending = {n['id']: n for n in nodes}; done = {}; resource_last = {}; order = []
    if len(pending)!=len(nodes) or any(n['duration_s']<0 for n in nodes):
        raise ValueError('Unique nonnegative computation nodes required')
    while pending:
        options = []
        for key, n in pending.items():
            if not all(p in done for p in n['parents']):
                continue
            resource = resource_last.get(n['resource'])
            parents = list(n['parents']) + ([resource] if resource else [])
            start = max((done[p]['end_s'] for p in parents), default=0.)
            options.append((start, n['field'], n['sequence'], key, parents))
        if not options:
            raise ValueError('Cyclic dependency')
        start, _, _, key, parents = min(options)
        n = pending.pop(key); n = dict(n, parents=list(dict.fromkeys(parents)), start_s=start, end_s=start+n['duration_s'])
        done[key] = n; resource_last[n['resource']] = key; order.append(n)
    terminal = max(done, key=lambda k: (done[k]['end_s'], k)); chain = []
    while True:
        chain.append(terminal); n = done[terminal]
        if not n['parents']:
            break
        terminal = max(n['parents'], key=lambda p: (done[p]['end_s'], p))
    wall = max(n['end_s'] for n in order)
    if abs(sum(done[k]['duration_s'] for k in chain)-wall) > 1e-9:
        raise ValueError('Additive equivalent contributions do not close')
    return dict(nodes=order, total_s=wall, additive_path=list(reversed(chain)))


def summarize_compute(kernels, patch_rows, token_rows, fields):
    grouped = defaultdict(list)
    for k in kernels:
        if 0 <= k['field'] < fields:
            grouped[(k['field'], k['phase'])].append((k['start_us'], k['end_us']))
    phase = {key: union_seconds(values) for key, values in grouped.items()}
    required = ['emb_text','emb_conv3d','emb_position','adapter_visual_pool','adapter_visual',
                'merger','adapter_language','decoder'] + [f'oeo{i}' for i in range(1,7)] + [f'optics{i}' for i in range(1,7)]
    for field in range(fields):
        if any((field, key) not in phase for key in required):
            raise ValueError('Missing measured kernel block: '+str(field))
    nodes=[]; raw=[]
    for field in range(fields):
        patches = [r for r in patch_rows if r['field']==field]
        if len(patches)!=16 or not all(r['raw_patch_bytes_equal'] for r in patches):
            raise ValueError('Sixteen exact native patch preparations required')
        # Keep the observed CPU worker assignment, not an invented 16x division.
        patch_ids=[]
        for r in patches:
            key=f'f{field}.patch{r["slot"]}';patch_ids.append(key)
            nodes.append(dict(id=key,field=field,sequence=0,phase='patch',resource='cpu_patch_'+str(r['worker_thread']),
                              parents=[f'f{field-2}.decoder'] if field>=2 else [],duration_s=r['thread_cpu_s']))
        previous=f'f{field}.tokenizer'
        nodes.append(dict(id=previous,field=field,sequence=0,phase='tokenizer',resource='cpu_tokenizer',
                          parents=[f'f{field-2}.decoder'] if field>=2 else [],
                          duration_s=token_rows[field]['thread_cpu_s']))
        stages=['emb_text','emb_conv3d','emb_position','adapter_visual_pool','adapter_visual']
        for i in range(1,7):
            stages += [f'optics{i}',f'oeo{i}']
            if i==3:stages+=['merger','adapter_language']
        stages+=['decoder']
        for sequence,key in enumerate(stages,1):
            optical=key.startswith('optics');duration=OPTICAL_S if optical else phase[(field,key)]
            parents=[previous]
            if key=='emb_conv3d':parents+=patch_ids
            current=f'f{field}.{key}'
            nodes.append(dict(id=current,field=field,sequence=sequence,phase=key,resource='optical' if optical else 'GPU',
                              parents=parents,duration_s=duration))
            raw.append(dict(field=field,phase=key,kernel_union_s=phase[(field,key)],charged_compute_s=duration))
            previous=current
    schedule=compute_schedule(nodes);by={n['id']:n for n in schedule['nodes']};equivalent=defaultdict(float)
    for key in schedule['additive_path']:equivalent[by[key]['phase']]+=by[key]['duration_s']/fields
    return dict(status='COMPUTE_SCOPE_RESOURCE_SCHEDULE_ESTIMATE',profile=PROFILE,fields=fields,video_calls=fields*16,
                time16_s=schedule['total_s']/fields,raw_phases=raw,equivalent_s_per_group=dict(equivalent),
                schedule=schedule,optical_layer_s=OPTICAL_S,propagation_distance_m=.5,propagation_time_ignored=True,
                physical_hardware_measured=False,CPU_dispatch_and_queues_excluded=True,
                electronic_residual_and_fusion_charged=False,software_execution_kept_for_numerical_identity=True,
                cross_field_hardware_pipeline_is_modelled_not_measured=True,
                energy_qualified=False,energy_reason='FFT-context board power is diagnostic, not hybrid deployment power')
