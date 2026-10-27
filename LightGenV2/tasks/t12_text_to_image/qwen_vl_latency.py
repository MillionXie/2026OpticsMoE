"""B1 raw-input timing of the full Qwen-VL spatial electronic baseline.

Keeps Qwen and the generator resident together. No cached hidden states, target,
reference VAE encoder, CLIP, model reload, or metrics inside inference. Diagnostic
hooks have no intermediate synchronize; the primary pass has no hooks at all.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def summarize(values):
    values = sorted(values)
    if not values:
        raise ValueError('Empty latency population')
    return {'count': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values),
            'p95': values[min(len(values)-1, int(.95*(len(values)-1)))],
            'min': values[0], 'max': values[-1]}


def validate_inputs(records):
    allowed = {'image', 'prompt', 'sample_id', 'category', 'mode'}
    if not records or any(set(r)-allowed for r in records):
        raise ValueError('Only input RGB/prompt/identity fields allowed')
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Duplicate sample identities')
    return records


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from .qwen_vl_spatial import build_spatial_head
    from .qwen_vl_turbo_data import sha256, write_json

    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'manifest', 'qwen', 'sd', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seed', type=int, default=1042)
    p.add_argument('--memory-fraction', type=float, default=.5)
    p.add_argument('--hybrid-checkpoint', type=Path)
    p.add_argument('--tail-checkpoint', type=Path)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu_uuid:
        raise ValueError('GPU UUID must match the authorized device')
    if a.rounds < 1 or (a.output/'report.json').exists():
        raise ValueError('Invalid rounds or existing report; do not overwrite')
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, 0)
    device = torch.device('cuda:0')
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if state.get('schema') != 'qwen_spatial_v1':
        raise ValueError('Expected the full-Qwen spatial checkpoint')
    cfg, layout = state['config'], state['visual_layout']
    rows = validate_inputs(json.loads(a.manifest.read_text(encoding='utf-8')))
    images = []
    for r in rows:
        with Image.open(r['image']) as im:
            im = im.convert('RGB')
            if im.size != (256, 256):
                raise ValueError('Native 256x256 input required')
            images.append(im.copy())
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(
        a.qwen, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True).to(device).eval().requires_grad_(False)
    processor = AutoProcessor.from_pretrained(a.qwen, local_files_only=True,
                                            min_pixels=65536, max_pixels=65536)
    head = build_spatial_head(a.sd, state, cfg, layout['positions'], device,
                             drop_vae_encoder=True).eval().requires_grad_(False)
    if a.hybrid_checkpoint:
        from .qwen_vl_hybrid import apply_configured_hybrid, load_macro_state, restore_head_overlay
        hybrid = torch.load(a.hybrid_checkpoint, map_location='cpu', weights_only=False)
        macros = apply_configured_hybrid(qwen,hybrid['config'])
        load_macro_state(macros, hybrid['macros']); qwen.eval().requires_grad_(False)
        restore_head_overlay(head,hybrid)
    if a.tail_checkpoint:
        if not a.hybrid_checkpoint:
            raise ValueError('Compressed-tail protocol requires its hybrid Qwen identity')
        from .qwen_vl_tail import build_tail_student
        tail = torch.load(a.tail_checkpoint, map_location='cpu', weights_only=False)
        if tail['hybrid_checkpoint_sha256'] != sha256(a.hybrid_checkpoint):
            raise ValueError('Tail and hybrid weights do not match')
        smaller, _ = build_tail_student(head.unet, tail['profile']['channels'], tail['profile']['heads'])
        smaller.load_state_dict(tail['unet']); head.unet = smaller.to(device).eval().requires_grad_(False)
    vision, language = qwen.model.visual, qwen.model.language_model
    if len(vision.blocks) != 24 or len(language.layers) != 28:
        raise ValueError('Wrong full-Qwen architecture')
    expected_positions = layout.get('qwen_image_positions', layout['positions'])
    torch.cuda.synchronize()
    execution = {'argv': __import__('sys').argv, 'pid': os.getpid(),
                 'started_unix': time.time(), 'checkpoint_sha256': sha256(a.checkpoint),
                 'source_sha256': sha256(Path(__file__)),
                 'source_commit': os.environ.get('T12_SOURCE_COMMIT'),
                 'gpu': subprocess.check_output(['nvidia-smi', '-i', a.gpu_uuid,
                         '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']).decode().strip(),
                 'torch': torch.__version__, 'cuda': torch.version.cuda,
                 'batch_size': 1, 'extra_warmups': 0,
                 'training_source_commit': state['git']['published_source_commit']}
    execution.update(hybrid_sha256=sha256(a.hybrid_checkpoint) if a.hybrid_checkpoint else None,
                     tail_sha256=sha256(a.tail_checkpoint) if a.tail_checkpoint else None,
                     optical_path_execution='GPU FFT simulation' if a.hybrid_checkpoint else 'none')
    write_json(a.output/'execution.json', execution)

    def prepare(index):
        message = [{'role': 'user', 'content': [
            {'type': 'image', 'image': images[index]},
            {'type': 'text', 'text': rows[index]['prompt']}]}]
        text = processor.apply_chat_template(message, tokenize=False,
                        add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=[images[index]], return_tensors='pt')
        positions = torch.where(inputs['input_ids'][0] == layout['image_token_id'])[0].tolist()
        if positions != expected_positions or inputs['image_grid_thw'].tolist() != [layout['unmerged_grid']]:
            raise ValueError('Raw processor layout differs from trained layout')
        return inputs.to(device)

    native = []
    handles = []
    # Compatibility for old native-visual-prefix checkpoints. This forward hook
    # keeps tensors on GPU; current fused-only checkpoint has prefix=0.
    if cfg.get('semantic_prefix', 0):
        handles.append(vision.register_forward_hook(
            lambda module, args, out: native.append(out[0].detach().bfloat16())))

    @torch.inference_mode()
    def forward(inputs):
        native.clear()
        output = qwen.model(**inputs, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state[:, inputs['attention_mask'][0].bool()].bfloat16()
        if cfg.get('semantic_prefix', 0):
            if len(native) != 1 or native[0].shape != (64, 2048):
                raise ValueError('Native visual prefix contract differs')
            hidden = torch.cat([native[0][None], hidden], 1)
        valid = torch.ones(hidden.shape[:2], device=device, dtype=torch.bool)
        noise = torch.randn(1, 4, 32, 32, device=device,
                    generator=torch.Generator(device=device).manual_seed(a.seed))
        _, rgb = head(hidden, valid, noise)
        return rgb

    def cpu_rgb(rgb):
        return ((rgb[0].clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()

    # Primary continuous wall-clock, no stage hooks. Includes first execution.
    records, fingerprints = [], {}
    for repeat in range(a.rounds):
        for index, row in enumerate(rows):
            torch.cuda.synchronize()
            start = time.perf_counter()
            inputs = prepare(index)
            torch.cuda.synchronize()
            ready = time.perf_counter()
            e0, e1 = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            e0.record()
            rgb = forward(inputs)
            e1.record()
            e1.synchronize()
            gpu_done = time.perf_counter()
            array = cpu_rgb(rgb)
            done = time.perf_counter()
            fingerprint = hashlib.sha256(array.tobytes()).hexdigest()
            if index in fingerprints and fingerprints[index] != fingerprint:
                raise ValueError('Repeated raw-input inference was not deterministic')
            fingerprints[index] = fingerprint
            records.append({'round': repeat, 'sample_id': row['sample_id'],
                'category': row.get('category'), 'mode': row.get('mode'),
                'sequence_tokens': int(inputs['input_ids'].shape[1]),
                'preprocess_h2d_ms': (ready-start)*1000,
                'gpu_ready_to_gpu_rgb_wall_ms': (gpu_done-ready)*1000,
                'gpu_stream_span_ms': e0.elapsed_time(e1),
                'rgb_cpu_conversion_ms': (done-gpu_done)*1000,
                'rgb_text_ready_to_cpu_rgb_ms': (done-start)*1000,
                'rgb_sha256': fingerprint})
        write_json(a.output/'progress.json', {'stage': 'primary', 'rounds_done': repeat+1})

    # Diagnostic second pass: no intermediate sync, per-layer stream events and
    # first-TF wall marker. Never mix this hook overhead with the primary pass.
    events, start_wall = {}, {}
    def pre(name):
        def hook(module, args):
            events[name] = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
            events[name][0].record()
            if name == 'vision.block.0':
                start_wall['first_tf'] = time.perf_counter()
        return hook
    def post(name):
        return lambda module, args, out: events[name][1].record()
    modules = {**{f'vision.block.{i}': b for i, b in enumerate(vision.blocks)},
               **{f'language.block.{i}': b for i, b in enumerate(language.layers)},
               'qwen.visual': vision, 'qwen.language': language,
               'head.spatial': head.spatial, 'head.adapter': head.adapter,
               'head.unet': head.unet, 'head.vae_decoder': head.vae.decoder}
    restored_methods = []
    if a.hybrid_checkpoint:
        for modality, macro in macros.items():
            for i, module in enumerate(macro.electronic):
                modules[f'hybrid.{modality}.electronic_residual.{i}'] = module
            for j, path in enumerate(macro.paths):
                original = path._t12_physical_simulate
                def detector(*args, _original=original, _modality=modality, _j=j, **kwargs):
                    name = f'optical.{_modality}.{_j}.'+kwargs['phase_support']
                    pre(name)(None, None)
                    out = _original(*args, **kwargs)
                    post(name)(None, None, out)
                    return out
                restored_methods.append((path, '_t12_physical_simulate', original))
                path._t12_physical_simulate = detector
            router = macro.paths[0].core.router
            original = router._t12_physical_simulate
            def route(*args, _original=original, _modality=modality, **kwargs):
                name = f'optical.{_modality}.router'
                pre(name)(None, None)
                out = _original(*args, **kwargs)
                post(name)(None, None, out)
                return out
            restored_methods.append((router, '_t12_physical_simulate', original))
            router._t12_physical_simulate = route
    for name, module in modules.items():
        handles += [module.register_forward_pre_hook(pre(name)), module.register_forward_hook(post(name))]
    diagnostics = []
    try:
        for index, row in enumerate(rows):
            inputs = prepare(index)
            torch.cuda.synchronize()
            events.clear(); start_wall.clear()
            rgb = forward(inputs)
            final = torch.cuda.Event(enable_timing=True); final.record(); final.synchronize()
            gpu_done = time.perf_counter()
            array = cpu_rgb(rgb)
            if hashlib.sha256(array.tobytes()).hexdigest() != fingerprints[index]:
                raise ValueError('Diagnostic hooks changed the RGB result')
            diagnostics.append({'sample_id': row['sample_id'],
                'first_tf_to_gpu_rgb_wall_ms': (gpu_done-start_wall['first_tf'])*1000,
                'first_tf_to_gpu_rgb_stream_ms': events['vision.block.0'][0].elapsed_time(final),
                'stages_ms': {k: x.elapsed_time(y) for k, (x, y) in events.items()}})
    finally:
        for handle in handles:
            handle.remove()
        for obj, key, method in restored_methods:
            setattr(obj, key, method)
    fields = ['preprocess_h2d_ms', 'gpu_ready_to_gpu_rgb_wall_ms', 'gpu_stream_span_ms',
              'rgb_cpu_conversion_ms', 'rgb_text_ready_to_cpu_rgb_ms']
    summaries = {key: summarize([r[key] for r in records]) for key in fields}
    stages = {key: summarize([r['stages_ms'][key] for r in diagnostics]) for key in modules}
    # Idealized floor: removes ALL Qwen TF cost, zero replacement overhead. It
    # is a feasibility diagnostic, NOT a prediction of the future optical model.
    tails = [r['stages_ms']['head.unet']+r['stages_ms']['head.vae_decoder'] for r in diagnostics]
    qtf = [sum(v for k, v in r['stages_ms'].items() if '.block.' in k) for r in diagnostics]
    baseline = statistics.mean(r['first_tf_to_gpu_rgb_wall_ms'] for r in diagnostics)
    report = {'complete': True, 'execution': execution, 'records': records,
        'summaries_ms': summaries, 'diagnostics': diagnostics, 'stages_ms': stages,
        'first_tf_to_gpu_rgb_wall_ms': summarize([r['first_tf_to_gpu_rgb_wall_ms'] for r in diagnostics]),
        'first_tf_to_gpu_rgb_stream_ms': summarize([r['first_tf_to_gpu_rgb_stream_ms'] for r in diagnostics]),
        'parameters': {'qwen': parameter_count(qwen.model), 'head_inference': parameter_count(head),
                       'vision_tf': parameter_count(vision.blocks), 'language_tf': parameter_count(language.layers)},
        'feasibility': {'mean_unet_plus_vae_decoder_ms': statistics.mean(tails),
             'mean_all_qwen_tf_ms': statistics.mean(qtf),
             'ideal_speedup_upper_bound_from_unet_vae_floor': baseline/statistics.mean(tails),
             'note': 'Zero overhead and all TF removed; not a trained-model or hardware speed claim.'},
        'protocol': {'boundary_main': 'PIL RGB and text already resident -> processor/tokenizer + H2D + full Qwen + spatial/semantic heads + single-step UNet + VAE -> CPU uint8 RGB',
             'boundary_user_requested': 'First visual TF pre-hook -> GPU RGB ready, diagnostic pass; preprocessing/embedding before first TF excluded and separately recorded.',
             'loading_disk_metrics_excluded': True, 'hidden_cache_used': False,
             'reference_vae_encoder': False, 'targets_or_masks_input': False,
             'no_extra_warmup_first_request_retained': True, 'batch_size': 1,
             'main_uses_stage_hooks': False, 'diagnostic_no_intermediate_synchronize': True,
             'rgb_outputs_hooked_and_unhooked_identical': True,
             'selection_test_independent': False, 'hardware_optical_estimate_included': False}}
    if a.hybrid_checkpoint:
        cycles = sum(1+2*m.pairs for m in macros.values())
        optical_ms = [sum(v for k, v in d['stages_ms'].items() if k.startswith('optical.')) for d in diagnostics]
        residual_ms = [sum(v for k, v in d['stages_ms'].items() if k.startswith('hybrid.') and '.electronic_residual.' in k) for d in diagnostics]
        estimate = [d['first_tf_to_gpu_rgb_stream_ms']-o-e+cycles*(1000/1400+.3+.0307)
                    for d, o, e in zip(diagnostics, optical_ms, residual_ms)]
        report['optical_timing_model'] = {'cycles': cycles, 'cycle_ms': 1000/1400+.3+.0307,
            'cycle_total_ms': cycles*(1000/1400+.3+.0307),
            'measured_fft_simulator_ms': summarize(optical_ms),
            'user_excluded_electronic_residual_ms': summarize(residual_ms),
            'user_scope_serial_cuda_span_substitution_ms': summarize(estimate),
            'actual_hardware_measured': False,
            'not_final_end_to_end_dag_claim': True,
            'limitations': 'Single CUDA stream diagnostic interval substitution only. CPU submission/optical device dependency changes, transfers and OEO require final DAG audit. Bounded amplitude encoding and readout remain outside excluded optical methods. Never call FFT simulation latency physical optical latency.'}
    write_json(a.output/'report.json', report)
    print(json.dumps({'complete': True, 'summaries_ms': summaries, 'feasibility': report['feasibility']}, indent=2))


if __name__ == '__main__':
    main()
