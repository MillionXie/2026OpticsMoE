"""Checkpoint-only audit. No optimizer steps, no checkpoint writes, no timing claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import torch

from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing import training as legacy
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.metrics import MetricAccumulator
from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.objectives import editing_objective
from .modeling import build_model
from .run import PROFILES, TASK_DIR
from .settings import load_settings
from .training import _set_phase_dropout


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def phase_delta(a: torch.Tensor, b: torch.Tensor) -> dict:
    pa, pb = [2 * math.pi * x.detach().float().sigmoid() for x in (a, b)]
    delta = torch.remainder(pa - pb + math.pi, 2 * math.pi) - math.pi
    centered = torch.remainder(delta - torch.angle(torch.exp(1j * delta).mean()) + math.pi, 2 * math.pi) - math.pi
    return {'raw_rms': float((a.float() - b.float()).square().mean().sqrt()),
            'wrapped_phase_rms_rad': float(delta.square().mean().sqrt()),
            'piston_removed_phase_rms_rad': float(centered.square().mean().sqrt())}


def hybrid(args, settings, device):
    train_loader, test_loader = legacy.build_loaders(settings)
    model = build_model(settings, device)
    best = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if best['architecture'] != model.checkpoint_architecture:
        raise ValueError('Checkpoint architecture mismatch')
    model.load_state_dict(best['model'], strict=True)
    model.eval()
    _set_phase_dropout(model, False)
    result = {'epoch': best['epoch'], 'architecture': model.architecture_report()}
    result['alpha'] = {name: [float(core.block1_optical_fusion), float(core.block2_optical_fusion)]
                       for name, core in [('language', model.language_core), ('vision', model.vision_core)]}
    counts = defaultdict(int)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            counts[name.split('.')[0]] += parameter.numel()
    result['trainable_parameters_by_top_level_module'] = dict(counts)
    result['metrics'], records, _ = legacy._evaluate(model, test_loader, settings, device)
    dump(args.output / 'predictions.json', records)
    for core in (model.language_core, model.vision_core):
        core.set_fusion_ablation('remove_optical')
    result['remove_optical_metrics'], records, _ = legacy._evaluate(model, test_loader, settings, device)
    result['remove_optical_definition'] = 'same weights; E only, coefficient restored to 1; no retraining'
    dump(args.output / 'remove_optical_predictions.json', records)
    for core in (model.language_core, model.vision_core):
        core.set_fusion_ablation('none')
    # Training labels on one fixed train batch: gradient connectivity, not test adaptation.
    batch = legacy._move(next(iter(train_loader)), device)
    model.zero_grad(set_to_none=True)
    output = model(batch['source_image'], batch['prompt_hidden'])
    losses = editing_objective(output, batch, settings)
    # Task losses only: exclude phase/CCD/Router regularizers as evidence of utility.
    loss = (settings.category_loss_weight * losses['category'] + settings.edit_loss_weight * losses['edit']
            + settings.preservation_loss_weight * losses['preservation'] + settings.task_loss_weight * losses['task'])
    loss.backward()
    result['no_task_gradient_parameters'] = {name: parameter.numel() for name, parameter in model.named_parameters()
                                             if parameter.requires_grad and parameter.grad is None}
    # Verify the suspected legacy output adapters affect neither output head.
    # This tests a functional bypass only; it does not mutate/save the model.
    handles = [core.output_adapter.register_forward_hook(lambda _m, _i, value: torch.zeros_like(value))
               for core in (model.language_core, model.vision_core)]
    try:
        with torch.inference_mode():
            bypass = model(batch['source_image'], batch['prompt_hidden'])
        result['legacy_output_adapter_bypass'] = {
            key: float((output[key].detach() - bypass[key]).abs().max())
            for key in ('category_logits', 'edit_logits')}
        result['legacy_output_adapter_bypass']['probe_samples'] = len(batch['sample_id'])
    finally:
        for handle in handles:
            handle.remove()
    last = torch.load(args.last_checkpoint, map_location='cpu', weights_only=False) if args.last_checkpoint else None
    phases = {}
    for name, parameter in model.named_parameters():
        if 'raw_phase' not in name and 'raw_router_phase' not in name:
            continue
        phase = 2 * math.pi * parameter.detach().float().sigmoid()
        relative = torch.angle(torch.exp(1j * (phase - torch.angle(torch.exp(1j * phase).mean()))))
        phases[name] = {'elements': parameter.numel(), 'requires_grad': parameter.requires_grad,
                        'task_gradient_rms': None if parameter.grad is None else float(parameter.grad.float().square().mean().sqrt()),
                        'task_gradient_finite': None if parameter.grad is None else bool(torch.isfinite(parameter.grad).all()),
                        'best_spatial_phase_std_rad': float(relative.std(unbiased=False))}
        if last is not None:
            phases[name]['best_to_last'] = phase_delta(best['model'][name], last['model'][name])
    result['phase_audit'] = phases
    result['gradient_probe'] = {'samples': len(batch['sample_id']), 'split': 'train', 'mode': 'eval/no augmentation',
                                'optimizer_steps': 0, 'loss': 'task losses only; no optical regularizers'}
    if last is not None:
        result['last_checkpoint'] = {'path': str(args.last_checkpoint), 'sha256': sha(args.last_checkpoint), 'epoch': last['epoch']}
    result['phase_caveat'] = 'best-to-last proves later change, not initial-to-best; no archived initialization assumed'
    return result


@torch.inference_mode()
def qwen(args, settings, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from .baseline_structured_5090d import StructuredOpenMojiHead, _prepare_one, _joint_hidden, _resolve_image_token_id
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    head = StructuredOpenMojiHead(payload['hidden_size'], payload['width']).to(device).eval()
    head.load_state_dict(payload['head'], strict=True)
    processor = AutoProcessor.from_pretrained(str(args.qwen_model), min_pixels=224**2, max_pixels=224**2, local_files_only=True)
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(str(args.qwen_model), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval().requires_grad_(False)
    token_id = _resolve_image_token_id(backbone)
    rows = [json.loads(line) for line in settings.test_manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    accumulator, records = MetricAccumulator(), []
    for index, row in enumerate(rows):
        inputs = {key: value.to(device) for key, value in _prepare_one(processor, row, settings.data_dir).items()}
        image, condition = _joint_hidden(backbone, inputs, token_id)
        output = head(image, condition)
        # Accumulator expects an auxiliary classifier absent from this baseline.
        # Supply a placeholder solely to reuse all grid metrics; discard task_accuracy below.
        output['task_logits'] = torch.zeros(1, 4, device=device)
        source = torch.tensor([row['source_grid']], device=device)
        target = torch.tensor([row['target_grid']], device=device)
        batch = {'source_grid': source, 'target_grid': target, 'edit_grid': source.ne(target), 'preserve_grid': source.eq(target),
                 'task_index': torch.tensor([row['task_index']], device=device), 'task': [row['task']],
                 'sample_id': [row['sample_id']], 'instruction': [row['instruction']]}
        sample_records, _, _ = accumulator.update(output, batch)
        for record in sample_records:
            record.pop('task_accuracy')
        records.extend(sample_records)
        if (index + 1) % 100 == 0:
            print(f'Qwen fresh online evaluation: {index+1}/{len(rows)}', flush=True)
    metrics = accumulator.compute()
    for group in metrics.values():
        group['task_accuracy'] = None
    dump(args.output / 'predictions.json', records)
    return {'epoch': payload['epoch'], 'metrics': metrics, 'fresh_online_features': True,
            'qwen_model': str(args.qwen_model), 'head_trainable_parameters': sum(p.numel() for p in head.parameters()),
            'image_hidden_shape': list(image.shape), 'condition_hidden_shape': list(condition.shape),
            'task_accuracy_status': 'not applicable: baseline has no auxiliary task classifier'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['main_dc20', 'd2nn_dc20', 'qwen'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--last-checkpoint', type=Path)
    parser.add_argument('--qwen-model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.method == 'qwen' and args.qwen_model is None:
        parser.error('--qwen-model is required for qwen')
    if args.output.exists():
        raise FileExistsError('Use a fresh audit output directory; existing evidence is never overwritten')
    settings = load_settings(TASK_DIR / 'configs' / PROFILES['main_dc20' if args.method == 'qwen' else args.method])
    settings.num_workers = 0
    args.output.mkdir(parents=True)
    settings.output_dir = args.output
    device = torch.device(args.device)
    torch.manual_seed(settings.seed)
    report = {'method': args.method, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
              'manifest_sha256': sha(settings.test_manifest), 'test_used_for_historical_selection': True,
              'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'speed_and_energy': 'not measured; this is performance/architecture audit only'}
    report.update(qwen(args, settings, device) if args.method == 'qwen' else hybrid(args, settings, device))
    dump(args.output / 'audit.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
