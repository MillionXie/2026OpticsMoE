"""Export one T08 stage using all preceding measured optical results.

CCD arrays are canonical raw uint8 from the bench. Router scores are computed
from the four trained detector windows; low PCC/brightness never suppresses a
sample. This is a resumable export, not a simulation-only replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from export_router_pilot import sha, EXPECTED
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs, preprocess_images, student_embeddings, validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import GroceryRetrievalDataset
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import load_checkpoint
from LightGenV2.tasks.t01_object_retrieval.modeling import build_student, load_backbone
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings
from LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe import (
    DOCUMENT_INSTRUCTION, TEXT_TO_IMAGE_QUERY_INSTRUCTION,
    _text_inputs, _validate_text_token_budget, load_contract,
)

STAGES = ('vision_router', 'vision_expert', 'vision_global',
          'language_router', 'language_expert', 'language_global')
ROUTER = {'vision_router', 'language_router'}
FEATURE = {'vision_expert', 'vision_global', 'language_expert', 'language_global'}


def stage_dir(root: Path, stage: str) -> Path:
    return root / f'{STAGES.index(stage) + 1:02d}_{stage}'


def ccd(root: Path, stage: str, key: str) -> np.ndarray:
    path = stage_dir(root, stage) / 'ccd_captured' / f'{key}.png'
    if not path.is_file():
        raise FileNotFoundError(path)
    image = np.asarray(Image.open(path))
    if image.shape != (478, 478) or image.dtype != np.uint8:
        raise RuntimeError(f'Invalid canonical CCD: {path} {image.shape} {image.dtype}')
    return image


def router_scores(images: np.ndarray, settings) -> dict[str, torch.Tensor]:
    windows = settings.optical_router_detector_intervals
    energies = []
    captures = []
    for image in images:
        value = image.astype(np.float64)
        energy = [float(value[y0:y1, x0:x1].sum())
                  for y0, y1 in windows for x0, x1 in windows]
        energies.append(energy)
        captures.append(sum(energy) / max(float(value.sum()), 1e-12))
    energy = torch.tensor(energies, dtype=torch.float64)
    eps = settings.optical_router_energy_eps
    if settings.optical_router_score_normalization == 'standardized_region_energy':
        centered = energy - energy.mean(dim=-1, keepdim=True)
        logits = centered / centered.square().mean(dim=-1, keepdim=True).add(eps).sqrt()
    else:
        logits = (energy / energy.sum(dim=-1, keepdim=True).clamp_min(eps)).clamp_min(eps).log()
    probs = torch.softmax(logits / settings.router_temperature, dim=-1)
    order = torch.topk(probs, settings.top_k, dim=-1).indices
    selected = torch.zeros_like(probs, dtype=torch.bool).scatter_(1, order, True)
    weights = torch.where(selected, probs, torch.zeros_like(probs))
    if settings.router_weight_normalization != 'power_l2':
        raise RuntimeError('Unexpected router weight normalization')
    weights = weights / weights.square().sum(dim=-1, keepdim=True).clamp_min(eps).sqrt()
    fractions = energy / energy.sum(dim=-1, keepdim=True).clamp_min(eps)
    return {'probabilities': probs.float(), 'weights': weights.float(),
            'selected_mask': selected, 'selected_indices': order.long(),
            'detector_energy': energy.float(),
            'detector_energy_fraction': fractions.float(),
            'raw_capture_fraction': torch.tensor(captures, dtype=torch.float32)}


def install_measured(replacement, root: Path, keys: list[str], target: str, settings):
    index = len(STAGES) if target == 'final' else STAGES.index(target)
    v = replacement.vision_surrogate.core.optical_branch
    l = replacement.language_surrogate.core.optical_branch
    v.clear_measured_ccd()
    l.clear_measured_ccd()
    for branch in (v, l):
        branch.core.router.set_measured_routing(None)
    is_title = keys[0].startswith('title_')
    for previous in STAGES[:index]:
        if is_title and previous.startswith('vision'):
            continue
        if previous in ROUTER:
            values = np.stack([ccd(root, previous, key) for key in keys])
            branch = v if previous.startswith('vision') else l
            branch.core.router.set_measured_routing(router_scores(values, settings))
    tensors = {}
    for previous in FEATURE:
        if STAGES.index(previous) >= index or (is_title and previous.startswith('vision')):
            tensors[previous] = None
        else:
            tensors[previous] = torch.from_numpy(np.stack(
                [ccd(root, previous, key) for key in keys]
            ).astype(np.float32))
    v.set_measured_ccd(expert=tensors['vision_expert'], global_=tensors['vision_global'])
    l.set_measured_ccd(expert=tensors['language_expert'], global_=tensors['language_global'])


def inputs_for(loaded, settings, contract, keys: list[str]):
    if keys[0].startswith('image_'):
        selected = tuple(contract.test[int(key[6:])] for key in keys)
        dataset = GroceryRetrievalDataset(selected, settings.image_size, augment=False)
        images = [dataset[i]['image'] for i in range(len(dataset))]
        inputs = preprocess_images(loaded.processor, images, DOCUMENT_INSTRUCTION)
        validate_token_budgets(inputs, settings)
    else:
        selected = [contract.titles[int(key[6:])] for key in keys]
        inputs = _text_inputs(loaded.processor,
                              [item.text for item in selected],
                              TEXT_TO_IMAGE_QUERY_INSTRUCTION)
        _validate_text_token_budget(inputs, settings)
    return move_inputs(inputs, loaded.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=STAGES[1:], required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--physical-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--limit', type=int, default=0,
                   help='Diagnostic prefix length; 0 means the complete split')
    p.add_argument('--title-only', action='store_true',
                   help='Diagnostic title-only export for a language stage')
    args = p.parse_args()
    if sha(args.checkpoint) != EXPECTED:
        raise RuntimeError('Checkpoint SHA256 mismatch')
    settings = load_settings(args.config)
    contract = load_contract(args.data_root)
    root = args.physical_root
    stage = stage_dir(root, args.stage)
    compact = stage / 'compact_amplitude'
    compact.mkdir(parents=True, exist_ok=True)
    manifest = stage / 'manifest.jsonl'
    prior = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            prior[row['key']] = row
    keys = [f'image_{i:04d}' for i in range(2400)]
    if args.stage.startswith('language'):
        keys += [f'title_{i:03d}' for i in range(100)]
    elif args.title_only:
        raise ValueError('title-only requires a language stage')
    if args.title_only:
        keys = keys[2400:]
    if args.limit:
        keys = keys[:args.limit]
    loaded = load_backbone(settings, torch.device('cuda'))
    replacement, readout = build_student(loaded, settings)
    try:
        load_checkpoint(args.checkpoint, replacement, readout)
        replacement.use_student()
        replacement.set_phase_dropout_active(False)
        replacement.vision_surrogate.eval()
        replacement.language_surrogate.eval()
        readout.eval()
        branch = (replacement.vision_surrogate if args.stage.startswith('vision')
                  else replacement.language_surrogate).core.optical_branch
        for group in (keys[:2400], keys[2400:]):
            for start in range(0, len(group), args.batch_size):
                batch = [key for key in group[start:start + args.batch_size]
                         if key not in prior or not (compact / prior[key]['file']).is_file()]
                if not batch:
                    continue
                install_measured(replacement, root, batch, args.stage, settings)
                inputs = inputs_for(loaded, settings, contract, batch)
                branch.core.capture_intermediate_fields = True
                branch.core.capture_sample_count = len(batch)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                           enabled=settings.amp_enabled):
                    student_embeddings(loaded.model, replacement, readout, inputs)
                if args.stage.endswith('router'):
                    fields = branch.core.last_input_fields
                elif args.stage.endswith('expert'):
                    fields = branch.last_expert_input_amplitude
                else:
                    fields = branch.last_global_input_amplitude
                arrays = fields.detach().float().cpu().numpy()
                expected_side = 224 if args.stage.endswith('router') else 478
                if arrays.shape != (len(batch), expected_side, expected_side):
                    raise RuntimeError(f'Unexpected field shape: {arrays.shape}')
                with manifest.open('a', encoding='utf-8') as stream:
                    for key, field in zip(batch, arrays):
                        positive = field[field > 0]
                        scale = float(np.percentile(positive, 99.5)) if len(positive) else 0.0
                        if not np.isfinite(scale) or scale <= 0:
                            raise RuntimeError(f'Invalid field for {key}')
                        gray = np.rint(np.clip(field / scale, 0, 1) * 255).astype(np.uint8)
                        filename = key + '.png'
                        path = compact / filename
                        Image.fromarray(gray).save(path)
                        row = {'key': key, 'file': filename, 'sha256': sha(path),
                               'scale_p995': scale, 'checkpoint_sha256': EXPECTED,
                               'stage': args.stage}
                        stream.write(json.dumps(row) + '\n')
                        stream.flush()
                        prior[key] = row
                if len(prior) % 100 < len(batch):
                    print(f'exported {args.stage} {len(prior)}/{len(keys)}', flush=True)
    finally:
        replacement.close()
    print(f'{args.stage.upper()}_EXPORTED {len(prior)}/{len(keys)}', flush=True)


if __name__ == '__main__':
    main()
