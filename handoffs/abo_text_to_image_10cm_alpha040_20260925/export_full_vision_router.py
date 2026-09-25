"""Export the pinned 2,400-image TEST split for physical vision-router capture.

Run from the T08 training worktree. Resumable; never changes the test order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from export_router_pilot import sha, EXPECTED, INSTRUCTION
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import (
    move_inputs, preprocess_images, student_embeddings, validate_token_budgets,
)
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.prepare_grocery_retrieval_subset import GroceryRetrievalDataset
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import load_checkpoint
from LightGenV2.tasks.t01_object_retrieval.modeling import build_student, load_backbone
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings
from LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe import load_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if sha(args.checkpoint) != EXPECTED:
        raise RuntimeError('Checkpoint SHA256 mismatch')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    settings = load_settings(args.config)
    samples = tuple(load_contract(args.data_root).test)
    if len(samples) != 2400:
        raise RuntimeError(f'Expected 2400 TEST images; got {len(samples)}')
    args.output.mkdir(parents=True, exist_ok=True)
    compact = args.output / 'compact_amplitude'
    compact.mkdir(exist_ok=True)
    manifest = args.output / 'manifest.jsonl'
    prior = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            prior[int(row['index'])] = row
    loaded = load_backbone(settings, torch.device('cuda'))
    replacement, readout = build_student(loaded, settings)
    try:
        load_checkpoint(args.checkpoint, replacement, readout)
        replacement.use_student()
        replacement.set_phase_dropout_active(False)
        replacement.vision_surrogate.eval()
        replacement.language_surrogate.eval()
        readout.eval()
        branch = replacement.vision_surrogate.core.optical_branch
        for start in range(0, len(samples), args.batch_size):
            indices = list(range(start, min(start + args.batch_size, len(samples))))
            missing = [i for i in indices if i not in prior or not (compact / prior[i]['file']).is_file()]
            if not missing:
                continue
            batch_samples = tuple(samples[i] for i in missing)
            dataset = GroceryRetrievalDataset(batch_samples, settings.image_size, augment=False)
            images = [dataset[i]['image'] for i in range(len(dataset))]
            inputs = preprocess_images(loaded.processor, images, INSTRUCTION)
            validate_token_budgets(inputs, settings)
            inputs = move_inputs(inputs, loaded.device)
            branch.core.capture_intermediate_fields = True
            branch.core.capture_sample_count = len(images)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=settings.amp_enabled):
                student_embeddings(loaded.model, replacement, readout, inputs)
            fields = branch.core.last_input_fields.detach().float().cpu().numpy()
            if fields.shape != (len(missing), 224, 224):
                raise RuntimeError(f'Unexpected amplitude shape: {fields.shape}')
            with manifest.open('a', encoding='utf-8') as stream:
                for i, field in zip(missing, fields):
                    positive = field[field > 0]
                    scale = float(np.percentile(positive, 99.5)) if len(positive) else 0.0
                    if not np.isfinite(scale) or scale <= 0:
                        raise RuntimeError(f'Invalid field at index {i}')
                    gray = np.rint(np.clip(field / scale, 0, 1) * 255).astype(np.uint8)
                    name = f'image_{i:04d}.png'
                    path = compact / name
                    Image.fromarray(gray).save(path)
                    row = {'index': i, 'sample_id': samples[i].sample_id,
                           'product_id': samples[i].sku_name, 'file': name,
                           'sha256': sha(path), 'scale_p995': scale,
                           'checkpoint_sha256': EXPECTED}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    prior[i] = row
            print(f'exported {len(prior)}/{len(samples)}', flush=True)
    finally:
        replacement.close()


if __name__ == '__main__':
    main()
