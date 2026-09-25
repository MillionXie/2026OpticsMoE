"""Cache frozen pre-readout features from six measured optical stages.

The saved tensors are the inputs to the final LN+Linear electronic readout,
not simulated CCD features or post-readout embeddings. Optical model/Qwen are
loaded only to reconstruct this one frozen feature representation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from export_router_pilot import EXPECTED, sha
from export_full_next_stage import inputs_for, install_measured, selected_train_keys
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import student_embeddings
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import load_checkpoint
from LightGenV2.tasks.t01_object_retrieval.modeling import build_student, load_backbone
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings
from LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe import load_contract


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--physical-root', type=Path, required=True)
    p.add_argument('--split', choices=('train', 'test'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--train-per-sku', type=int, default=8)
    p.add_argument('--selection-seed', type=int, default=20260926)
    args = p.parse_args()
    if sha(args.checkpoint) != EXPECTED:
        raise RuntimeError('Checkpoint SHA256 mismatch')
    settings = load_settings(args.config)
    contract = load_contract(args.data_root)
    image_keys = ([f'image_{i:04d}' for i in range(2400)] if args.split == 'test'
                  else selected_train_keys(contract, args.train_per_sku, args.selection_seed))
    title_keys = [f'title_{i:03d}' for i in range(100)]
    for stage_number, stage in enumerate(('vision_router', 'vision_expert', 'vision_global',
                                          'language_router', 'language_expert', 'language_global'), 1):
        expected = image_keys if stage_number <= 3 else image_keys + title_keys
        directory = args.physical_root / f'{stage_number:02d}_{stage}' / 'ccd_captured'
        missing = [key for key in expected if not (directory / f'{key}.png').is_file()]
        if missing:
            raise FileNotFoundError(f'{stage}: {len(missing)} CCD files missing, first={missing[0]}')
    loaded = load_backbone(settings, torch.device('cuda'))
    replacement, readout = build_student(loaded, settings)
    try:
        load_checkpoint(args.checkpoint, replacement, readout)
        replacement.use_student()
        replacement.set_phase_dropout_active(False)
        replacement.vision_surrogate.eval()
        replacement.language_surrogate.eval()
        readout.eval()
        outputs = []
        for group in (image_keys, title_keys):
            chunks = []
            for start in range(0, len(group), args.batch_size):
                batch = group[start:start + args.batch_size]
                install_measured(replacement, args.physical_root, batch, 'final', settings)
                inputs = inputs_for(loaded, settings, contract, batch)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                           enabled=settings.amp_enabled):
                    _, features = student_embeddings(loaded.model, replacement, readout, inputs)
                chunks.append(features.detach().float().cpu())
                if (start + len(batch)) % 100 == 0:
                    print(f'cached {group[0].split("_")[0]} '
                          f'{start + len(batch)}/{len(group)}', flush=True)
            outputs.append(torch.cat(chunks))
        image_features, title_features = outputs
        image_samples = (contract.train if args.split == 'train' else contract.test)
        image_labels = [image_samples[int(key[6:])].sku_index for key in image_keys]
        payload = {
            'schema': 1, 'split': args.split, 'checkpoint_sha256': EXPECTED,
            'image_keys': image_keys, 'image_labels': image_labels,
            'title_keys': title_keys, 'image_features': image_features,
            'title_features': title_features,
            'physical_root': str(args.physical_root),
            'selection_seed': args.selection_seed if args.split == 'train' else None,
            'train_per_sku': args.train_per_sku if args.split == 'train' else None,
            'feature_shape': list(image_features.shape),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.output)
        (args.output.with_suffix('.json')).write_text(json.dumps({
            key: value for key, value in payload.items()
            if key not in ('image_features', 'title_features', 'image_keys', 'image_labels', 'title_keys')
        }, indent=2), encoding='utf-8')
        print(f'CACHE_COMPLETE {args.split} {len(image_keys)} images 100 titles '
              f'feature_dim={image_features.shape[-1]}', flush=True)
    finally:
        replacement.close()


if __name__ == '__main__':
    main()
