"""Evaluate all 100 title queries against 2,400 physically captured images."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from export_router_pilot import sha, EXPECTED
from export_full_next_stage import inputs_for, install_measured
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.features import student_embeddings
from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.train_optical_retrieval import load_checkpoint
from LightGenV2.tasks.t01_object_retrieval.modeling import build_student, load_backbone
from LightGenV2.tasks.t01_object_retrieval.settings import load_settings
from LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe import (
    _text_to_image_metrics, load_contract,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--physical-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--expected-sha', default=EXPECTED)
    p.add_argument('--report-prefix', default='physical_test')
    a = p.parse_args()
    if sha(a.checkpoint) != a.expected_sha:
        raise RuntimeError('Checkpoint SHA256 mismatch')
    if not a.report_prefix.replace('_', '').isalnum():
        raise ValueError('report-prefix must contain only letters, digits, and underscores')
    settings = load_settings(a.config)
    contract = load_contract(a.data_root)
    loaded = load_backbone(settings, torch.device('cuda'))
    replacement, readout = build_student(loaded, settings)
    try:
        load_checkpoint(a.checkpoint, replacement, readout)
        replacement.use_student()
        replacement.set_phase_dropout_active(False)
        replacement.vision_surrogate.eval()
        replacement.language_surrogate.eval()
        readout.eval()
        collections = []
        for keys in ([f'image_{i:04d}' for i in range(2400)],
                     [f'title_{i:03d}' for i in range(100)]):
            vectors = []
            for start in range(0, len(keys), a.batch_size):
                batch = keys[start:start + a.batch_size]
                install_measured(replacement, a.physical_root, batch, 'final', settings)
                inputs = inputs_for(loaded, settings, contract, batch)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16,
                                                           enabled=settings.amp_enabled):
                    embedding, _ = student_embeddings(loaded.model, replacement,
                                                      readout, inputs)
                vectors.append(embedding.detach().float().cpu())
                if (start + len(batch)) % 100 == 0:
                    print(f'evaluated {keys[0].split("_")[0]} '
                          f'{start+len(batch)}/{len(keys)}', flush=True)
            collections.append(torch.cat(vectors))
        images, titles = collections
        metrics, rows = _text_to_image_metrics(
            titles, images, [sample.sku_index for sample in contract.test]
        )
        a.physical_root.mkdir(parents=True, exist_ok=True)
        report = {'schema': 1, 'protocol': '100 physical title queries to 2400 physical TEST images',
                  'checkpoint_sha256': a.expected_sha, 'metrics': metrics,
                  'image_count': 2400, 'title_count': 100,
                  'all_six_optical_stages_measured': True}
        (a.physical_root / f'{a.report_prefix}_report.json').write_text(
            json.dumps(report, indent=2), encoding='utf-8'
        )
        with (a.physical_root / f'{a.report_prefix}_predictions.csv').open(
            'w', newline='', encoding='utf-8'
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        replacement.close()


if __name__ == '__main__':
    main()
