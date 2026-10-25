"""CPU-only category and full saliency plotting data from sealed predictions.

No encoder/head forward, optimization, checkpoint selection or source-run writes.
The original scalar CC is retained; other metrics replay native evaluation rules.
"""
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

from .prepare_final_plot_data import TASK, ROOT, read, sha, immutable_json

CATEGORY_ORDER = [
    ('vase', 'Vases'), ('home mirror', 'Mirrors'), ('bed', 'Beds'),
    ('pillow', 'Pillows'), ('stool seating', 'Stools'),
    ('light fixture', 'Light fixtures'), ('wall art', 'Wall art'),
    ('sofa', 'Sofas'), ('rug', 'Rugs'), ('chair', 'Chairs'),
]


def category_data(spec):
    run = TASK / spec['run']
    pred_path = run / 'fixed_weight_predictions.json'
    pred = read(pred_path)
    result = read(run / 'result.json')
    weight = run / 'last_checkpoint.pt'
    assert sha(weight) == spec['checkpoint_SHA']
    protocol_path = ROOT / 'LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/abo200_enrolled_protocol_20260913/protocol.json'
    assert sha(protocol_path) == 'f1749d5fc22d2dfee6a1333ce2b35e9fa600a070f949eba8420b4def41906dde'
    protocol = read(protocol_path)
    parent_path = ROOT / 'data/abo_similarity10_data/data/abo_similarity10_manifest.csv'
    assert sha(parent_path) == protocol['parent_manifest_sha256']
    with parent_path.open(encoding='utf-8-sig', newline='') as f:
        parent_rows = list(csv.DictReader(f))
    parent = {r['sample_id']: r for r in parent_rows}
    assert len(parent) == len(parent_rows)
    query = {r['sample_id']: r for r in protocol['rows'] if r['split'] == 'query'}
    gallery = {r['sample_id']: r for r in protocol['rows'] if r['split'] == 'train'}
    assert len(query) == len(set(pred['query_ids'])) == pred['queries'] == 800
    assert set(query) == set(pred['query_ids'])
    assert set(gallery) == set(pred['candidate_ids']) and len(gallery) == 1600
    grouped = {key: [] for key, _ in CATEGORY_ORDER}
    details = []
    for i, sid in enumerate(pred['query_ids']):
        row, original = query[sid], parent[sid]
        sku = pred['query_skus'][i]
        assert row['product_id'] == original['product_id'] == sku
        assert int(row['category_id']) == int(original['category_id'])
        category = original['class_name']
        assert category in grouped
        top = pred['order'][i][0]
        candidate = pred['candidate_ids'][top]
        assert gallery[candidate]['product_id'] == pred['candidate_skus'][top]
        hit = int(sku == pred['candidate_skus'][top])
        assert hit == int(pred['first_positive_rank'][i] == 1)
        grouped[category].append((sid, sku, hit))
        details.append([sid, sku, category, hit])
    rows = []
    for key, label in CATEGORY_ORDER:
        group = grouped[key]
        assert len(group) == 80 and len({r[1] for r in group}) == 20
        correct = sum(r[2] for r in group)
        rows.append([label, len(group), correct, correct / len(group)])
    correct = sum(r[2] for r in rows)
    assert correct == 623 and math.isclose(correct / 800, result['metrics']['hit_at_1'], abs_tol=1e-7)
    assert sha(weight) == spec['checkpoint_SHA']
    return dict(headers=['Category', 'Queries', 'Correct Top-1', 'H@1'], rows=rows,
                overall=[800, correct, correct / 800], query_category_rows=details,
                definition='Exact-SKU H@1 grouped by QUERY category, not category classification accuracy.',
                checkpoint_SHA=spec['checkpoint_SHA'], sources={str(p.relative_to(ROOT)): sha(p) for p in (pred_path, protocol_path, parent_path)})


def saliency_data(spec):
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from .task_problems import Problem
    from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency.objectives import (
        density_from_logits, normalize_density, correlation_coefficient,
        kl_divergence, normalized_scanpath_saliency, similarity,
    )
    torch.set_num_threads(2)
    run = TASK / spec['run']
    source_paths = [run / name for name in ('fixed_weight_maps.pt', 'fixed_weight_predictions.json', 'result.json', 'run_manifest.json')]
    maps_path, pred_path, result_path, manifest_path = source_paths
    audit_path = TASK / 'runs/control/salicon_optimization_cacheclone1_maps_audit_20261024.json'
    assert read(audit_path)['status'] == 'passed'
    assert sha(maps_path) == 'bfaca0c535aa18fbcc0ecdf3b0285b2081a3e3c93f7ce9a339d479fb1308d519'
    weight = run / 'last_checkpoint.pt'
    assert sha(weight) == spec['checkpoint_SHA']
    maps = torch.load(maps_path, map_location='cpu', weights_only=False)
    assert maps['logits'].shape == (5000, 1, 7, 7)
    original = read(pred_path)
    result = read(result_path)
    problem = Problem('salicon')
    assert problem.identity == read(manifest_path)['data_identity']
    base = problem.test.base
    assert not base.training and int(problem.settings.image_size) == 224

    def target(index):
        record = base.records[index]
        with Image.open(record.density_path) as f:
            density = np.asarray(f.convert('L').resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
        mass = float(density.sum())
        assert mass > 0
        density /= mass
        with Image.open(record.fixation_path) as f:
            fixation = (np.asarray(f.convert('L').resize((224, 224), Image.Resampling.NEAREST), dtype=np.uint8) > 0).astype(np.float32)
        if not fixation.sum():
            y, x = np.unravel_index(density.argmax(), density.shape)
            fixation[y, x] = 1.
        return torch.from_numpy(density)[None], torch.from_numpy(fixation)[None]

    # GT-only decoding avoids unused RGB images; exact equality against the
    # native data adapter establishes the same evaluation target convention.
    checked_indices = [0, 1, 2499, 4998, 4999]
    for i in checked_indices:
        native = base[i]
        density, fixation = target(i)
        assert torch.equal(density, native['density']) and torch.equal(fixation, native['fixation'])
    rows, max_cc_error = [], 0.
    for start in range(0, 5000, 16):
        stop = min(start + 16, 5000)
        ids = [r.sample_id for r in base.records[start:stop]]
        assert ids == maps['sample_ids'][start:stop] == [r['sample_id'] for r in original[start:stop]]
        targets = [target(i) for i in range(start, stop)]
        density = density_from_logits(F.interpolate(maps['logits'][start:stop], (224, 224), mode='bilinear', align_corners=False))
        for j, sid in enumerate(ids):
            pred = density[j:j+1]
            gt = normalize_density(targets[j][0][None])
            fixation = targets[j][1][None]
            mae = float((pred / pred.max().clamp_min(1e-8) - gt / gt.max().clamp_min(1e-8)).abs().mean())
            kld = float(kl_divergence(pred, gt))
            cc = float(correlation_coefficient(pred, gt))
            nss = float(normalized_scanpath_saliency(pred, fixation))
            sim = float(similarity(pred, gt))
            old_cc = original[start+j]['cc']
            max_cc_error = max(max_cc_error, abs(old_cc - cc))
            rows.append([sid, mae, kld, old_cc, nss, sim])
        if stop % 800 == 0:
            print(json.dumps(dict(exported_saliency_samples=stop)), flush=True)
    headers = ['sample_id', 'MAE', 'KLD', 'CC', 'NSS', 'SIM']
    means = {key: sum(r[i] for r in rows) / len(rows) for i, key in enumerate(headers) if i}
    assert len(rows) == len({r[0] for r in rows}) == 5000
    assert all(math.isfinite(v) for r in rows for v in r[1:]) and max_cc_error < 5e-6
    for key, value in means.items():
        assert abs(value-result['metrics'][key.lower()]) < (2e-5 if key == 'NSS' else 5e-6), (key, value)
    assert sha(weight) == spec['checkpoint_SHA']
    assert sha(maps_path) == 'bfaca0c535aa18fbcc0ecdf3b0285b2081a3e3c93f7ce9a339d479fb1308d519'
    return dict(headers=headers, rows=rows, means=means, reported_metrics=result['metrics'],
                minima={key: min(r[i] for r in rows) for i, key in enumerate(headers) if i},
                maxima={key: max(r[i] for r in rows) for i, key in enumerate(headers) if i},
                max_recomputed_CC_error=max_cc_error, exact_native_GT_checks=checked_indices,
                checkpoint_SHA=spec['checkpoint_SHA'],
                definitions={'MAE': 'Peak-normalized density pixel MAE; lower is better.',
                             'KLD': 'KL(target || prediction), natural logarithm; lower is better.',
                             'CC': 'Original saved per-image Pearson CC; higher is better. Recomputed CC checked within5e-6.',
                             'NSS': 'Population-standardized prediction at binary fixation locations; higher is better.',
                             'SIM': 'Sum of minimum of mass-normalized prediction and target; higher is better.'},
                sources={str(p.relative_to(ROOT)): sha(p) for p in source_paths+[audit_path]},
                notes=['All5000 public validation TEST images; full tails retained, no clipping or sampling.',
                       'Native7x7 saved logits -> bilinear224 -> spatial softmax; zero new neural forwards or optimization.',
                       'Individual scalar means may differ from original batch-reduced FP32 means by rounding.'])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only export requires CUDA_VISIBLE_DEVICES empty')
    from .train_easy100 import require_published_source
    require_published_source()
    out = TASK / 'runs/outputs/t19_plot_supplement_20261025/data.json'
    if out.exists():
        print(json.dumps(dict(existing_payload=str(out), sha256=sha(out))))
        return
    config = read(TASK / 'configs/final_selected9_20261024.json')
    specs = {r['task']: r for r in config['rows']}
    payload = dict(status='passed', abo_categories=category_data(specs['abo_image_to_image']),
                   saliency=saliency_data(specs['salicon']), new_neural_forwards=0, new_optimizer_steps=0)
    immutable_json(out, payload)
    print(json.dumps(dict(status='passed', payload=str(out), sha256=sha(out),
                          categories=payload['abo_categories']['rows'], saliency_means=payload['saliency']['means'])))


if __name__ == '__main__':
    main()
