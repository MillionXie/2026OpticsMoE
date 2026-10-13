"""Render fixed layered-scene examples from audited OURS and Qwen predictions.

No checkpoint execution or scene editing occurs here. Prediction grids must
come from actual fixed-weight inference; targets are never used to repair them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .layered_scene_data import (
    ANCHOR_BOTTOM, ANCHOR_X, DEPTH_GROUP, _artwork, _sprite, render_grid,
)

TASK = Path(__file__).resolve().parent
IDS = ('test_000588', 'test_000097', 'test_000766', 'test_000839')
TITLES = ('Completing the scene', 'Replacing one object',
          'Changing a spatial relation', 'Selective removal')
CHECKPOINTS = {
    'ours': '03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21',
    'qwen': '0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_records(evidence):
    if tuple(evidence['sample_ids']) != IDS:
        raise ValueError('Use the four previously selected story IDs in their original order')
    models = evidence['models']
    if models['ours']['test_manifest_sha256'] != models['qwen']['test_manifest_sha256']:
        raise ValueError('OURS and Qwen use different TEST manifests')
    for name in ('ours', 'qwen'):
        if models[name]['checkpoint_sha256'] != CHECKPOINTS[name]:
            raise ValueError(f'Wrong {name} checkpoint')
        if tuple(r['sample_id'] for r in models[name]['records']) != IDS:
            raise ValueError('Record order differs from the fixed selection')
    for ours, qwen in zip(models['ours']['records'], models['qwen']['records']):
        for key in ('sample_id', 'task', 'instruction', 'source_grid', 'target_grid'):
            if ours[key] != qwen[key]:
                raise ValueError(f'Input or target mismatch: {key}')
        for record in (ours, qwen):
            source = np.asarray(record['source_grid'])
            target = np.asarray(record['target_grid'])
            prediction = np.asarray(record['prediction_grid'])
            edit = np.asarray(record['predicted_edit_grid'], dtype=bool)
            if any(g.shape != (6, 6) for g in (source, target, prediction, edit)):
                raise ValueError('Expected 6x6 grids')
            if prediction.min() < 0 or prediction.max() > 16:
                raise ValueError('Unknown predicted category')
            changed = source != target
            cca = float((prediction == target)[changed].mean())
            union = np.logical_or(edit, changed).sum()
            iou = float(np.logical_and(edit, changed).sum() / union) if union else 1.
            for key, value in [('changed_cell_accuracy', cca), ('edit_grid_iou', iou),
                               ('scene_exact_match', float(np.array_equal(prediction, target)))]:
                if abs(value - record['metrics'][key]) > 1e-6:
                    raise ValueError(f'Grid/metric mismatch: {record["sample_id"]}/{key}')
            if record['archived_metric_max_error'] > 1e-6:
                raise ValueError('Fresh inference differs from the archived per-case metrics')


def scene_display(grid, svg_dir, scale=4):
    """Same fixed geometry for all methods; out-of-frame predictions stay clipped."""
    grid = np.asarray(grid)
    canvas = Image.new('RGBA', (224 * scale, 224 * scale), (250, 250, 247, 255))
    objects = sorted((DEPTH_GROUP.get(int(grid[r, c]), 4), int(r), int(c), int(grid[r, c]))
                     for r, c in zip(*np.nonzero(grid)))
    clipped = []
    for _, row, col, category in objects:
        base = _sprite(svg_dir, category)
        icon = _artwork(str(svg_dir.resolve()), category).resize(
            (base.width * scale, base.height * scale), Image.Resampling.LANCZOS)
        # Scale the original integer bounding box; do not recenter/rearrange objects.
        x = (ANCHOR_X[col] - base.width // 2) * scale
        y = (ANCHOR_BOTTOM[row] - base.height) * scale
        if x < 0 or y < 0 or x + icon.width > canvas.width or y + icon.height > canvas.height:
            clipped.append({'category': category, 'anchor_rc': [row, col]})
        layer = Image.new('RGBA', canvas.size)
        layer.paste(icon, (x, y))
        canvas = Image.alpha_composite(canvas, layer)
    return canvas.convert('RGB'), clipped


def make_figure(cases, panels, indices, destination):
    n = len(indices)
    fig = plt.figure(figsize=(8.2, 0.85 + n * 2.63 + .36), facecolor='white')
    height = fig.get_figheight()
    inch_y = lambda value: 1 - value / height
    headers = ('Input scene', 'Target scene', 'Ours', 'Qwen baseline')
    left, width, gap = .055, .211, .031
    fig.text(.055, inch_y(.22), 'Text-guided scene editing', fontsize=12, weight='bold', va='top')
    fig.text(.055, inch_y(.47), 'Fixed story examples | identical inputs and instructions',
             fontsize=8, color='#59616a', va='top')
    for j, header in enumerate(headers):
        fig.text(left + j * (width + gap) + width / 2, inch_y(.77), header,
                 ha='center', va='top', fontsize=9.3, weight='bold')
    for line, i in enumerate(indices):
        record = cases['ours'][i]
        top = .98 + line * 2.63
        fig.text(left, inch_y(top), f'{chr(97+i)}  {TITLES[i]}', fontsize=9.2,
                 weight='bold', va='top')
        fig.text(left, inch_y(top + .22), record['instruction'], fontsize=8.5,
                 color='#3e4751', va='top')
        image_top = top + .43
        for j, name in enumerate(('input', 'target', 'ours', 'qwen')):
            ax = fig.add_axes([left + j * (width + gap), inch_y(image_top + 1.73), width, 1.73 / height])
            ax.imshow(panels[i][name]); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#e5e7e8'); spine.set_linewidth(.6)
            text_y = image_top + 1.79
            if name in ('ours', 'qwen'):
                metrics = cases[name][i]['metrics']
                fig.text(left + j * (width + gap), inch_y(text_y),
                         f'Changed-cell acc. {metrics["changed_cell_accuracy"]:.0%}\n'
                         f'Edit IoU {metrics["edit_grid_iou"]:.2f}',
                         fontsize=7.5, color='#26343d', va='top', linespacing=1.5)
            else:
                fig.text(left + j * (width + gap), inch_y(text_y),
                         'RGB + instruction' if name == 'input' else 'Ground truth',
                         fontsize=7.5, color='#6d747b', va='top')
        fig.text(left, inch_y(top + 2.41), record['sample_id'], fontsize=7,
                 color='#7e858b', va='top')
    fig.text(left, .016, '1000-case development TEST: Ours 0.8765; Qwen baseline 0.8120 (changed-cell accuracy).',
             fontsize=7.3, color='#59616a', va='bottom')
    for suffix in ('.png', '.pdf', '.svg'):
        fig.savefig(destination.with_suffix(suffix), dpi=300, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path,
                        default=TASK / 'runs/smoke/story_comparison_20261013/inference_records.json')
    parser.add_argument('--output', type=Path, default=TASK / 'reports/story_comparison_20261013/figures')
    parser.add_argument('--svg-dir', type=Path, default=TASK / 'assets/openmoji-17.0.0-svg')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new empty output path; preserve the previous figure exports')
    evidence = json.loads(args.evidence.read_text(encoding='utf-8'))
    verify_records(evidence)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    cases = {name: evidence['models'][name]['records'] for name in ('ours', 'qwen')}
    panels, metadata, metric_rows = [], [], []
    args.output.mkdir(parents=True)
    for i, record in enumerate(cases['ours']):
        row, meta = {}, {'sample_id': record['sample_id'], 'panels': {}}
        for name, grid in [('input', record['source_grid']), ('target', record['target_grid']),
                           ('ours', record['prediction_grid']),
                           ('qwen', cases['qwen'][i]['prediction_grid'])]:
            if name in ('input', 'target'):
                original = args.evidence.parent / 'samples' / record['sample_id'] / ('source.png' if name == 'input' else 'target.png')
                raster, _ = render_grid(np.asarray(grid), args.svg_dir)
                if not np.array_equal(np.asarray(raster), np.asarray(Image.open(original).convert('RGB'))):
                    raise ValueError(f'Original scene pixels differ from the renderer: {record["sample_id"]}/{name}')
            row[name], clipped = scene_display(grid, args.svg_dir)
            filename = f'{chr(97+i)}_{record["task"]}_{name}.png'
            row[name].save(args.output / filename)
            meta['panels'][name] = {'file': filename, 'clipped_objects': clipped,
                                    'grid': grid}
        panels.append(row); metadata.append(meta)
        for name in ('ours', 'qwen'):
            metric_rows.append({'sample_id': record['sample_id'], 'task': record['task'],
                               'instruction': record['instruction'], 'method': name,
                               **{k: v for k, v in cases[name][i]['metrics'].items()
                                  if isinstance(v, (int, float))}})
        make_figure(cases, panels, [i], args.output / f'{chr(97+i)}_{record["task"]}_comparison')
    make_figure(cases, panels, list(range(4)), args.output / 'story_ours_qwen_comparison')
    with (args.output / 'case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    manifest = {'evidence': str(args.evidence.resolve()), 'evidence_sha256': digest(args.evidence),
                'renderer_script_sha256': digest(__file__),
                'svg_asset_sha256': {p.name: digest(p) for p in sorted(args.svg_dir.glob('*.svg'))},
                'checkpoint_sha256': CHECKPOINTS, 'selection': 'the four existing story IDs; no reselection by model performance',
                'renderer': 'same proportional size, integer anchor and depth order for all four columns; SVG-resampled display at 4x',
                'original_224px_input_and_target_match': True, 'samples': metadata,
                'files': [{'file': p.name, 'sha256': digest(p)} for p in sorted(args.output.iterdir()) if p.is_file()]}
    (args.output / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'fixed_cases': 4, 'metric_rows': 8}, indent=2))


if __name__ == '__main__':
    main()
