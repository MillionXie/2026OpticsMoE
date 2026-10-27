"""CPU-only, exposure-audited held-out evaluation assets for native Qwen/LG.

No feature cache, training row or model score is produced. Final evaluation must
bind a frozen checkpoint; existing development rows remain selection-only.
"""
import argparse
import collections
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from .qwen_vl_curated import reviewed_pairs
from .qwen_vl_turbo_data import _pixels_hash, normalized_product, render_pair, sha256, write_json


def exposure_inventory(run, root):
    """Follow both spatial parents and early --resume checkpoints, fail closed."""
    todo, seen, manifests, rows = [run.resolve()], set(), {}, []
    executions = {}
    while todo:
        current = todo.pop().resolve()
        if current in seen:
            continue
        seen.add(current)
        execution = current / 'execution.json'
        metadata = json.loads(execution.read_text())
        executions[str(execution)] = sha256(execution)
        argv = metadata['argv']
        for flag in ('--parent', '--resume'):
            if flag in argv:
                parent = Path(argv[argv.index(flag)+1])
                parent = parent if parent.is_absolute() else root / parent
                todo.append(parent.parent if parent.suffix == '.pt' else parent)
        data_run = metadata.get('data_run')
        candidates = [current / 'annotated_pairs.json']
        if data_run:
            candidates.insert(0, Path(data_run) / 'annotated_pairs.json')
        available = [p for p in candidates if p.exists()]
        if not available:
            raise ValueError('Missing ancestor exposure rows: '+str(current))
        for path in available:
            if str(path) in manifests:
                continue
            manifests[str(path)] = sha256(path)
            rows.extend(json.loads(path.read_text()))
    return rows, {'ancestor_runs': sorted(map(str, seen)),
                  'execution_sha256': executions, 'pair_manifest_sha256': manifests,
                  'exclusion_scope': 'All TRAIN and development rows, both source and target roles, in the full recorded warm-start lineage; not a claim about public pretraining data.'}


def reject_exposure(products, prior_rows):
    exposed = {key: set() for key in ('product_id', 'family_id', 'rgba_sha256')}
    for row in prior_rows:
        for role in ('source', 'target'):
            for key in exposed:
                if row[role].get(key):
                    exposed[key].add(row[role][key])
    for product in products:
        for key, values in exposed.items():
            if not product.get(key):
                raise ValueError('Missing identity: '+key)
            if product[key] in values:
                raise ValueError('Held-out exposure overlap '+key+': '+product[key])
    return {key: len(values) for key, values in exposed.items()}


def raw_products(data):
    path = data / 'test.jsonl'
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['category'] not in ('lamp', 'pillow'):
            continue
        if row.get('license') != 'CC BY 4.0':
            raise ValueError('Unreviewed asset license')
        row['product_id'] = row['sequence_id']
        row['family_id'] = row.get('parent_id') or row.get('family_id') or row['product_id']
        for key in ('image_path', 'mask_path'):
            row[key] = str((data / row[key]).resolve())
        row['rgba_sha256'] = _pixels_hash(row)
        rows.append(row)
    return rows


def screen(data, baseline, output, root):
    if output.exists():
        raise FileExistsError('Inspect existing screen; never overwrite')
    prior, audit = exposure_inventory(baseline, root)
    rows = raw_products(data)
    # Do not silently remove an overlap and call the complete pool clean.
    reject_exposure(rows, prior)
    output.mkdir(parents=True)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['category'], row['product_id']].append(row)
    for category in ('lamp', 'pillow'):
        chosen = sorted((key, sorted(group, key=lambda r: r['sample_id']))
                        for key, group in groups.items() if key[0] == category)
        width = max(len(group) for _, group in chosen)
        sheet = Image.new('RGB', (width*160, len(chosen)*194), 'white')
        draw = ImageDraw.Draw(sheet)
        for y, (key, group) in enumerate(chosen):
            for x, row in enumerate(group):
                fg, alpha = normalized_product(row, 256)
                im = Image.composite(fg, Image.new('RGB', fg.size, 'white'), alpha)
                im.thumbnail((156, 156))
                sheet.paste(im, (x*160, y*194+32))
                draw.text((x*160+2, y*194+3), row['sample_id'], fill='black')
        sheet.save(output / (category+'_candidate_views.png'))
    write_json(output / 'candidate_products.json', rows)
    write_json(output / 'screen_audit.json', {**audit, 'raw_test_manifest': str(data/'test.jsonl'),
               'raw_test_manifest_sha256': sha256(data/'test.jsonl'),
               'candidate_views': len(rows), 'candidate_products': len(groups),
               'overlap_with_recorded_lineage': False, 'model_predictions_used': False,
               'visual_review_complete': False, 'gpu_used': False})


def check_pairs(pairs):
    if len(pairs) != 240 or any(row['split'] != 'test' for row in pairs):
        raise ValueError('Require exactly 240 held-out TEST rows')
    cells = collections.Counter((row['category'], row['mode']) for row in pairs)
    expected = {(cat, mode): 40 for cat in ('lamp', 'pillow')
                for mode in ('background', 'object', 'joint')}
    if dict(cells) != expected or len({r['sample_id'] for r in pairs}) != 240:
        raise ValueError('Unbalanced or duplicate TEST rows')
    return {cat+'/'+mode: count for (cat, mode), count in cells.items()}


def prepare(screen_dir, selection, output, baseline, root):
    if output.exists():
        raise FileExistsError('Immutable held-out assets already exist')
    profile = json.loads(selection.read_text())
    audit = json.loads((screen_dir/'screen_audit.json').read_text())
    if profile['screen_audit_sha256'] != sha256(screen_dir/'screen_audit.json'):
        raise ValueError('Visual review is not bound to this exposure audit')
    if sha256(Path(audit['raw_test_manifest'])) != audit['raw_test_manifest_sha256']:
        raise ValueError('Raw TEST changed after screening')
    raw = json.loads((screen_dir/'candidate_products.json').read_text())
    lookup = {row['sample_id']: row for row in raw}
    products, descriptions = [], {}
    if len(profile['products']) != 8 or collections.Counter(p['category'] for p in profile['products']) != {'lamp': 4, 'pillow': 4}:
        raise ValueError('Require four reviewed products per category')
    for spec in profile['products']:
        if len(set(spec['views'])) != 2 or not spec['description'].strip():
            raise ValueError('Two complete distinct views and reviewed description required')
        for view in spec['views']:
            row = lookup[f"{spec['category']}-{spec['product_id']}-{view:02d}"]
            if _pixels_hash(row) != row['rgba_sha256']:
                raise ValueError('Selected raw pixels changed')
            products.append(row)
            descriptions[row['sample_id']] = spec['description']+', entirely visible'
    if len({r['product_id'] for r in products}) != 8 or len({r['rgba_sha256'] for r in products}) != 16:
        raise ValueError('Duplicate selected product or view pixels')
    prior, lineage = exposure_inventory(baseline, root)
    exposures = reject_exposure(products, prior)
    # Reuse the exact data renderer/prompt policy. Its legacy validator admits
    # only val; immediately relabel the new rows and export NO training cache.
    rows = reviewed_pairs({'val': products}, descriptions, {'seed': 1042, 'edits_per_mode': {'val': 5}})
    for index, row in enumerate(rows):
        row['split'] = 'test'
        row['sample_id'] = 'holdout_'+str(index).zfill(5)
    cells = check_pairs(rows)
    output.mkdir(parents=True)
    inputs, pixel_hashes = [], {}
    for row in rows:
        reference, target = render_pair(row)
        for role, image in (('reference', reference), ('target', target)):
            path = output / role / (row['sample_id']+'.png')
            path.parent.mkdir(exist_ok=True)
            image.save(path)
            pixel_hashes[str(path.relative_to(output))] = sha256(path)
        inputs.append({'sample_id': row['sample_id'], 'category': row['category'], 'mode': row['mode'],
                       'image': str((output/'reference'/(row['sample_id']+'.png')).resolve()),
                       'prompt': row['prompt']})
    # This is the only inference manifest: no GT, masks, teacher captions or SKU.
    write_json(output/'inference_inputs.json', inputs)
    write_json(output/'evaluation_pairs.json', rows)
    write_json(output/'asset_sha256.json', pixel_hashes)
    write_json(output/'protocol.json', {**lineage, 'complete': True, 'selection_sha256': sha256(selection),
               'screen_audit_sha256': sha256(screen_dir/'screen_audit.json'),
               'pair_manifest_sha256': sha256(output/'evaluation_pairs.json'),
               'inference_manifest_sha256': sha256(output/'inference_inputs.json'),
               'raw_test_manifest_sha256': audit['raw_test_manifest_sha256'],
               'pair_count': 240, 'products': 8, 'raw_views': 16, 'image_size': 256, 'cells': cells,
               'ancestor_exposure_unique': exposures, 'recorded_lineage_overlap': False,
               'gpu_used': False, 'gradients': False, 'model_predictions_used_for_selection': False,
               'existing_train_pairs': 960, 'existing_development_pairs': 96,
               'evaluation_status': 'Prepared, unevaluated; only evaluate after checkpoint/selection protocol freeze. Any later feedback tuning revokes held-out claim.',
               'family_limit': 'Upstream family if present, otherwise SKU; visual review is not exhaustive near-duplicate certification.',
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
               'source_sha256': sha256(Path(__file__))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['screen', 'prepare'], required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--screen', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.stage == 'screen':
        if not a.data:
            p.error('--data required')
        screen(a.data, a.baseline, a.output, Path.cwd())
    else:
        if not a.screen or not a.selection:
            p.error('--screen and --selection required')
        prepare(a.screen, a.selection, a.output, a.baseline, Path.cwd())


if __name__ == '__main__':
    main()
