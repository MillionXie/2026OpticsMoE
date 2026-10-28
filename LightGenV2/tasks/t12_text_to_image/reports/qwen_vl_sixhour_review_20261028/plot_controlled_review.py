"""Matched controlled-room DEV figures with explicitly fallible offline audits.

Only completed raw DEV96 reviews and their SHA-bound CPU instruction audits
are accepted. Native PNG pixels are displayed without resizing or enhancement.
CLIP family predictions, including predictions on GT, are diagnostics; they
are never relabelled as ground truth or used to claim semantic superiority.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


MODES = ('background', 'object', 'joint')
CATEGORIES = ('lamp', 'pillow')
WARNING = ('CLIP-B16 family predictions and nearest wall-palette labels are offline diagnostics, '
           'not verified semantic success. The classifier can misclassify GT; a higher classifier '
           'score than GT does not establish better generation. Inspect native images and prompts.')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def labelled_paths(values, role):
    result = {}
    for value in values:
        if '=' not in value:
            raise ValueError(role+' requires LABEL=directory')
        label, directory = value.split('=', 1)
        if not label.strip() or label in result or not directory:
            raise ValueError(role+' labels must be nonempty and unique')
        result[label] = Path(directory)
    return result


def finite(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError('Missing/nonfinite '+name)
    return float(value)


def source_product(sample_id, category):
    parts = sample_id.split(':')
    if (len(parts) != 6 or parts[:2] != ['val', 'room'] or
            not parts[2].startswith(category+'-') or
            not parts[2].rsplit('-', 1)[-1].isdigit()):
        raise ValueError('Require controlled-room DEV source-view identities, never TRAIN/TEST')
    return parts[2].rsplit('-', 1)[0]


def native_png(path, expected_sha):
    if not path.is_file():
        raise FileNotFoundError('Download the complete native PNG export before plotting: '+str(path))
    actual = digest(path)
    if actual != expected_sha:
        raise ValueError('Audit-bound PNG bytes changed: '+str(path))
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != (256, 256):
            raise ValueError('Unmodified native RGB256 PNG required: '+str(path))
        pixels = hashlib.sha256(image.tobytes()).hexdigest()
    return {'path': str(path.resolve()), 'png_sha256': actual, 'RGB_sha256': pixels}


def load_model(label, directory, audit_directory):
    report_path = directory/'report.json'; report = read_json(report_path)
    audit_path = audit_directory/'report.json'; audit = read_json(audit_path)
    records = report.get('metrics', {}).get('images', [])
    if (not report.get('complete') or report.get('cohort') != 'development' or
            report.get('evaluation_label') != 'development_test' or
            report.get('independent_test_opened') is not False or
            len(records) != 96 or len({r['sample_id'] for r in records}) != 96):
        raise ValueError('Only completed controlled-room raw DEV96 reviews are accepted')
    if (not audit.get('complete') or audit.get('evaluation_label') != 'development_test' or
            audit.get('evaluation_used_for_selection') is not True or
            audit.get('independent_TEST_opened') is not False or audit.get('editor_run') is not False or
            audit.get('device') != 'cpu' or audit.get('review_report_sha256') != digest(report_path) or
            not report.get('parent_checkpoint_sha256') or
            audit.get('review_parent_checkpoint_sha256') != report['parent_checkpoint_sha256']):
        raise ValueError('Instruction audit is not bound to this completed raw review/checkpoint')
    audited = audit.get('images', [])
    if len(audited) != 96 or len({r['sample_id'] for r in audited}) != 96:
        raise ValueError('Each instruction audit must contain exactly96 distinct DEV identities')
    lookup = {r['sample_id']: r for r in audited}
    if set(lookup) != {r['sample_id'] for r in records}:
        raise ValueError('Instruction audit sample identities differ from the raw review')
    counts = Counter((r['category'], r['mode']) for r in records)
    if counts != Counter({(c, m): 16 for c in CATEGORIES for m in MODES}):
        raise ValueError('The balanced controlled-room DEV cohort has changed')
    combined = {}
    for index, metric in enumerate(records):
        sid = metric['sample_id']; item = lookup[sid]
        product = source_product(sid, metric['category'])
        if any(item[key] != metric[key] for key in ('sample_id', 'prompt', 'category', 'mode', 'object_crop_box')):
            raise ValueError('Per-image raw metrics and audit identities/regions disagree')
        for key in ('psnr', 'lpips', 'object_crop_lpips'):
            finite(metric.get(key), label+'/'+sid+'/'+key)
        requested = item.get('requested_family')
        pngs = {}
        for role, expected in (('reference', item['reference_png_sha256']),
                ('target', item['target_control']['png_sha256']), ('generated', item['generated']['png_sha256'])):
            pngs[role] = native_png(directory/'images'/'raw_dev'/role/f'sample_{index:04d}.png', expected)
        for role in ('generated', 'target_control'):
            data = item[role]
            finite(data['retained'].get('L1_RGB_01'), label+'/'+sid+'/retained_L1')
            if metric['mode'] == 'background':
                if requested is not None or data.get('family') is not None:
                    raise ValueError('Background-only rows cannot be labelled with replacement-family success')
            elif data.get('family', {}).get('requested_family') != requested or requested is None:
                raise ValueError('Requested family differs between instruction and audit')
            if not data.get('wall', {}).get('expected_palette'):
                raise ValueError('Expected wall-palette label is missing')
        combined[sid] = {'index': index, 'source_product': product, 'metrics': metric, 'audit': item, 'PNGs': pngs}
    return {'label': label, 'directory': directory, 'report': report, 'audit_directory': audit_directory,
        'audit_report': audit, 'records': combined,
        'provenance': {'review_directory': str(directory.resolve()), 'review_report_sha256': digest(report_path),
            'parent_checkpoint_sha256': report['parent_checkpoint_sha256'],
            'audit_directory': str(audit_directory.resolve()), 'audit_report_sha256': digest(audit_path),
            'data_pair_sha256': audit['data_pair_sha256'], 'prepared_PNG_manifest_sha256': audit['prepared_PNG_manifest_sha256'],
            'clip_repo_sha': audit['clip_repo_sha'], 'CLIP_weights_sha256': audit['weights_sha256'],
            'review_summary': {k: v for k, v in report['metrics'].items() if k != 'images'},
            'offline_audit_summary': audit['overall'], 'offline_audit_by_mode': audit['by_mode'],
            'family_classifier_is_fallible': True}}


def match_models(models):
    first = models[0]
    for model in models[1:]:
        if set(model['records']) != set(first['records']):
            raise ValueError('Models do not cover the same DEV identities')
        for key in ('data_pair_sha256', 'prepared_PNG_manifest_sha256', 'clip_repo_sha', 'weights_sha256', 'family_prototypes'):
            if model['audit_report'][key] != first['audit_report'][key]:
                raise ValueError('Models use different prepared data or diagnostic protocols: '+key)
        for sid, reference in first['records'].items():
            row = model['records'][sid]
            if any(row['metrics'][key] != reference['metrics'][key] for key in ('prompt', 'category', 'mode', 'object_crop_box')):
                raise ValueError('Matched prompts, tasks or offline crop regions differ')
            for role in ('reference', 'target'):
                if row['PNGs'][role]['RGB_sha256'] != reference['PNGs'][role]['RGB_sha256']:
                    raise ValueError('Matched reference/GT pixels differ')
            if row['audit']['requested_family'] != reference['audit']['requested_family']:
                raise ValueError('Matched requested families differ')
            if row['audit']['generated']['wall']['expected_palette'] != reference['audit']['generated']['wall']['expected_palette']:
                raise ValueError('Matched expected wall palettes differ')


def load_curves(curve_paths, models):
    by_label = {m['label']: m for m in models}; curves = {}
    for label, directory in curve_paths.items():
        if label not in by_label:
            raise ValueError('Each curve label must match a supplied model label')
        report = read_json(directory/'report.json'); history = read_json(directory/'history.json')
        if (not report.get('complete') or report.get('test_evaluated') is not False or
                report.get('checkpoint_sha256') != by_label[label]['report']['parent_checkpoint_sha256'] or
                report.get('best_step', 0) <= 0):
            raise ValueError('Curve run is incomplete, tested, or not the selected raw-review parent')
        selected_images = report.get('best_val', {}).get('images', [])
        if {r['sample_id'] for r in selected_images} != set(by_label[label]['records']):
            raise ValueError('Curve development identities differ from the raw review')
        points = []
        for row in history:
            points.append({'epoch': finite(row['epoch'], 'epoch'), 'step': row['step'],
                'DEV_LPIPS': finite(row.get('dev_lpips', row.get('val_lpips')), 'DEV_LPIPS'),
                'DEV_crop_LPIPS': finite(row.get('dev_crop_lpips', row.get('val_object_crop_lpips')), 'DEV_crop_LPIPS'),
                'DEV_PSNR': finite(row.get('dev_psnr', row.get('val_psnr')), 'DEV_PSNR'),
                'TRAIN_diagnostic_PSNR': None if row.get('train_psnr') is None else finite(row['train_psnr'], 'TRAIN_PSNR')})
        if not points or sum(p['step'] == report['best_step'] for p in points) != 1:
            raise ValueError('Curve does not contain a unique selected-best observation')
        curves[label] = {'run_directory': str(directory.resolve()), 'report_sha256': digest(directory/'report.json'),
            'history_sha256': digest(directory/'history.json'), 'best_step': report['best_step'],
            'points': points, 'raw_history_records': history}
    return curves


def fonts():
    paths = ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans.ttf')
    for path in paths:
        try:return [ImageFont.truetype(path, size) for size in (14, 12, 11)]
        except OSError:pass
    return [ImageFont.load_default() for _ in range(3)]


def wrapped_lines(draw, text, font, width):
    """Wrap complete text using measured glyph widths; never truncate prompts."""
    lines = []; current = ''
    for word in str(text).split():
        proposed = (current+' '+word).strip()
        if draw.textlength(proposed, font=font) <= width:
            current = proposed; continue
        if current:lines.append(current); current = ''
        piece = ''
        for char in word:
            if piece and draw.textlength(piece+char, font=font) > width:
                lines.append(piece); piece = ''
            piece += char
        current = piece
    if current:lines.append(current)
    return lines or ['']


def role_lines(title, metric, audit_role):
    result = [title]
    if metric is not None:
        result.append(f"P {metric['psnr']:.2f} | L {metric['lpips']:.3f} | crop {metric['object_crop_lpips']:.3f}")
    if audit_role is not None:
        family = audit_role['family']
        result += ['B16 family prediction / requested:',
            'N/A (background-only)' if family is None else
            str(family['predicted_family'] or 'tie')+' / '+family['requested_family']]
        wall = audit_role['wall']
        result += ['Wall prediction / expected:',
            str(wall['predicted_palette'] or 'tie').replace('_', ' ')+' / '+wall['expected_palette'].replace('_', ' '),
            f"Retained L1 [0,1]: {audit_role['retained']['L1_RGB_01']:.4f}"]
    return result


def fixed_sample_ids(model, mode):
    chosen = []
    for category in CATEGORIES:
        seen = set()
        for sid, record in model['records'].items():
            metric = record['metrics']
            if metric['mode'] != mode or metric['category'] != category or record['source_product'] in seen:
                continue
            seen.add(record['source_product']); chosen.append(sid)
            if len(seen) == 2:break
        if len(seen) != 2:
            raise ValueError('Need two distinct reviewed reference products in each category/task')
    return chosen


def draw_panels(models, output):
    normal, small, mini = fonts(); width = 256*(2+len(models)); chosen_modes = {}
    measure = ImageDraw.Draw(Image.new('RGB', (width, 1)))
    header_text = ['Controlled-room DEV96: used for selection; not independent TEST. Native256 pixels.',
        'P=PSNR(dB); L=LPIPS; crop=object LPIPS. Retained L1 uses RGB[0,1].', WARNING,
        'Rows: first two distinct source products/category/task, without quality filtering. GT diagnostics use the first audit.']
    header = [line for text in header_text for line in wrapped_lines(measure, text, small, width-12)]
    header_height = 8+16*len(header)
    for mode in MODES:
        selected = fixed_sample_ids(models[0], mode); chosen_modes[mode] = selected
        layouts = []; y = header_height
        for sid in selected:
            first = models[0]['records'][sid]
            prompt = wrapped_lines(measure, first['metrics']['prompt'], normal, width-12)
            descriptions = [role_lines('INPUT', None, None),
                role_lines('GT: offline classifier control', None, first['audit']['target_control'])]
            descriptions += [role_lines(m['label'], m['records'][sid]['metrics'], m['records'][sid]['audit']['generated']) for m in models]
            blocks = [[line for text in block for line in wrapped_lines(measure, text, mini, 246)] for block in descriptions]
            details_y = y+22+18*len(prompt)
            image_y = details_y+14*max(len(block) for block in blocks)+8
            layouts.append((sid, y, prompt, details_y, blocks, image_y))
            y = image_y+256+18
        panel = Image.new('RGB', (width, y), 'white'); draw = ImageDraw.Draw(panel)
        for index, line in enumerate(header):draw.text((6, 4+16*index), line, font=small, fill='black')
        for sid, top, prompt, details_y, blocks, image_y in layouts:
            first = models[0]['records'][sid]
            draw.text((6, top+2), sid, font=small, fill='black')
            for index, line in enumerate(prompt):draw.text((6, top+22+18*index), line, font=normal, fill='black')
            paths = [first['PNGs']['reference']['path'], first['PNGs']['target']['path']]
            paths += [m['records'][sid]['PNGs']['generated']['path'] for m in models]
            for column, (block, path) in enumerate(zip(blocks, paths)):
                for index, line in enumerate(block):draw.text((column*256+5, details_y+14*index), line, font=mini, fill='black')
                with Image.open(path) as image:panel.paste(image, (column*256, image_y))
        panel.save(output/(mode+'_matched.png'))
    return chosen_modes


def draw_curves(curves, output):
    if not curves:return
    figure, axes = plt.subplots(1, 4, figsize=(16, 3.7), layout='constrained')
    for label, curve in curves.items():
        points = curve['points']; best = next(p for p in points if p['step'] == curve['best_step'])
        for axis, key in zip(axes[:3], ('DEV_LPIPS', 'DEV_crop_LPIPS', 'DEV_PSNR')):
            line, = axis.plot([p['epoch'] for p in points], [p[key] for p in points], marker='.', label=label)
            axis.scatter([best['epoch']], [best[key]], marker='*', s=100, color=line.get_color())
            axis.set(xlabel='Epoch within stage', ylabel=key); axis.grid(alpha=.2)
        axis = axes[3]
        line, = axis.plot([p['epoch'] for p in points], [p['DEV_PSNR'] for p in points], label=label+' DEV')
        train = [p for p in points if p['TRAIN_diagnostic_PSNR'] is not None]
        if train:
            axis.plot([p['epoch'] for p in train], [p['TRAIN_diagnostic_PSNR'] for p in train],
                linestyle='--', color=line.get_color(), label=label+' TRAIN diagnostic')
        axis.scatter([best['epoch']], [best['DEV_PSNR']], marker='*', s=100, color=line.get_color())
        axis.set(xlabel='Epoch within stage', ylabel='PSNR: TRAIN diagnostic / DEV'); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8); axes[3].legend(fontsize=7)
    figure.suptitle('Same controlled-room DEV96. Stars: selected best. TRAIN diagnostics are not independent generalization.')
    figure.savefig(output/'development_curves.png', dpi=180); plt.close(figure)
    (output/'curve_points.json').write_text(json.dumps(curves, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', required=True, help='LABEL=completed controlled-room raw DEV96 review')
    parser.add_argument('--audit', action='append', required=True, help='Same LABEL=completed SHA-bound CPU instruction audit')
    parser.add_argument('--curve', action='append', default=[], help='Same LABEL=completed selected training run')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError('Require a new immutable output directory')
    model_paths = labelled_paths(args.model, '--model'); audit_paths = labelled_paths(args.audit, '--audit')
    if set(model_paths) != set(audit_paths):raise ValueError('Supply exactly one same-label audit for every model')
    models = [load_model(label, directory, audit_paths[label]) for label, directory in model_paths.items()]
    match_models(models); curves = load_curves(labelled_paths(args.curve, '--curve'), models)
    args.output.mkdir(parents=True)
    figures = draw_panels(models, args.output); draw_curves(curves, args.output)
    rows = []
    for sid, first in models[0]['records'].items():
        rows.append({'sample_id': sid, 'source_product': first['source_product'],
            'category': first['metrics']['category'], 'mode': first['metrics']['mode'], 'prompt': first['metrics']['prompt'],
            'requested_family': first['audit']['requested_family'],
            'models': {m['label']: m['records'][sid] for m in models}})
    (args.output/'plot_data.json').write_text(json.dumps({'complete': True,
        'evaluation': 'Controlled-room DEV96 used for selection, not independent TEST',
        'warning': WARNING, 'pixels': 'Native256 PNG pixels copied unchanged; no resize/enhancement/crop of displayed images',
        'sample_selection': 'First two distinct source products/category/task in the first completed review; no score-based selection',
        'GT_classifier_display_source': models[0]['label'], 'figure_sample_ids': figures,
        'model_provenance': {m['label']: m['provenance'] for m in models}, 'curves': curves, 'images': rows},
        indent=2), encoding='utf-8')


if __name__ == '__main__':main()
