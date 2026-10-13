"""Build the complete numbered comparison catalogue from audited predictions.

Two independent commands: images needs the task's Torch/resvg environment;
pdf needs only ReportLab/Pillow. No inference, selection or prediction repair.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import tarfile

OPERATIONS = ('add', 'replace', 'move', 'remove')
FILES = ('01_input.png', '02_target.png', '03_ours.png', '04_qwen.png')
ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pairs(evidence):
    from .export_full_comparison import IDENTITIES, MANIFESTS
    provenance = json.loads((evidence / 'provenance.json').read_text())
    pairs = []
    for name in ('ours', 'qwen'):
        assert provenance['models'][name]['checkpoint_sha256'] == IDENTITIES[name][4]
    for split, count in (('test', 1000), ('train', 5000)):
        records = {}
        for name in ('ours', 'qwen'):
            path = evidence / f'{name}_{split}.jsonl'
            meta = provenance['models'][name]['splits'][split]
            assert meta['manifest_sha256'] == MANIFESTS[split]
            assert digest(path) == meta['records_sha256']
            records[name] = {r['sample_id']: r for r in map(json.loads, path.read_text().splitlines())}
            assert len(records[name]) == count
        assert records['ours'].keys() == records['qwen'].keys()
        for operation in OPERATIONS:
            group = sorted(r for r in records['ours'] if records['ours'][r]['task'] == operation)
            assert len(group) == count // 4
            for sample_id in group:
                ours, qwen = (records[name][sample_id] for name in ('ours', 'qwen'))
                for key in ('sample_id', 'task', 'instruction', 'source_grid', 'target_grid',
                            'relative_dir', 'original_png_sha256'):
                    assert ours[key] == qwen[key], (sample_id, key)
                if split == 'test':
                    assert ours['archived_metric_max_error'] < 1e-6
                    if qwen['archived_metric_max_error'] >= 1e-6:
                        assert sample_id == 'test_000486'
                        assert set(qwen['archived_secondary_differences']) == {'object_f1'}
                for record in (ours, qwen):
                    source, target, prediction, edit = (record[k] for k in
                        ('source_grid', 'target_grid', 'prediction_grid', 'predicted_edit_grid'))
                    assert all(len(g) == 6 and all(len(r) == 6 for r in g)
                               for g in (source, target, prediction, edit))
                    cells = [(source[r][c], target[r][c], prediction[r][c], bool(edit[r][c]))
                             for r in range(6) for c in range(6)]
                    changed = [v for v in cells if v[0] != v[1]]
                    assert changed
                    cca = sum(p == t for s, t, p, e in changed) / len(changed)
                    union = sum(e or s != t for s, t, p, e in cells)
                    iou = sum(e and s != t for s, t, p, e in cells) / union if union else 1.
                    assert abs(cca-record['metrics']['changed_cell_accuracy']) < 1e-6
                    assert abs(iou-record['metrics']['edit_grid_iou']) < 1e-6
                pairs.append({'number': len(pairs) + 1, 'split': split, 'operation': operation,
                              'sample_id': sample_id, 'instruction': ours['instruction'],
                              'ours': ours, 'qwen': qwen})
    return provenance, pairs


def plan_pages(pairs):
    """Deterministic plan shared by PDF pages, image folders and machine index."""
    full_page = 2
    split_pages = {'test': 2, 'train': 2}
    sections = []
    for split in ('test', 'train'):
        for operation in OPERATIONS:
            rows = [r for r in pairs if (r['split'], r['operation']) == (split, operation)]
            sections.append({'split': split, 'operation': operation, 'count': len(rows),
                             'first_number': rows[0]['number'], 'last_number': rows[-1]['number'],
                             'master_section_page': full_page,
                             'split_section_page': split_pages[split]})
            full_page += 1
            split_pages[split] += 1
            for i, row in enumerate(rows):
                row['master_pdf_page'] = full_page + i // 4
                row['split_pdf_page'] = split_pages[split] + i // 4
                row['folder'] = (f'{"01_test" if split == "test" else "02_train"}/'
                                 f'{OPERATIONS.index(operation)+1:02d}_{operation}/'
                                 f'{row["number"]:06d}_{row["sample_id"]}')
            pages = math.ceil(len(rows) / 4)
            full_page += pages
            split_pages[split] += pages
    return sections, {'all': full_page - 1, **{s: p - 1 for s, p in split_pages.items()}}


def images(args):
    import numpy as np
    from PIL import Image
    from .layered_scene_data import ANCHOR_X, ANCHOR_BOTTOM, DEPTH_GROUP, _sprite, _artwork
    provenance, pairs = load_pairs(args.evidence)
    sections, pages = plan_pages(pairs)
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a fresh catalogue directory')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    archive = args.evidence / 'original_pngs.tar.gz'
    assert digest(archive) == provenance['original_images']['archive_sha256']

    @lru_cache(maxsize=64)
    def sprite(category, scale):
        base = _sprite(args.svg_dir, category)
        return _artwork(str(args.svg_dir.resolve()), category).resize(
            (base.width * scale, base.height * scale), Image.Resampling.LANCZOS)

    def scene(grid, scale):
        array = np.asarray(grid)
        assert array.shape == (6, 6) and array.min() >= 0 and array.max() <= 16
        canvas = Image.new('RGBA', (224 * scale, 224 * scale), (250, 250, 247, 255))
        clipped = []
        objects = sorted((DEPTH_GROUP.get(int(array[r, c]), 4), int(r), int(c), int(array[r, c]))
                         for r, c in zip(*np.nonzero(array)))
        for _, r, c, category in objects:
            icon = sprite(category, scale)
            base = _sprite(args.svg_dir, category)
            x, y = (ANCHOR_X[c] - base.width // 2) * scale, (ANCHOR_BOTTOM[r] - base.height) * scale
            if x < 0 or y < 0 or x + icon.width > canvas.width or y + icon.height > canvas.height:
                clipped.append({'category': category, 'anchor_rc': [r, c]})
            # Composite only the visible rectangle. This is pixel-identical
            # to a full-size transparent layer but avoids 896px blank work.
            left, top = max(0, x), max(0, y)
            right, bottom = min(canvas.width, x + icon.width), min(canvas.height, y + icon.height)
            if left < right and top < bottom:
                patch = icon.crop((left-x, top-y, right-x, bottom-y))
                canvas.alpha_composite(patch, (left, top))
        return canvas.convert('RGB'), clipped

    # Read the 122 MiB original archive once, avoiding gzip backwards seeks
    # as operation-grouped examples traverse the interleaved source order.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == 12000
        originals = {}
        for member in members:
            assert member.isfile() and member.size < 1024 * 1024
            originals[member.name] = tar.extractfile(member).read()
    for category in range(1, 17):
        for scale in (1, 4):
            sprite(category, scale)

    def render_row(row):
            folder = args.output / row['folder']
            case_path = folder / 'case.json'
            if args.resume and case_path.exists():
                try:
                    existing = json.loads(case_path.read_text(encoding='utf-8'))
                except json.JSONDecodeError:
                    existing = None  # Our interrupted write, not a completed case.
                if existing is not None:
                    for key in row:
                        assert existing[key] == row[key], ('Resume identity mismatch', row['sample_id'], key)
                    for file, sha in existing['png_sha256'].items():
                        assert digest(folder / file) == sha
                    for file, sha in existing['paper_png_sha256'].items():
                        assert digest(folder / 'paper_896' / file) == sha
                    row.update(existing)
                    return
            highres = folder / 'paper_896'
            highres.mkdir(parents=True, exist_ok=args.resume)
            row['clipped_predictions'] = {}
            row['png_sha256'] = {}
            row['paper_png_sha256'] = {}
            row['image_files'] = [f'{row["number"]:06d}_{file}' for file in FILES]
            for j, (key, model) in enumerate((('source_grid', 'ours'), ('target_grid', 'ours'),
                                             ('prediction_grid', 'ours'), ('prediction_grid', 'qwen'))):
                original, clipped = scene(row[model][key], 1)
                filename = row['image_files'][j]
                path = folder / filename
                if j < 2:
                    name = ('source.png', 'target.png')[j]
                    raw = originals[row['ours']['relative_dir'] + '/' + name]
                    assert hashlib.sha256(raw).hexdigest() == row['ours']['original_png_sha256'][name]
                    with Image.open(BytesIO(raw)) as source:
                        assert np.array_equal(np.asarray(source.convert('RGB')), np.asarray(original)), row['sample_id']
                    path.write_bytes(raw)  # Exact acquired dataset PNG bytes, not re-encoded.
                else:
                    original.save(path, compress_level=2)
                    row['clipped_predictions'][model] = clipped
                large, _ = scene(row[model][key], 4)
                large.save(highres / filename, compress_level=2)
                row['png_sha256'][filename] = digest(path)
                row['paper_png_sha256'][filename] = digest(highres / filename)
            case_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(render_row, pairs)):
            if (i + 1) % 100 == 0:
                print(f'Rendered or verified {i+1}/6000 numbered folders', flush=True)
    index = {'provenance': provenance, 'sections': sections, 'pdf_pages': pages, 'cases': pairs,
             'native_png_count': 24000, 'paper_png_count': 24000,
             'svg_asset_sha256': {p.name: digest(p) for p in sorted(args.svg_dir.glob('*.svg'))},
             'renderer_sha256': digest(__file__)}
    (args.output / '00_index.json').write_text(json.dumps(index, ensure_ascii=False) + '\n', encoding='utf-8')
    write_viewer(pairs, args.output)
    print(json.dumps({'cases': len(pairs), 'pages': pages, 'output': str(args.output)}, indent=2))


def write_viewer(pairs, output):
    # Offline searchable viewer; results are not filtered or reordered by score.
    html = r'''<!doctype html><meta charset="utf-8"><title>Full scene comparison index</title>
<style>body{font:15px Arial;margin:28px;background:#fafafa}table{border-collapse:collapse;width:100%}
td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{width:100px}input,select{padding:8px;margin:8px}</style>
<h1>OpenMoji: all 6,000 paired predictions</h1><p>1,000 TEST + 5,000 TRAIN. CCA / IoU values are fractions (1 = 100%).
Raw 224px PNGs and unannotated 896px paper exports are in each case folder.</p>
<input id="query" placeholder="Number, sample ID or instruction" size="48"><select id="split"><option value="">All splits</option><option>test</option><option>train</option></select>
<select id="op"><option value="">All operations</option><option>add</option><option>replace</option><option>move</option><option>remove</option></select>
<span id="count"></span><table><thead><tr><th>ID / pages</th><th>Instruction</th><th>Input</th><th>Target</th><th>Ours CCA / IoU</th><th>Qwen CCA / IoU</th></tr></thead><tbody id="rows"></tbody></table>
<script>const data=DATA;const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function show(){const q=document.querySelector('#query').value.trim().toLowerCase(),s=document.querySelector('#split').value,o=document.querySelector('#op').value,exact=/^\d{1,6}$/.test(q);
const selected=data.filter(r=>(!s||r.split===s)&&(!o||r.operation===o)&&(exact?r.number===Number(q):(`${r.number} ${r.sample_id} ${r.instruction}`.toLowerCase().includes(q))));
document.querySelector('#count').textContent=`${selected.length} cases; showing first 100 (narrow search to find others)`;
document.querySelector('#rows').innerHTML=selected.slice(0,100).map(r=>`<tr><td><a href="${r.folder}/case.json">${String(r.number).padStart(6,'0')}<br>${r.sample_id}</a><br>${r.split} / ${r.operation}<br>Full p.${r.master_pdf_page}; split p.${r.split_pdf_page}</td><td>${esc(r.instruction)}</td>`+
r.image_files.map((f,i)=>`<td><a href="${r.folder}/paper_896/${f}"><img loading="lazy" src="${r.folder}/${f}"></a>${i>1?`<br>${r[i===2?'ours':'qwen'].metrics.changed_cell_accuracy.toFixed(3)} / ${r[i===2?'ours':'qwen'].metrics.edit_grid_iou.toFixed(3)}`:''}</td>`).join('')+'</tr>').join('');}
document.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',show));show();</script>'''
    (output / '00_browse.html').write_text(html.replace('DATA', json.dumps(pairs, ensure_ascii=True).replace('</', '<\\/')), encoding='utf-8')


def pdf(args):
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor
    from reportlab.lib.utils import ImageReader
    from pypdf import PdfReader
    index = json.loads((args.output / '00_index.json').read_text())
    pairs = index['cases']
    destination = args.output / 'pdf'
    destination.mkdir(exist_ok=False)
    width, height = A4
    ink, gray = HexColor('#202b36'), HexColor('#63717f')
    pdf_files = {'all': '00_all_train_test.pdf', 'test': '01_test.pdf', 'train': '02_train.pdf'}
    written = {}
    for split, filename in pdf_files.items():
        path = destination / filename
        c = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
        c.setTitle(f'OpenMoji complete {split.upper()} comparison catalogue')
        c.setAuthor('LightGenV2 - audited fixed-checkpoint results')
        selected = pairs if split == 'all' else [r for r in pairs if r['split'] == split]

        def text(x, y, value, size=10, bold=False, color=ink):
            c.setFillColor(color)
            font = 'Helvetica-Bold' if bold else 'Helvetica'
            assert c.stringWidth(value, font, size) <= width-x-30, ('Text exceeds page', value)
            c.setFont(font, size)
            c.drawString(x, y, value)

        def footer():
            text(34, 25, 'OpenMoji layered scenes | fixed predictions | CCA / IoU in [0,1]', 8, color=gray)
            text(width - 70, 25, f'{c.getPageNumber():04d}', 9)

        text(40, height-70, 'Text-guided scene editing', 24, True)
        text(40, height-100, f'Complete {split.upper()} comparison catalogue', 18)
        text(40, height-137, f'{len(selected):,} examples. No performance-based filtering.', 12)
        descriptions = [
            'Four columns: original input, target, Ours prediction, Qwen prediction.',
            'Each case folder contains four unannotated 224px PNGs and 896px SVG renders.',
            'The 224px input and target PNG files are exact copies of the dataset originals.',
            'Prediction scenes use the identical compositor, sizes, anchors and depth order.',
            'Wrong predictions are shown unchanged. Out-of-canvas objects remain clipped.',
            'CCA: accuracy on cells changed by the instruction (not full-image accuracy).',
            'IoU: predicted edit mask vs. target edit mask, including incorrect edits.',
            'TRAIN is training fit. TEST was used for model selection: development metrics.',
            'Ours: selected epoch 45, 30% unmodulated intensity per SLM; small CCD noise',
            'during training only. Clean fixed-checkpoint evaluation is shown here.',
            'Qwen: frozen complete native vision/language features, trained shared readout.',
            'Cached backbone features are reused. No fresh Qwen backbone timing is implied.',
            'Overall TEST CCA: Ours 0.8765; Qwen 0.8120. Same 1,000 TEST inputs.',
            'One Qwen BF16 boundary case has secondary F1 drift; CCA / IoU are unchanged.',
            'Stable case numbers: 000001-001000 TEST; 001001-006000 TRAIN.',
            'Use the operation bookmarks, PDF search or 00_browse.html to locate a case.',
        ]
        for i, line in enumerate(descriptions):
            text(40, height-177-i*20, line, 9.4)
        footer(); c.showPage()
        for section in index['sections']:
            if split != 'all' and section['split'] != split:
                continue
            group = [r for r in selected if r['split'] == section['split'] and r['operation'] == section['operation']]
            expected = section['master_section_page'] if split == 'all' else section['split_section_page']
            assert c.getPageNumber() == expected
            mark = section['split'] + '_' + section['operation']
            c.bookmarkPage(mark)
            c.addOutlineEntry(mark.upper().replace('_', ' / '), mark, 0)
            text(40, height-85, mark.upper().replace('_', ' / '), 26, True)
            text(40, height-125, f'{len(group):,} examples. Case {group[0]["number"]:06d} - {group[-1]["number"]:06d}.', 14)
            y = height-175
            for name, label in (('ours', 'Ours'), ('qwen', 'Qwen baseline')):
                metrics = index['provenance']['models'][name]['splits'][section['split']]['metrics'][section['operation']]
                text(40, y, label, 14, True)
                text(40, y-25, f'Changed-cell accuracy: {metrics["changed_cell_accuracy"]:.6f}', 12)
                text(40, y-48, f'Edit-grid IoU: {metrics["edit_grid_iou"]:.6f}', 12)
                y -= 100
            text(40, y-15, 'Sorted by original sample ID within this operation, not by performance.', 10)
            footer(); c.showPage()
            for start in range(0, len(group), 4):
                chunk = group[start:start+4]
                text(34, height-32, mark.upper().replace('_', ' / '), 13, True)
                labels = ('Original input', 'Target', 'Ours', 'Qwen baseline')
                gap = 9
                side = (width - 68 - gap*3) / 4
                for j, label in enumerate(labels):
                    text(34+j*(side+gap), height-54, label, 10, True)
                for i, row in enumerate(chunk):
                    expected_page = row['master_pdf_page'] if split == 'all' else row['split_pdf_page']
                    assert c.getPageNumber() == expected_page
                    top = height-82-i*181
                    text(34, top, f'#{row["number"]:06d}   {row["sample_id"]}', 10, True)
                    text(34, top-15, row['instruction'], 9)
                    for j, file in enumerate(row['image_files']):
                        x, y = 34+j*(side+gap), top-25-side
                        c.drawImage(ImageReader(str(args.output/row['folder']/file)), x, y, side, side)
                        if j > 1:
                            metric = row['ours' if j == 2 else 'qwen']['metrics']
                            text(x, y-13, f'CCA {metric["changed_cell_accuracy"]:.3f}   IoU {metric["edit_grid_iou"]:.3f}', 8.7)
                    c.setStrokeColor(HexColor('#dce2e8'))
                    c.line(34, top-170, width-34, top-170)
                footer(); c.showPage()
                if start % 400 == 0:
                    print(f'PDF {split}: {mark} cases {start+len(chunk)}/{len(group)}', flush=True)
        c.save()
        reader = PdfReader(path)
        assert len(reader.pages) == index['pdf_pages'][split]
        # Exact ID coverage via page text, not merely a page-count check.
        import re
        observed = []
        for page in reader.pages:
            observed.extend(int(n) for n in re.findall(r'#(\d{6})', page.extract_text()))
        assert observed == [r['number'] for r in selected]
        written[split] = {'file': str(path.relative_to(args.output)), 'pages': len(reader.pages),
                          'cases': len(observed), 'sha256': digest(path), 'bytes': path.stat().st_size}
    (args.output/'pdf_manifest.json').write_text(json.dumps(written, indent=2)+'\n')
    print(json.dumps(written, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('images', 'pdf', 'viewer'))
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--svg-dir', type=Path, default=ROOT/'assets/openmoji-17.0.0-svg')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Resume own interrupted catalogue after SHA checks')
    args = parser.parse_args()
    if args.command == 'images' and args.evidence is None:
        parser.error('images requires --evidence')
    if args.command == 'viewer':
        write_viewer(json.loads((args.output / '00_index.json').read_text(encoding='utf-8'))['cases'], args.output)
    else:
        (images if args.command == 'images' else pdf)(args)
