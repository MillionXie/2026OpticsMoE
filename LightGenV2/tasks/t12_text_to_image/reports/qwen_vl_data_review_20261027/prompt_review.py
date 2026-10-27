"""CPU-only, lossless prompt-labelled review of existing VAL predictions."""
import argparse
import base64
import csv
import hashlib
import html
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    for name in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run/'report.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.run/'overview/image_manifest.json').read_text(encoding='utf-8'))
    assert report['complete'] and not report['test_evaluated']
    assert report['checkpoint_sha256'] == manifest['checkpoint_sha256']
    records = report['best_val']['images']
    selected = [(i, r) for i, r in enumerate(records) if r['category'] in ('lamp', 'pillow')]
    assert selected and len({r['sample_id'] for _, r in selected}) == len(selected)
    evaluation_label=report.get('evaluation_label','validation')
    args.output.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, record in selected:
        paths = {}
        for role in ('reference', 'target', 'generated'):
            name = f'{role}/sample_{i:04d}.png'
            path = args.run/'images/val_best'/name
            assert digest(path) == manifest['files'][name]['sha256']
            assert Image.open(path).size == (256, 256)
            paths[role] = path
        entries.append((i, record, paths))
    with (args.output/'all_prompts_metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(r for _, r in selected)
    blocks = ['<!doctype html><meta charset="utf-8"><title>T12 exact prompts and VAL outputs</title>',
              '<style>body{font:16px Arial;max-width:1050px;margin:24px auto}article{border-bottom:1px solid #aaa;padding:16px 0}figure{display:inline-block;margin:8px}img{width:256px;height:256px}pre{white-space:pre-wrap}figcaption{margin:8px}</style>',
              f'<h1>Lamp + pillow: all {len(selected)} {html.escape(evaluation_label)} cases</h1><p>Existing checkpoint outputs, not new inference or training. Native 256 PNG pixels, no enhancement. Target is offline evaluation only. Exact English prompts are retained. Evaluation participates in model selection; no independent TEST claim.</p>']
    for i, r, paths in entries:
        blocks.append(f'<article><h2>{i:02d} {html.escape(r["category"]+" / "+r["mode"])}</h2><pre>{html.escape(r["sample_id"])}</pre><p><b>Prompt:</b> {html.escape(r["prompt"])}</p><p>PSNR {r["psnr"]:.3f} dB; LPIPS {r["lpips"]:.3f}; edited-region PSNR {r["edited_psnr"]:.3f} dB</p>')
        for role, path in paths.items():
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            blocks.append(f'<figure><figcaption>{role}</figcaption><img src="data:image/png;base64,{encoded}"></figure>')
        blocks.append('</article>')
    (args.output/'all_two_category_with_prompts.html').write_text('\n'.join(blocks), encoding='utf-8')
    panels = {}
    for category in ('lamp', 'pillow'):
        panels[category] = [next(e for e in entries if (e[1]['category'], e[1]['mode']) == (category, mode))
                           for mode in ('background', 'object', 'joint')]
    for mode in ('background', 'object', 'joint'):
        examples = []
        for category in ('lamp', 'pillow'):
            seen = set()
            for entry in entries:
                r = entry[1]
                source_id = r['sample_id'].split(':')[2].rsplit('-',1)[0]
                if (r['category'], r['mode']) == (category, mode) and source_id not in seen and len(seen)<2:
                    seen.add(source_id)
                    examples.append(entry)
        assert len(examples) == 4, 'Require both held-out products in both categories'
        panels[mode] = examples
    for name, examples in panels.items():
        # Each row has its full prompt followed by three native 256-pixel tiles.
        sheet = Image.new('RGB', (1024, 480*len(examples)), 'white')
        draw = ImageDraw.Draw(sheet)
        for row, (i, r, paths) in enumerate(examples):
            y = row*480
            draw.text((12, y+8), f'{r["category"]} / {r["mode"]} | PSNR {r["psnr"]:.2f} dB | LPIPS {r["lpips"]:.3f}', font=font(21), fill='black')
            lines = textwrap.wrap(r['prompt'], width=93)
            assert len(lines) <= 6, 'Prompt requires a larger caption area, never truncate'
            draw.multiline_text((12, y+40), '\n'.join(lines), font=font(19), spacing=4, fill='black')
            for col, (role, path) in enumerate(paths.items()):
                x = 12+col*338
                draw.text((x, y+182), role.upper(), font=font(19), fill='black')
                image = Image.open(path).convert('RGB')
                sheet.paste(image, (x, y+212))
                assert np.array_equal(np.asarray(sheet.crop((x, y+212, x+256, y+468))), np.asarray(image))
        sheet.save(args.output/f'{name}_prompt_examples.png')
    summary = {'scope': 'Existing checkpoint, lamp/pillow development evaluation outputs; no independent TEST claim',
               'evaluation_label':evaluation_label,
               'checkpoint_sha256': report['checkpoint_sha256'], 'source_report_sha256': digest(args.run/'report.json'),
               'input_manifest_sha256': digest(args.run/'overview/image_manifest.json'),
               'renderer_source_sha256': digest(Path(__file__)), 'exact_prompts': True, 'native_tile_size': [256, 256],
               'train_launched': False, 'test_opened': False, 'samples': [r['sample_id'] for _, r in selected],
               'subset_metrics': {key: sum(r[key] for _, r in selected)/len(selected) for key in ('psnr', 'lpips')},
               'output_sha256': {p.name: digest(p) for p in args.output.iterdir() if p.is_file() and p.suffix in ('.png', '.html', '.csv')}}
    (args.output/'review_manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Verified and labelled', len(entries), 'existing VAL cases; no GPU or training used')


if __name__ == '__main__':
    main()
