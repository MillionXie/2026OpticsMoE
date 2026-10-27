"""Native PNG review of hybrid/compressed model outputs with exact prompts."""
import argparse
import base64
import csv
import html
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from .qwen_vl_turbo_data import sha256, write_json
from .reports.qwen_vl_data_review_20261027.prompt_review import font


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.run/'report.json').read_text())
    if not report['complete'] or report['best_step'] <= 0:
        raise ValueError('Need a completed trained checkpoint')
    records = report['best_val']['images']
    output = a.run/'prompt_review'; output.mkdir(exist_ok=True)
    manifest, articles = {}, []
    for i, r in enumerate(records):
        pictures = {}
        for role in ('reference', 'target', 'generated'):
            path = a.run/'images/dev_best'/role/f'sample_{i:04d}.png'
            if Image.open(path).size != (256, 256):
                raise ValueError('Require native unenhanced PNGs')
            manifest[str(path.relative_to(a.run))] = sha256(path)
            pictures[role] = path
        fragment = f'<article><h2>{html.escape(r["category"]+" / "+r["mode"])}</h2><p>{html.escape(r["sample_id"])}</p><p>{html.escape(r["prompt"])}</p><p>PSNR {r["psnr"]:.3f} dB; LPIPS {r["lpips"]:.4f}; object LPIPS {r["object_crop_lpips"]:.4f}</p>'
        for role, path in pictures.items():
            fragment += f'<figure><figcaption>{role}</figcaption><img width="256" height="256" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"></figure>'
        articles.append(fragment+'</article>')
    for mode in ('background', 'object', 'joint'):
        selected, counts, used = [], {}, set()
        for i, r in enumerate(records):
            key = r['category']; product = r['sample_id'].split(':')[2].rsplit('-', 1)[0]
            if r['mode'] != mode or counts.get(key, 0) >= 2 or (key, product) in used:
                continue
            used.add((key, product)); counts[key] = counts.get(key, 0)+1; selected.append((i, r))
        panel = Image.new('RGB', (1024, 480*len(selected)), 'white'); draw = ImageDraw.Draw(panel)
        for j, (i, r) in enumerate(selected):
            y = j*480
            draw.text((12, y+8), f"{r['category']} / {mode} | PSNR {r['psnr']:.2f} | LPIPS {r['lpips']:.3f}", font=font(20), fill='black')
            lines = textwrap.wrap(r['prompt'], width=93)
            if len(lines) > 6:
                raise ValueError('Never truncate a prompt to fit a panel')
            draw.multiline_text((12, y+40), '\n'.join(lines), font=font(19), fill='black', spacing=4)
            for k, role in enumerate(('reference', 'target', 'generated')):
                draw.text((12+k*338, y+182), role, font=font(19), fill='black')
                panel.paste(Image.open(a.run/'images/dev_best'/role/f'sample_{i:04d}.png'), (12+k*338, y+212))
        panel.save(output/(mode+'_prompt_examples.png'))
    (output/'all_cases.html').write_text('<!doctype html><meta charset="utf-8"><style>figure{display:inline-block}article{border-bottom:1px solid #888}body{font:16px Arial;max-width:1080px;margin:24px auto}</style><h1>Development TEST, used for model selection</h1>'+''.join(articles), encoding='utf-8')
    with (output/'all_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    write_json(output/'manifest.json', {'checkpoint_sha256': report['checkpoint_sha256'],
          'count': len(records), 'native_png_sha256': manifest, 'exact_prompts': True,
          'evaluation_label': 'development_test', 'no_sharpening_or_rgb_paste': True})


if __name__ == '__main__':
    main()
