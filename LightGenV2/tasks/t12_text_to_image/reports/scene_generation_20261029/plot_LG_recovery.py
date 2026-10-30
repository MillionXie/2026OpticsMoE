"""Hash-checked full-cohort LG recovery review; no quality-based selection."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw, ImageFont

METRICS = ['CLIP_T', 'DINOv2_I', 'CLIP_I', 'ImageReward_compact']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    for name in ['C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def cohort_scope(reports, count, frozen_test=False):
    """A presentation flag cannot silently relabel development outputs as TEST."""
    if not reports or not all(r['complete'] for r in reports):
        raise ValueError('Only completed generation runs may be compared')
    if frozen_test:
        if count != 500 or not all(r.get('scope') == 'TEST' and r.get('independent_TEST_opened') for r in reports):
            raise ValueError('Explicit frozen TEST comparison requires all 500 requests per model')
        identities = {r.get('frozen_TEST_protocol', {}).get('manifest_sha256') for r in reports}
        if len(identities) != 1:
            raise ValueError('Different frozen model/request protocols cannot be pooled')
        identity = next(iter(identities))
        if not isinstance(identity, str) or len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('Missing frozen evaluation manifest identity')
        return 'frozen TEST', identity
    if count not in (32, 128) or any(r.get('scope') == 'TEST' or r.get('independent_TEST_opened') for r in reports):
        raise ValueError('Use an explicit frozen TEST flag; do not mix development and TEST')
    return 'DEV', None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tolerance',type=float,default=.05)
    p.add_argument('--reward-normalization',choices=['absolute','baseline-magnitude'],default='absolute')
    p.add_argument('--frozen-test',action='store_true',help='Require 500 completed same-freeze TEST requests per model')
    a = p.parse_args()
    assert not a.output.exists() and len(a.runs) == len(a.labels) and 0<=a.tolerance<1
    rows = json.loads((a.root/a.runs[0]/'images.json').read_text())
    scores = {(r['run_id'], r['sample_id']): r for r in json.loads((a.metrics/'per_image.json').read_text())}
    summary = json.loads((a.metrics/'summary.json').read_text())
    reports = [json.loads((a.root/name/'report.json').read_text()) for name in a.runs]
    scope, frozen_identity = cohort_scope(reports, len(rows), a.frozen_test)
    assert len({r['sample_id'] for r in rows}) == len(rows)
    by_run = {}
    for name in a.runs:
        folder = a.root/name
        assert json.loads((folder/'report.json').read_text())['complete']
        actual = json.loads((folder/'images.json').read_text())
        by_run[name] = {r['sample_id']: r for r in actual}
        assert set(by_run[name]) == {r['sample_id'] for r in rows}
        for r in rows:
            item = by_run[name][r['sample_id']]
            for k in ['prompt', 'seed', 'input_sha256', 'initial_latent_sha256']:
                assert item[k] == r[k], (name, k)
            s = scores[name, r['sample_id']]
            assert s['generated_sha256'] == item['generated_sha256'] == sha(folder/item['generated_PNG'])
            assert sha(folder/(r['reference_id']+'.png')) == r['input_sha256']
            assert all(math.isfinite(s[k]) for k in METRICS)
    a.output.mkdir(parents=True)
    pages = []
    w = (len(a.runs)+1)*258
    for start in range(0, len(rows), 4):
        panel = Image.new('RGB', (w, 1380), 'white')
        d = ImageDraw.Draw(panel)
        d.text((8, 5), f'All {scope} requests, fixed order | input 256 / output 512 | same initial noise', font=font(17), fill='black')
        d.text((8, 30), 'T=CLIP-T, D=DINO-I, I=CLIP-I, R=ImageReward compact. Proxies, not instruction accuracy.', font=font(14), fill='black')
        for i, r in enumerate(rows[start:start+4]):
            y = 60+i*330
            d.multiline_text((8, y), '\n'.join(textwrap.wrap(r['prompt'], max(90, w//8))), font=font(12), fill='black')
            d.text((8, y+39), r['reference_id']+' / scene '+r['condition_id'], font=font(11), fill='black')
            d.text((8, y+58), 'ABO reference', font=font(13), fill='black')
            with Image.open(a.root/a.runs[0]/(r['reference_id']+'.png')) as im:
                panel.paste(im.convert('RGB').resize((248, 248), Image.Resampling.LANCZOS), (5, y+79))
            for j, (name, label) in enumerate(zip(a.runs, a.labels)):
                x = (j+1)*258+5
                s = scores[name, r['sample_id']]
                d.text((x, y+43), label, font=font(13), fill='black')
                d.text((x, y+61), ' '.join(f'{k}:{s[m]:.3f}' for k,m in zip('TDIR',METRICS)), font=font(11), fill='black')
                with Image.open(a.root/name/by_run[name][r['sample_id']]['generated_PNG']) as im:
                    panel.paste(im.convert('RGB').resize((248, 248), Image.Resampling.LANCZOS), (x, y+79))
        name = f'QA_page{start//4+1}.png'
        panel.save(a.output/name)
        pages.append(name)
    scope_note = ('Frozen weights/protocol before TEST prediction; results are not for retuning. Seed draws share references/contexts and are not independent products.' if a.frozen_test else 'No TEST use.')
    heading = [f'<h1>LG recovery: all {len(rows)} {scope} requests</h1>', f'<p>Same reference, two-sentence prompt and initial noise. Native output 512. {scope_note} No image feedback. Scores are proxies; semantic review is required.</p>',
        f'<p>Group means relative to the first model. Relative tolerance: {a.tolerance:.0%}; negative ImageReward normalization: {a.reward_normalization}. Green/red checks proxy means, not complete instruction compliance or statistical equivalence.</p>',
        '<table><tr><th>Model</th><th>Group</th><th>CLIP-T</th><th>DINO-I</th><th>CLIP-I</th><th>IR compact</th></tr>']
    for name, label in zip(a.runs, a.labels):
        for group in ['all','lamp','pillow']:
            s=summary[name+'/'+group]['scores'];reference=summary[a.runs[0]+'/'+group]['scores'];cells=[]
            for metric in METRICS:
                value=s[metric]['mean'];base=reference[metric]['mean']
                absolute=metric=='ImageReward_compact' and a.reward_normalization=='absolute'
                change=value-base if absolute else (value-base)/abs(base)
                text=f'{change:+.3f}' if absolute else f'{change:+.1%}'
                passed=change>=(-.1 if absolute else -a.tolerance)
                suffix='' if name==a.runs[0] else f'<br><small style="color:{"#176838" if passed else "#a02020"}">{text}</small>'
                cells.append(f'<td>{value:.4f}{suffix}</td>')
            heading.append('<tr><td>'+html.escape(label)+'</td><td>'+group+'</td>'+''.join(cells)+'</tr>')
    heading.append('</table>')
    heading.extend(f'<p><a href="{n}"><img loading="lazy" style="width:100%;max-width:1300px" src="{n}"></a></p>' for n in pages)
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>LG recovery review</title><style>body{font:16px Arial;margin:24px}td,th{padding:8px;border:1px solid #aaa}table{border-collapse:collapse}</style>'+''.join(heading),encoding='utf8')
    (a.output/'plot_data.json').write_text(json.dumps([scores[n,r['sample_id']] for r in rows for n in a.runs],indent=2),encoding='utf8')
    (a.output/'report.json').write_text(json.dumps({'complete':True,'all_requests':len(rows),'model_runs':a.runs,'pages':pages,'image_and_noise_SHA_checked':True,'quality_selected':False,'source_sha256':sha(Path(__file__)),
        'relative_tolerance':a.tolerance,'ImageReward_normalization':a.reward_normalization,
        'scope':scope,'frozen_evaluation_manifest_sha256':frozen_identity,
        'all_image_SHA_checked_is_not_a_claim_of_all_image_manual_semantic_review':True},indent=2))
    print(a.output.resolve())


if __name__ == '__main__':
    main()
