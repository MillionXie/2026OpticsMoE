"""CPU, fixed-seed feasibility previews; never select samples by quality."""
import argparse
import json
import hashlib
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sdxl-run', default='white_reference_sdxl_ipadapter_20261029')
    a = p.parse_args()
    if a.output.exists(): raise ValueError('New review directory required')
    inputs = a.runs/'white_reference_inputs_20261029'
    refs = json.loads((inputs/'reference_manifest.json').read_text(encoding='utf-8'))['references']
    groups = {}
    for model in ['ip2p', 'sdxl_ipadapter']:
        root = a.runs/(a.sdxl_run if model == 'sdxl_ipadapter' else 'white_reference_ip2p_20261029')
        report = json.loads((root/'report.json').read_text(encoding='utf-8'))
        rows = json.loads((root/'images.json').read_text(encoding='utf-8'))
        if not report['complete'] or report['generated_count'] != 16 or len(rows) != 16:
            raise ValueError('Both complete16-image model runs required')
        for row in rows:
            if sha(root/row['generated_PNG']) != row['generated_sha256']: raise ValueError('Output changed')
        groups[model] = (root, rows)
    a.output.mkdir(parents=True)
    for category in ['lamp', 'pillow']:
        chosen = [r for r in refs if r['category'] == category]
        sheet = Image.new('RGB', (768, 42+4*386), 'white'); draw = ImageDraw.Draw(sheet)
        draw.text((5, 4), 'AI-generated outputs | fixed seed1042, all4 category requests | preview only, no success claim', fill='black')
        draw.text((5, 20), 'ABO white reference256 | IP2P native512 | SDXL+IP-Adapter native1024; previews resized256', fill='black')
        for i, (ref, condition) in enumerate((r, c) for r in chosen for c in ['0', '1']):
            y = 42+i*386
            selected = {}
            for model, (root, rows) in groups.items():
                selected[model] = next(r for r in rows if r['reference_id'] == ref['reference_id'] and r['condition_id'] == condition and r['seed'] == 1042)
            if selected['ip2p']['prompt'] != selected['sdxl_ipadapter']['prompt']: raise ValueError('Prompt mismatch')
            draw.text((5, y), ref['reference_id']+' | condition'+condition+' | same full prompt; same integer seed is not identical noise across pipelines', fill='black')
            draw.multiline_text((5, y+18), '\n'.join(textwrap.wrap(selected['ip2p']['prompt'], 124)), fill='black')
            for x, label in [(0, 'Reference (TRAIN)'), (256, 'InstructPix2Pix / AI-generated'), (512, 'SDXL+IP-Adapter / AI-generated')]: draw.text((x+4, y+104), label, fill='black')
            with Image.open(inputs/ref['input_PNG']) as im: sheet.paste(im, (0, y+126))
            for model, x in [('ip2p', 256), ('sdxl_ipadapter', 512)]:
                root, rows = groups[model]
                with Image.open(root/selected[model]['generated_PNG']) as im: sheet.paste(im.resize((256, 256), Image.Resampling.LANCZOS), (x, y+126))
        sheet.save(a.output/(category+'_seed1042_overview.png'))
    summary = Image.new('RGB', (1024, 28+2*500), 'white'); draw = ImageDraw.Draw(summary)
    draw.text((5, 4), 'AI-GENERATED | first source/category, both requests, fixed seed1042; includes failures, not best-of-seed', fill='black')
    root, rows = groups['sdxl_ipadapter']
    for column, category in enumerate(['lamp', 'pillow']):
        ref = next(r for r in refs if r['category'] == category)
        for condition in ['0', '1']:
            x = column*512; y = 28+int(condition)*500
            row = next(r for r in rows if r['reference_id'] == ref['reference_id'] and r['condition_id'] == condition and r['seed'] == 1042)
            prompt = row.get('model_prompt', row['prompt'])
            draw.multiline_text((x+4, y), '\n'.join(textwrap.wrap(prompt, 82)), fill='black')
            draw.text((x+4, y+88), 'ABO reference', fill='black')
            draw.text((x+132, y+88), 'SDXL+IP-Adapter / AI-generated / native1024', fill='black')
            with Image.open(inputs/ref['input_PNG']) as im: summary.paste(im.resize((128, 128), Image.Resampling.LANCZOS), (x, y+110+128))
            with Image.open(root/row['generated_PNG']) as im: summary.paste(im.resize((384, 384), Image.Resampling.LANCZOS), (x+128, y+110))
    summary.save(a.output/'fixed_four_requests.png')
    (a.output/'plot_data.json').write_text(json.dumps({'scope': 'Feasibility review, not independent quality or resolution-matched comparison',
       'fixed_selection': 'Overviews:all4references x2conditions at1042. Summary:first reference/category,bothconditions,1042. No quality selection;1043 remains in complete run.',
       'source_reports': {m: sha(root/'report.json') for m,(root,rows) in groups.items()},
       'rows': [dict(row, source_run=str(root)) for root,rows in groups.values() for row in rows]}, indent=2), encoding='utf-8')
    print(a.output.resolve())


if __name__ == '__main__': main()
