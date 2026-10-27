"""Create matched development figures from immutable native model exports."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', action='append', required=True, help='LABEL=completed raw review or run')
    p.add_argument('--curve', action='append', default=[], help='LABEL=completed training run')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    models, provenance = [], {}
    for value in a.model:
        label, directory = value.split('=', 1)
        directory = Path(directory)
        report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
        if not report['complete']:
            raise ValueError('Only completed exports may be compared')
        metrics = report.get('metrics', report.get('best_val'))
        folder = 'raw_dev' if 'metrics' in report else 'dev_best'
        records = {r['sample_id']: (i, r) for i, r in enumerate(metrics['images'])}
        if len(records) != 96:
            raise ValueError('Expected the same development 96, not independent TEST')
        models.append((label, directory, folder, records))
        provenance[label] = {'directory': str(directory.resolve()), 'report_sha256': digest(directory/'report.json'),
                             'metrics': {k:v for k,v in metrics.items() if k != 'images'}}
    if any(set(m[3]) != set(models[0][3]) for m in models):
        raise ValueError('Model sample identities differ')
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
        small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
    except OSError:
        font = small = ImageFont.load_default()
    figure_rows = {}
    for mode in ('background', 'object', 'joint'):
        chosen = []
        for category in ('lamp', 'pillow'):
            seen = set()
            for sid, (_, row) in models[0][3].items():
                product = sid.split(':')[2].rsplit('-', 1)[0]
                if row['mode'] == mode and row['category'] == category and product not in seen:
                    seen.add(product); chosen.append(sid)
                if len(seen) == 2:
                    break
        figure_rows[mode] = chosen
        height, width = 384, 256*(2+len(models))
        panel = Image.new('RGB', (width, height*len(chosen)+32), 'white')
        draw = ImageDraw.Draw(panel)
        draw.text((6, 4), 'DEV96: selected during development. Candidate failures retained; native 256 PNGs.', fill='black', font=font)
        for j, sid in enumerate(chosen):
            index, row = models[0][3][sid]
            y = 32+j*height
            draw.text((6, y), sid, fill='black', font=small)
            draw.multiline_text((6,y+22), '\n'.join(textwrap.wrap(row['prompt'], max(70, width//9))),
                                fill='black', font=font, spacing=3)
            model0 = models[0]
            for k, role in enumerate(('reference', 'target')):
                path = model0[1]/'images'/model0[2]/role/f'sample_{index:04d}.png'
                draw.text((k*256+5,y+82), 'INPUT' if role == 'reference' else 'GT (offline only)', fill='black', font=font)
                with Image.open(path) as image:
                    if image.size != (256,256): raise ValueError('Native pixel size changed')
                    panel.paste(image,(k*256,y+122))
            for k, (label, directory, folder, records) in enumerate(models,2):
                i, metric = records[sid]
                if metric['prompt'] != row['prompt']: raise ValueError('Prompts differ')
                for role in ('reference','target'):
                    if digest(directory/'images'/folder/role/f'sample_{i:04d}.png') != digest(model0[1]/'images'/model0[2]/role/f'sample_{index:04d}.png'):
                        raise ValueError('Matched source/GT pixels differ')
                draw.text((k*256+5,y+82), label, fill='black', font=font)
                draw.text((k*256+5,y+102), f"P {metric['psnr']:.2f} | L {metric['lpips']:.3f} | crop {metric['object_crop_lpips']:.3f}", fill='black', font=small)
                with Image.open(directory/'images'/folder/'generated'/f'sample_{i:04d}.png') as image:
                    if image.size != (256,256): raise ValueError('Native prediction size changed')
                    panel.paste(image,(k*256,y+122))
        panel.save(a.output/(mode+'_matched.png'))
    curves = {}
    if a.curve:
        fig, axes = plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
        for value in a.curve:
            label, directory = value.split('=',1);directory=Path(directory)
            history=json.loads((directory/'history.json').read_text(encoding='utf-8'))
            report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
            if not report['complete']: raise ValueError('Curves require completed stages')
            points=[]
            for r in history:
                points.append({'epoch':r['epoch'],'step':r['step'],
                    'DEV_LPIPS':r.get('dev_lpips',r.get('val_lpips')),
                    'DEV_crop_LPIPS':r.get('dev_crop_lpips',r.get('val_object_crop_lpips')),
                    'DEV_PSNR':r.get('dev_psnr',r.get('val_psnr')),
                    'TRAIN_diagnostic_PSNR':r.get('train_psnr')})
            curves[label]={'points':points,'best_step':report['best_step'],'history_sha256':digest(directory/'history.json')}
            for ax,key in zip(axes,('DEV_LPIPS','DEV_crop_LPIPS','DEV_PSNR')):
                line,=ax.plot([v['epoch'] for v in points],[v[key] for v in points],marker='.',label=label)
                best=next(v for v in points if v['step']==report['best_step'])
                ax.scatter([best['epoch']],[best[key]],marker='*',s=100,color=line.get_color())
                ax.set(xlabel='Epoch within this stage',ylabel=key);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle('Same DEV96; each stage has its own initialization. Stars: selected best.')
        fig.savefig(a.output/'development_curves.png',dpi=180);plt.close(fig)
    (a.output/'plot_data.json').write_text(json.dumps({'evaluation':'development_test; not independent TEST',
        'model_provenance':provenance,'figure_sample_ids':figure_rows,'curves':curves,
        'selection':'First two distinct reference products per category and task; never selected by metric value'},indent=2),encoding='utf-8')


if __name__ == '__main__': main()
