"""Lossless native-resolution image inventory and full-precision per-item metrics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def export(project,run):
    report=json.loads((run/'report.json').read_text(encoding='utf-8'))
    inputs=torch.load(project/'assets/pilot_inputs.pt',map_location='cpu',weights_only=False)
    outputs=torch.load(run/'outputs.pt',map_location='cpu',weights_only=False)
    if len(inputs['metadata'])!=report['sample_count']:raise ValueError('Sample metadata mismatch')
    rows=[]
    for i,metadata in enumerate(inputs['metadata']):
        sid=f'pilot_{i:03d}'
        row=dict(sample_id=sid,source_sample_id=metadata['sample_id'],prompt=metadata['prompt'],
                 category=metadata['category'],mode=metadata['mode'],scope=report['scope'],
                 checkpoint_sha256=report['contract']['checkpoint_sha256'])
        target=inputs['target'][i].float().add(1).div(2).clamp(0,1)
        for label,key in [('physical','actual'),('simulation','simulation')]:
            value=outputs[key][i].float().add(1).div(2).clamp(0,1)
            mse=float((value-target).square().mean());row[label+'_mse_0_1']=mse
            row[label+'_psnr_db']=10*math.log10(1/max(mse,1e-15))
            row[label+'_ssim']=float(structural_similarity(target.permute(1,2,0).numpy(),
                value.permute(1,2,0).numpy(),channel_axis=-1,data_range=1.))
            row[label+'_mae_0_1']=float((value-target).abs().mean())
        for label in ('reference','target','simulation','physical'):
            path=run/(sid+'_'+label+'.png')
            with Image.open(path) as image:width,height=image.size
            row[label+'_image']=path.name;row[label+'_width']=width;row[label+'_height']=height
            row[label+'_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(row)
    (run/'sample_metrics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    with (run/'sample_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (run/'FIGURE_README.md').write_text(
        '# Figure data\n\nAll rows retained, including low-scoring images. Join using sample_id; '
        'source_sample_id and full prompt identify each edit. CSV opens in Excel; JSON preserves full precision. '
        'MSE/MAE use RGB [0,1], PSNR range1, SSIM skimage RGB channel_axis=-1. '
        'FID/KID are dataset-level, not per-image metrics.\n\n'
        'PNG files are lossless native decoder outputs (256x256), not contact-sheet crops or JPEGs. '
        'They are not higher-resolution reconstructions; upscaling cannot recover missing detail. '
        'outputs.pt preserves unrounded FP32 predictions. Pilot results must not be reported as full-test metrics.\n',encoding='utf-8')
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();print(json.dumps(dict(exported=len(export(a.project,a.run))),flush=True))
