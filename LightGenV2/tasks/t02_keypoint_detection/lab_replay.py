"""SDK-free exact three-CCD replay; exports original argmax indices, not PCK."""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from .build_lab_package import sha
from .lab_field_units import FieldUnits, STAGES
from .lab_runtime import load_model, phase_planes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve replay outputs')
    report=json.loads((a.session/'capture_report.json').read_text())
    if report['status']!='complete' or not report['sdk_released'] or report['ccd_count']!=3000:
        raise ValueError('Capture not complete/released')
    contract=report['contract']
    release=json.loads((a.release/'release.json').read_text())
    if sha(a.release/'release.json')!=contract['release_sha256']:raise ValueError('Release differs')
    model=load_model(a.release)
    planes=phase_planes(model)
    torch.set_num_threads(4)
    a.output.mkdir(parents=True)
    records=[]
    for item in release['fields']:
        key=item['key'];path=a.release/item['file']
        if sha(path)!=item['sha256']:raise ValueError('Cache identity differs')
        batch=torch.load(path,map_location='cpu',weights_only=False)
        measured={}
        for stage in STAGES:
            png=a.session/'ccd'/stage/(key+'.png')
            receipt=json.loads(png.with_suffix('.json').read_text())
            if sha(png)!=receipt['ccd_sha256']:raise ValueError('CCD differs')
            if receipt['phase_sha256']!=contract['phase_sha256'][stage]:raise ValueError('Phase differs')
            bmp=a.session/'amplitude'/stage/(key+'.bmp')
            if sha(bmp)!=receipt['amplitude_sha256']:raise ValueError('BMP differs')
            for up,digest in receipt['upstream_ccd_sha256'].items():
                if sha(a.session/'ccd'/up/(key+'.png'))!=digest:raise ValueError('Upstream differs')
            measured[stage]=torch.from_numpy(np.asarray(Image.open(png),dtype=np.float32).copy())[None]/255
        with torch.inference_mode(),FieldUnits(model,quantize=True,measured=measured,planes=planes):
            physical=model(batch)[0]
        sim=batch['simulation_heatmap']
        records.append({'key':key,'shape':list(physical.shape),'physical_indices':physical.flatten(2).argmax(-1).tolist(),
                        'reference_indices':sim.flatten(2).argmax(-1).tolist()})
        if len(records)%50==0:
            (a.output/'progress.json').write_text(json.dumps({'samples':len(records)}))
            print('REPLAY',len(records),flush=True)
    (a.output/'argmax_indices.json').write_text(json.dumps(records))
    (a.output/'report.json').write_text(json.dumps({'status':'complete','samples':len(records),'checkpoint_sha256':contract['checkpoint_sha256'],
        'release_sha256':contract['release_sha256'],'sdk_used':False,'weights_unchanged':True,
        'metric_pending':'Bind exact original TEST order and targets; these indices are not PCK'}))


if __name__=='__main__':main()
