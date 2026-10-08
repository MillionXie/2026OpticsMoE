"""Prepare an explicitly pinned bounded PT using immutable frozen-stem caches.

Never reuse upstream CCDs or legacy simulated heatmaps. No SDK is imported.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .build_lab_package import sha
from .lab_acquire import read, write
from .lab_field_units import FieldUnits, STAGES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-release',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing release')
    if sha(a.checkpoint)!=a.checkpoint_sha256:raise ValueError('Pinned checkpoint SHA mismatch')
    old=read(a.source_release/'release.json')
    settings=read(a.source_release/'settings.json')
    if sha(a.source_release/'settings.json')!=old['settings_sha256']:raise ValueError('Source settings SHA mismatch')
    if len(old['fields'])!=1000 or len({r['key'] for r in old['fields']})!=1000:raise ValueError('Require exact TEST1000 stem cache')
    fields=[]
    for row in old['fields']:
        path=(a.source_release/row['file']).resolve()
        if sha(path)!=row['sha256']:raise ValueError('Frozen stem cache SHA mismatch')
        fields.append(dict(row,file=str(path)))
    settings['physical_amplitude_mode']='tanh05_uint8'
    a.output.mkdir(parents=True)
    (a.output/'weights').mkdir()
    shutil.copyfile(a.checkpoint,a.output/'weights/best_checkpoint.pt')
    write(a.output/'settings.json',settings)
    release={'checkpoint_sha256':a.checkpoint_sha256,'physical_amplitude_mode':'tanh05_uint8',
             'amplitude_scale':1.,'encoding':'tanh(abs(E)/.5), round255 before propagation; no intensity unit restoration',
             'fields':fields,'stages':STAGES,'settings_sha256':sha(a.output/'settings.json'),
             'source_stem_release_sha256':sha(a.source_release/'release.json'),
             'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write(a.output/'release.json',release)
    import numpy as np
    import torch
    from .lab_runtime import load_model, phase_planes
    torch.set_num_threads(4)
    model=load_model(a.output)
    planes=phase_planes(model)
    (a.output/'phases').mkdir()
    for stage,value in planes.items():np.save(a.output/'phases'/(stage+'.npy'),value)
    release['phase_sha256']={s:sha(a.output/'phases'/(s+'.npy')) for s in STAGES}
    maximum={s:0. for s in STAGES}
    error=0.
    for index,row in enumerate(fields):
        batch=torch.load(row['file'],map_location='cpu',weights_only=False)
        with torch.inference_mode(),FieldUnits(model,scale=1.,quantize=True,planes=planes) as tap:
            expected=model(batch)[0]
        with torch.inference_mode(),FieldUnits(model,scale=1.,quantize=True,planes=planes,measured=tap.detectors):
            actual=model(batch)[0]
        error=max(error,float((expected-actual).abs().max()))
        if not torch.allclose(expected,actual,atol=2e-5,rtol=2e-5):raise ValueError('Ideal CCD bridge differs')
        for stage in STAGES:maximum[stage]=max(maximum[stage],float(tap.amplitudes[stage].max()))
        if (index+1)%50==0:print('BOUNDED_BRIDGE',index+1,flush=True)
    release['ideal_bridge']={'samples':1000,'heatmap_max_abs_error':error,'physical_maximum':maximum}
    release['status']='complete'
    write(a.output/'release.json',release)
    print('BOUNDED_RELEASE_COMPLETE',json.dumps(release['ideal_bridge']),flush=True)


if __name__=='__main__':main()
