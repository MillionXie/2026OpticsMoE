"""Server-side export of fixed T16 encoded inputs; no optical inference or SDK.

All negative pairs are constructed on the full original split before selection.
Visual features are computed only for selected images, using original frozen
CNN and float16 cache conversion. No image/model correctness influences sampling.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_lifelong_moe import load_dataset
from LightGenV2.tasks.t16_zero_phase_ccd_lifelong.frozen_vision import FrozenVisionFeatures, visual_tiles
from .subset import make_manifest

TASKS=('eurosat','clevr','speech_binary','physical_binary_raw')
VISION_SHA='d20273e1828fec4e5a7a5545fd892a722f3909f7205a791e0e682b437526aa11'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--protocols',type=Path,required=True,help='Private JSON task->original protocol paths, plus vision')
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--split',choices=('train','val','test'),required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    paths={k:Path(v) for k,v in json.loads(a.protocols.read_text()).items()}
    identities=json.loads((Path(__file__).parent/'source_identity.json').read_text())
    expected=identities['models'][3]['config']['protocol_sha256'][a.task]
    if sha(paths[a.task])!=expected or sha(paths['vision'])!=VISION_SHA:
        raise ValueError('Original protocol/vision identity mismatch')
    a.out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cpu')
    data=load_dataset(paths,a.task,a.split,vision_checkpoint=None,device=device)
    manifest=make_manifest(data,a.task,split=a.split,source_protocol_sha256=expected)
    ix=np.asarray(manifest['source_indices'])
    # Build original fields first; the original SAR/text quadrant remains intact.
    fields=data.get_batch(ix,device)
    if a.task in ('eurosat','clevr'):
        raw=(data.data.rgb[ix] if a.task=='eurosat' else data.images[data.image_index[ix]])
        front=FrozenVisionFeatures(paths['vision'],raw,device,batch_size=256)
        tiles=visual_tiles(raw,front.select(np.arange(len(ix)),device),device)
        fields=torch.cat((torch.cat((tiles[:,0],tiles[:,1]),-1),
                          torch.cat((tiles[:,2],fields[:,112:,112:]),-1)),-2)
    if fields.shape!=(1000,224,224) or not torch.isfinite(fields).all() or (fields<0).any():
        raise ValueError('Invalid encoded deployment fields')
    power=fields.square().sum((-2,-1))
    torch.testing.assert_close(power,torch.ones_like(power),atol=3e-6,rtol=3e-6)
    np.savez_compressed(a.out/'inputs.npz',amplitudes=fields.numpy(),labels=np.asarray(data.labels)[ix],source_indices=ix)
    manifest.update(vision_checkpoint_sha256=VISION_SHA,inputs_sha256=sha(a.out/'inputs.npz'),
                    scope='encoded original T16 input, not measured camera data',
                    hardware_admitted=False)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('task','split','class_counts','strata','identity_sha256','inputs_sha256')}))


if __name__=='__main__':main()
