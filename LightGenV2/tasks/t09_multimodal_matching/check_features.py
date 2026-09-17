"""Recompute per-device frozen features and quantify differences without training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .run import load_data,setseed
from .vision import frozen_features
from .prepare import digest,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    arrays=[];records=[];vocab=json.loads((a.data/'vocab.json').read_text())
    for run in a.runs:
        meta=json.loads((run/'metadata.json').read_text());original=json.loads((run/'shared_visual_frontend.json').read_text())
        gpu=int(meta['cuda_visible_devices']);torch.cuda.set_device(gpu);setseed(meta['config']['seed'])
        # Match the allocation order used by training.
        sets={s:load_data(a.data,s,vocab,'cuda') for s in ['train','val']}
        features={};actual={}
        for s,data in sets.items():
            features[s]=frozen_features(Path(meta['config']['vision_checkpoint']),data['images']).cpu().numpy()
            actual[s+'_feature_sha256']=digest(features[s].tobytes())
        records.append(dict(run=str(run),original=original,recomputed=actual,
                            original_hashes_reproduced=all(actual[s+'_feature_sha256']==original[s+'_feature_sha256'] for s in sets)))
        arrays.append(features);del sets;torch.cuda.empty_cache()
    comparison={}
    for s in ['train','val']:
        x,y=arrays[0][s],arrays[1][s]
        comparison[s]=dict(max_absolute=float(np.max(np.abs(x-y))),
                           relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-20)),
                           allclose=bool(np.allclose(x,y,rtol=1e-5,atol=1e-6)))
    save(a.out/'comparison.json',dict(records=records,comparison=comparison))
    assert records[0]['original_hashes_reproduced'], 'Canonical cache must reproduce the reference run exactly'
    np.savez_compressed(a.out/'feature_cache.npz',**arrays[0])
    save(a.out/'feature_cache.json',dict(checkpoint_sha256=records[0]['original']['checkpoint_sha256'],
         data_manifest_sha256=digest((a.data/'manifest.json').read_bytes()),
         cache_sha256=digest((a.out/'feature_cache.npz').read_bytes()),
         features=records[0]['recomputed'],source_run=records[0]['run']))
    print(json.dumps(dict(records=records,comparison=comparison),indent=2))


if __name__=='__main__':main()
