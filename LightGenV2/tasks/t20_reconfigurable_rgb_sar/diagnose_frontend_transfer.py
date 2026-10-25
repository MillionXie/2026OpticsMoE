"""Read-only TRAIN/VAL transfer probe for frozen original and paired frontends.

Probes are diagnostics, not optical scores. Does not open TEST or save models.
"""
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from LightGenV2.tasks.t09_multimodal_matching.vision import VisionEncoder
from LightGenV2.tasks.t21_continual_learning_hardware.train_native import sha,put
from .train_unified_static import load_arrays
from .diagnose_unified import ridge_weights,class_scores


@torch.no_grad()
def checked_features(data,path,expected_sha256):
    """Pinned TRAIN-only same32128 structure; no temporary head in encoding."""
    if sha(path)!=expected_sha256:raise ValueError('Locked frontend identity differs')
    cp=torch.load(path,map_location='cpu',weights_only=False)
    if cp['TEST_access'] is not False or cp['config']['feature_parameters']!=32128:raise ValueError('Wrong frontend provenance')
    model=VisionEncoder();model.load_state_dict(cp['model'],strict=True);model.eval().requires_grad_(False)
    if sum(p.numel() for p in model.features.parameters())!=32128:raise ValueError('Changed frontend architecture')
    rows=[model(torch.from_numpy(data['images'][i:i+128]))[0].half() for i in range(0,len(data['labels']),128)]
    result=torch.cat(rows)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite fixed frontend')
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frontends',type=Path,nargs='+',required=True);a=p.parse_args();torch.set_num_threads(2)
    a.out.mkdir(parents=True,exist_ok=False);tr=load_arrays(a.data,'train');va=load_arrays(a.data,'validation');rows=[];started=time.time()
    for path in a.frontends:
        digest=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['TEST_access'] is False and cp['config']['feature_parameters']==32128
        m=VisionEncoder();m.load_state_dict(cp['model'],strict=True);m.eval().requires_grad_(False)
        x=np.concatenate([m(torch.from_numpy(v))[0].numpy() for v in np.array_split(tr['images'],48)])
        z=np.concatenate([m(torch.from_numpy(v))[0].numpy() for v in np.array_split(va['images'],16)])
        probes=[]
        for normalization in ('raw','unit'):
            tx=x if normalization=='raw' else x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            vx=z if normalization=='raw' else z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
            for domains in ([0],[1],[0,1]):
                mask=np.isin(tr['domains'],domains);w=ridge_weights(tx[mask],tr['labels'][mask],.001)
                r=class_scores(vx@w.T,va['labels'],va['domains'])
                probes.append(dict(normalization=normalization,training_domains=domains,validation={d:s['accuracy'] for d,s in r.items()}))
        assert sha(path)==digest;rows.append(dict(path=str(path),SHA256=digest,epoch=cp['epoch'],probes=probes))
        put(a.out/'progress.json',dict(completed_frontends=len(rows),updated=time.time()))
    put(a.out/'report.json',dict(rows=rows,TRAIN=6000,validation=2000,TEST_loaded=False,SDK=False,model_mutations=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seconds=time.time()-started))


if __name__=='__main__':main()
