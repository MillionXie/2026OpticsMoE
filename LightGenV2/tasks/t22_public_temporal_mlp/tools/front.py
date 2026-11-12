"""One-time targeted extraction of the small frozen Qwen front.

Only safetensors headers and the patch/position tensors are read. The complete
Qwen network is never constructed. Runtime model.py needs only PyTorch.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def position_28(table):
    """Original endpoint-aligned bilinear interpolation, official group order."""
    side=int(table.shape[0]**.5)
    if side*side!=table.shape[0]:raise ValueError('Position table must be square')
    idx=torch.linspace(0,side-1,28)
    lo=idx.int();hi=(lo+1).clip(max=side-1);delta=idx-lo
    indices=torch.stack([(lo[:,None]*side+lo[None,:]).flatten(),
        (lo[:,None]*side+hi[None,:]).flatten(),(hi[:,None]*side+lo[None,:]).flatten(),
        (hi[:,None]*side+hi[None,:]).flatten()]).long()
    weights=torch.stack([((1-delta[:,None])*(1-delta[None,:])).flatten(),
        ((1-delta[:,None])*delta[None,:]).flatten(),(delta[:,None]*(1-delta[None,:])).flatten(),
        (delta[:,None]*delta[None,:]).flatten()]).to(table.dtype)
    # Preserve the original bf16 rounding of each product and the left-to-right sum.
    parts=table[indices]*weights[:,:,None]
    pos=parts[0]+parts[1]+parts[2]+parts[3]
    return pos.reshape(1,14,2,14,2,-1).permute(0,1,3,2,4,5).flatten(0,4).contiguous()


def extract(qwen_dir,language_cache):
    from safetensors import safe_open
    folder=Path(qwen_dir)
    wanted=['visual.patch_embed.proj.weight','visual.patch_embed.proj.bias','visual.pos_embed.weight']
    found={}
    index=folder/'model.safetensors.index.json'
    files=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {p.name for p in folder.glob('*.safetensors')}
    for filename in sorted(files):
        with safe_open(folder/filename,framework='pt',device='cpu') as f:
            for key in f.keys():
                for ending in wanted:
                    if key.endswith(ending):
                        if ending in found:raise ValueError('Ambiguous front tensor '+ending)
                        found[ending]=f.get_tensor(key)
    if set(found)!=set(wanted):raise ValueError('Missing patch/position tensors')
    lang=torch.load(language_cache,map_location='cpu',weights_only=False)
    embedding=lang['language_tokens'][0].clone()
    if not bool(lang['attention_mask'].all()):raise ValueError('Fixed prompt must have no padded tokens')
    source=lang.get('qwen_source_identity',{})
    revision=source.get('checkpoint_revision')
    if revision and folder.name!=revision:raise ValueError('Front and language checkpoint revisions differ')
    vision_config=json.loads((folder/'config.json').read_text())['vision_config']
    if (vision_config['patch_size'],vision_config['temporal_patch_size'],vision_config['hidden_size'])!=(16,2,1024):
        raise ValueError('Wrong Qwen front geometry')
    pre=json.loads((folder/'preprocessor_config.json').read_text())
    if pre['image_mean']!=[.5,.5,.5] or pre['image_std']!=[.5,.5,.5]:raise ValueError('Wrong pixel normalization')
    return {'front':{'patch_weight':found[wanted[0]],'patch_bias':found[wanted[1]],
                     'position':position_28(found[wanted[2]]),'ready':torch.tensor(True)},
            'prompt_embedding':embedding,'source':{'qwen_revision':revision,'prompt':lang['prompt'],
            'language_cache_sha256':digest(language_cache),'preprocessor':pre,
            'patch_and_position_shapes':{k:list(v.shape) for k,v in found.items()}}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qwen-dir',type=Path,required=True)
    p.add_argument('--language-cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    payload=extract(a.qwen_dir,a.language_cache)
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,a.output)
    a.output.with_suffix('.json').write_text(json.dumps({'sha256':digest(a.output),**payload['source']},indent=2)+'\n','utf-8')
    print('Small standalone front saved:',a.output,a.output.stat().st_size,'bytes')


if __name__=='__main__':main()
