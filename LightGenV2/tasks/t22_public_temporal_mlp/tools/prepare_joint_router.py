"""Expand the fusion router to all tokens and preserve the fixed-prompt function."""
from __future__ import annotations

import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

if __package__:
    from . import _paths
else:
    import _paths

if __package__ and __package__ != 'tools':
    from .data import sha256
    from ..model import LightGenTemporal,load_lightgen
    from .run import save_checkpoint
else:
    from data import sha256
    from model import LightGenTemporal,load_lightgen
    from run import save_checkpoint


def expand_router(parent,seed):
    if parent.cfg.fusion_router_input!='visual':
        raise ValueError('Expansion expects the historical visual-only router')
    torch.manual_seed(seed)
    model=LightGenTemporal(replace(parent.cfg,fusion_router_input='all_tokens'))
    state=parent.state_dict()
    current=model.state_dict()
    for name,value in state.items():
        if name not in ('video_router.fc1.weight','video_router.fc1.bias'):
            current[name].copy_(value)
    model.load_state_dict(current,strict=True)
    width=parent.cfg.width
    with torch.no_grad():
        weight=model.video_router.fc1.weight
        visual=parent.video_router.fc1.weight
        weight[:,:4*width].copy_(visual)
        torch.nn.init.normal_(weight[:,4*width:],std=float(visual.std())*.25)
        prompt=model.language_adapter(model.prompt_embedding.float())
        prompt=prompt+model.sequence_position[0,4:4+len(prompt)]
        normalized=model.video_router.norm(prompt).flatten()
        model.video_router.fc1.bias.copy_(parent.video_router.fc1.bias-
                                         F.linear(normalized,weight[:,4*width:]))
    return model.eval()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=277)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(4)
    parent,metadata=load_lightgen(args.source)
    model=expand_router(parent,args.seed)
    state=parent.state_dict()
    changed=[name for name,value in model.state_dict().items()
             if value.shape!=state[name].shape or not torch.equal(value,state[name])]
    if changed!=['video_router.fc1.weight','video_router.fc1.bias']:
        raise ValueError('Unexpected changes during router expansion: '+str(changed))
    args.out.mkdir(parents=True)
    root=_paths.ROOT
    record={'role':'JOINT_ROUTER_INITIALIZATION','dataset':metadata['dataset'],
            'parent_SHA256':sha256(args.source),'seed':args.seed,'changed_tensors':changed,
            'fusion_router_shape':[4+model.cfg.prompt_length,model.cfg.width],
            'text_weight_norm':float(model.video_router.fc1.weight[:,4*model.cfg.width:].detach().norm()),
            'fixed_prompt_bias_compensation':True,
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'source_SHA256':{name:sha256(_paths.source_path(name)) for name in ['model.py','prepare_joint_router.py']}}
    save_checkpoint(args.out/'initialization.pt',model.checkpoint(**record))
    (args.out/'preparation.json').write_text(json.dumps(record,indent=2)+'\n','utf-8')
    (args.out/'model_config.json').write_text(json.dumps(asdict(model.cfg),indent=2)+'\n','utf-8')
    print(json.dumps({'out':str(args.out),'router_shape':record['fusion_router_shape'],
                      'text_weight_norm':record['text_weight_norm']}))


if __name__=='__main__':main()
