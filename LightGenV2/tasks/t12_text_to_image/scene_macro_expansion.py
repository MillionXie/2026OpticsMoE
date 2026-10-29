"""Approved two-pair optical capacity ablation, with one router per modality.

The new pair starts with zero optical/electronic deltas. Placement preserves the
old pair's native DeepStack capture boundaries; no native TF is reintroduced.
All extra physical E/G cycles remain payable, including at initialization.
"""
import re
import torch
from torch import nn
from .qwen_vl_hybrid import OpticalMacro,IdentityBlock,MacroStageBlock
from .qwen_vl_postfree import QuietFusion,apply_postfree


def contract():
    return dict(kind='scene_two_EG_pairs_v1',vision_pairs=2,language_pairs=2,
        vision_stages=[6,12,14,16],language_stages=[4,14,18,22],
        one_router_per_modality=True,optical_cycles=10,retained_TF={'vision':6,'language':4})


def placement(qwen,macros):
    spec=contract()
    for name,blocks,start,stop,vision in [('vision',qwen.model.visual.blocks,6,18,True),
            ('language',qwen.model.language_model.layers,4,24,False)]:
        positions=spec[name+'_stages'];macro=macros[name]
        if macro.pairs!=2:raise ValueError('Two E/G pairs required')
        if any(not isinstance(blocks[i],(IdentityBlock,MacroStageBlock)) for i in range(start,len(blocks))):
            raise ValueError('Expansion must not replace or reintroduce an unapproved native TF')
        for i in range(start,stop):
            blocks[i]=MacroStageBlock(macro,positions.index(i),vision) if i in positions else IdentityBlock()


def build(qwen,config):
    choice=config.get('pair_expansion')
    if choice is not None and choice!=contract():raise ValueError('Unknown optical placement/cycle contract')
    pairs=2 if choice else 1
    if any(config.get(key,pairs)!=pairs for key in ['vision_pairs','language_pairs']):
        raise ValueError('Pair-count metadata differs from explicit optical contract')
    macros=apply_postfree(qwen,pairs,pairs,config['residual_width'])
    if choice:placement(qwen,macros)
    return macros


def parent_key(name):
    """Map only approved extra-pair tensors; never duplicate the router."""
    if '.paths.1.' in name:return name.replace('.paths.1.','.paths.0.')
    return re.sub(r'\.(codecs|electronic|fusions)\.[23]\.',r'.\1.1.',name)


def validate_state(new,old):
    if new['config'].get('pair_expansion')!=contract():raise ValueError('Missing explicit extra-cycle contract')
    if any(new['config'].get(key)!=2 for key in ['vision_pairs','language_pairs']):
        raise ValueError('Expanded pair-count metadata is inconsistent')
    expected=set(old['macros'])
    for name in old['macros']:
        if '.paths.0.' in name and '.core.router.' not in name:
            expected.add(name.replace('.paths.0.','.paths.1.'))
        for group in ['codecs','electronic','fusions']:
            if '.'+group+'.1.' in name:
                for i in [2,3]:expected.add(name.replace('.'+group+'.1.','.'+group+'.'+str(i)+'.'))
    if set(new['macros'])!=expected:raise ValueError('Unexpected layers or a second router in expanded state')
    if any(value.shape!=old['macros'][parent_key(name)].shape for name,value in new['macros'].items()):
        raise ValueError('Expanded tensor dimensions changed')


def expand(qwen,path_factory=None):
    old=qwen.model.hybrid
    if any(m.pairs!=1 for m in old.values()):raise ValueError('Only a one-pair parent may expand')
    state={n:v.detach().clone() for n,v in old.state_dict().items()}
    before=sum(p.numel() for p in old.parameters());device=next(old.parameters()).device
    kwargs={} if path_factory is None else dict(path_factory=path_factory)
    macros=nn.ModuleDict({name:OpticalMacro(name,m.width,2,m.electronic[0].down.out_features,**kwargs)
        for name,m in old.items()}).to(device)
    for macro in macros.values():
        for i,fusion in enumerate(macro.fusions):
            quiet=QuietFusion(.65,.6,.9,1e-6).to(device);quiet.load_state_dict(fusion.state_dict());macro.fusions[i]=quiet
    initialized=macros.state_dict()
    for name in initialized:
        source=parent_key(name)
        if source not in state or initialized[name].shape!=state[source].shape:raise ValueError('Expansion mapping changed: '+name)
        initialized[name]=state[source].clone()
        if re.search(r'\.codecs\.[23]\.output\.(weight|bias)$',name) or re.search(r'\.electronic\.[23]\.up\.(weight|bias)$',name):
            initialized[name].zero_()
    macros.load_state_dict(initialized,strict=True);qwen.model.hybrid=macros;placement(qwen,macros)
    extra=sum(p.numel() for p in macros.parameters())-before
    if extra<=0:raise ValueError('Expected the declared additional E/G capacity')
    return macros,extra
