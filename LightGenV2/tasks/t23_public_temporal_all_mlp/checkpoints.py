"""Single-file checkpoints for the independent MLP inference model."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import importlib

import torch

from LightGenV2.tasks.t22_public_temporal_mlp.tools.run import save_checkpoint

public_model=importlib.import_module('LightGen-II-MLP.model')
ARCHITECTURE=public_model.ARCHITECTURE
LightGen2=public_model.LightGen2
ModelConfig=public_model.ModelConfig
load_lightgen2=public_model.load_lightgen2


BASE_SHA256='c6593a2b801d74698e4065cd63dcc3ba42fd24c8f571d4f8f3b19bdd8e41701e'


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def inherit_parent(parent,hidden=32):
    config=ModelConfig(global_hidden=hidden)
    model=LightGen2(config)
    target=model.state_dict()
    expected={name:value for name,value in parent.state_dict().items()
              if not name.startswith(('vision_global.','language_global.'))}
    for name,value in expected.items():
        if name not in target or value.shape!=target[name].shape:
            raise ValueError('Parent component dimensions differ: '+name)
        target[name].copy_(value)
    model.load_state_dict(target,strict=True)
    for name,value in expected.items():
        if not torch.equal(value,model.state_dict()[name]):
            raise ValueError('Inherited parent tensor differs: '+name)
    return model


def checkpoint(model,base_payload,training=None):
    state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()
           if name!='vision_frontend.position'}
    payload=dict(schema=3,architecture=ARCHITECTURE,config=asdict(model.cfg),model=state,
                 frontend={'position_table':base_payload['frontend']['position_table'].detach().cpu().clone()},
                 metadata=dict(base_payload['metadata']))
    if training is not None:payload['training']=training
    return payload


def export_checkpoint(source,destination):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    payload=torch.load(source,map_location='cpu',weights_only=True)
    payload.pop('training',None)
    save_checkpoint(destination,payload)
    model,_=load_lightgen2(destination)
    expected,_=load_lightgen2(source)
    for name,value in expected.state_dict().items():
        if not torch.equal(value,model.state_dict()[name]):
            raise ValueError('Exported inference weight differs: '+name)
    return sha256(destination)
