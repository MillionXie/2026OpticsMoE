"""Selective native vision/language LoRA with real raw RGB+text gradients."""
import torch
from torch import nn

from .qwen_vl_language_adapt import inject_language_lora
from .qwen_vl_turbo_model import pack_hidden


SCHEMA='qwen_spatial_joint_lora_v1'


def install_joint_lora(qwen,cfg,state=None,train=False):
    from peft import LoraConfig,inject_adapter_in_model
    visual=qwen.model.visual;language=qwen.model.language_model
    if len(visual.blocks)!=24 or len(language.layers)!=28:raise ValueError('Original full Qwen24V/28L required')
    qwen.requires_grad_(False)
    wrapper=nn.ModuleList(list(visual.blocks[20:24]));rank=cfg['qwen_vision_lora_rank']
    inject_adapter_in_model(LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0,bias='none',
        target_modules=['qkv','proj','linear_fc1','linear_fc2'],init_lora_weights=True),wrapper)
    inject_language_lora(nn.ModuleList(list(language.layers[24:28])),cfg['qwen_language_lora_rank'])
    parameters={n:p for n,p in qwen.named_parameters() if p.requires_grad}
    if not parameters or any('.lora_' not in n for n in parameters):raise ValueError('Unexpected trainable native Qwen parameter')
    for n,p in parameters.items():
        allowed=any(n.startswith('model.visual.blocks.'+str(i)+'.') for i in range(20,24)) or any(n.startswith('model.language_model.layers.'+str(i)+'.') for i in range(24,28))
        if not allowed:raise ValueError('Qwen adapter outside approved final-four scopes')
        p.data=p.data.float()
    if state is not None:
        if set(state)!=set(parameters):raise ValueError('Native multimodal adapter scope differs')
        with torch.no_grad():
            for n,p in parameters.items():p.copy_(state[n].to(p))
    if not train:qwen.eval().requires_grad_(False)
    return qwen


def joint_adapter_state(qwen):
    return {n:p.detach().cpu().clone() for n,p in qwen.named_parameters() if '.lora_' in n}


def load_joint_adapters(qwen,state):
    actual={n:p for n,p in qwen.named_parameters() if '.lora_' in n}
    if set(actual)!=set(state):raise ValueError('Native multimodal adapter scope differs')
    with torch.no_grad():
        for n,p in actual.items():p.copy_(state[n].to(p))


class RawQwenHead(nn.Module):
    """Training wrapper; the image generator consumes actual raw Qwen states."""
    def __init__(self,head,qwen,processor,layout):
        super().__init__();self.head=head;self.raw_qwen=qwen;self.processor=processor;self.layout=layout
    @property
    def adapter(self):return self.head.adapter
    @property
    def spatial(self):return self.head.spatial
    @property
    def unet(self):return self.head.unet
    @property
    def vae(self):return self.head.vae
    @property
    def zero_spatial(self):return self.head.zero_spatial
    @zero_spatial.setter
    def zero_spatial(self,value):self.head.zero_spatial=value
    def raw_features(self,payload,ids,device):
        from .qwen_vl_hybrid_run import raw_hidden
        # Whitelist only reference RGB and supplied text. No paired labels,
        # teacher caption, product ID, target or mask reaches Qwen.
        inputs={'reference':payload['reference'][ids],
                'rows':[{'prompt':payload['rows'][i]['prompt']} for i in ids]}
        return pack_hidden([raw_hidden(self.raw_qwen,self.processor,inputs,j,self.layout,device) for j in range(len(ids))],device)
    def forward(self,*args,**kwargs):return self.head(*args,**kwargs)
    def forward_with_aux(self,*args,**kwargs):return self.head.forward_with_aux(*args,**kwargs)
    def semantic_conditions(self,*args,**kwargs):return self.head.semantic_conditions(*args,**kwargs)
    def spatial_inputs(self,*args,**kwargs):return self.head.spatial_inputs(*args,**kwargs)
