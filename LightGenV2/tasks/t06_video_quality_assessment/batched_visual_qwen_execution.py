"""Group equal-length independent vision chunks, same native SDPA and weights.

Not adopted without strict native hidden/logit/MOS and actual dtype validation.
No Graph, feature cache, mask change, or concatenation of attention contexts.
"""
from __future__ import annotations
import torch
from .residual_sync_execution import IndexedHostQwen


class Delegate(torch.nn.Module):
    def __init__(self,wrapped):
        super().__init__();self.wrapped=wrapped;self.training=False
    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('wrapped'),name)


class EqualChunkAttention(Delegate):
    def forward(self,hidden_states,cu_seqlens,rotary_pos_emb=None,position_embeddings=None,**kwargs):
        if self.config._attn_implementation!='sdpa' or cu_seqlens.device.type!='cpu':raise ValueError('Audited native SDPA with host grid only')
        lengths=(cu_seqlens[1:]-cu_seqlens[:-1]).tolist()
        if not lengths or len(set(lengths))!=1:
            return self.wrapped(hidden_states,cu_seqlens,rotary_pos_emb,position_embeddings,**kwargs)
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision,ALL_ATTENTION_FUNCTIONS
        seq_length=hidden_states.shape[0];chunks=len(lengths);length=lengths[0]
        if seq_length!=chunks*length:raise ValueError('Invalid independent chunk shape')
        q,k,v=self.qkv(hidden_states).reshape(seq_length,3,self.num_heads,-1).permute(1,0,2,3).unbind(0)
        cos,sin=position_embeddings
        q,k=apply_rotary_pos_emb_vision(q,k,cos,sin)
        # Each former length chunk becomes an independent SDPA batch member.
        # Token dimension is NOT concatenated across frames/videos.
        def batch(t):return t.transpose(0,1).reshape(self.num_heads,chunks,length,-1).permute(1,0,2,3)
        out,_=ALL_ATTENTION_FUNCTIONS['sdpa'](self.wrapped,batch(q),batch(k),batch(v),attention_mask=None,
            scaling=self.scaling,dropout=0.0,is_causal=False,**kwargs)
        return self.proj(out.reshape(seq_length,-1).contiguous())


class VisionBlock(Delegate):
    def __init__(self,wrapped):super().__init__(wrapped);self.attn=EqualChunkAttention(wrapped.attn)
    def forward(self,*args,**kwargs):return type(self.wrapped).forward(self,*args,**kwargs)


class Vision(Delegate):
    def __init__(self,wrapped):super().__init__(wrapped);self.blocks=torch.nn.ModuleList([VisionBlock(x) for x in wrapped.blocks])
    def forward(self,*args,**kwargs):return type(self.wrapped).forward(self,*args,**kwargs)


class BatchedVisualQwen(IndexedHostQwen):
    def __init__(self,wrapped,cpu_inputs,resident_inputs):
        super().__init__(wrapped,cpu_inputs,resident_inputs);self.visual=Vision(wrapped.visual)
        if {id(p) for p in self.parameters()}!={id(p) for p in wrapped.parameters()}:raise RuntimeError('Parameter identity changed')
    def get_image_features(self,*args,**kwargs):return type(self.wrapped).get_image_features(self,*args,**kwargs)
    def get_video_features(self,*args,**kwargs):return type(self.wrapped).get_video_features(self,*args,**kwargs)


def normalized_learned_work(operators):
    """Same real dtype and work shape; only independent SDPA batch grouping differs."""
    import json
    from collections import Counter
    result=Counter()
    for item in operators:
        item=dict(item,tensors=[dict(t) for t in item['tensors']]);weight=1
        if 'scaled_dot_product' in item['op']:
            weight=item['tensors'][0]['shape'][0]
            for t in item['tensors'][:3]:t['shape']=t['shape'][1:]
        result[json.dumps(item,sort_keys=True)]+=weight
    return dict(result)
