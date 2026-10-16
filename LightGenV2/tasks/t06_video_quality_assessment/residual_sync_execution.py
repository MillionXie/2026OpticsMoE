"""Execution-only candidates; original learned modules, operators and dtypes."""
from __future__ import annotations

import torch
from .qwen_host_metadata_execution import HostMetadataQwenModel


class IndexedQwenText(torch.nn.Module):
    def __init__(self,wrapped,host_ids,host_attention,video_id):
        super().__init__();self.wrapped=wrapped
        self.host_ids=host_ids;self.host_attention=host_attention;self.video_id=video_id
        self.training=False

    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('wrapped'),name)

    def _deepstack_process(self,hidden_states,visual_pos_masks,visual_embeds):
        # Recompute integer indices HERE, at the original deepstack call point.
        # No position/feature/hidden result cache or moved learned computation.
        if tuple(visual_pos_masks.shape)!=tuple(self.host_ids.shape):raise ValueError('Mask shape changed')
        index=torch.nonzero((self.host_ids==self.video_id).reshape(-1),as_tuple=False).flatten()
        if len(index)!=visual_embeds.shape[0]:raise ValueError('Visual feature count mismatch')
        index=index.to(hidden_states.device,non_blocking=True)
        visual_embeds=visual_embeds.to(hidden_states.device,hidden_states.dtype)
        flat=hidden_states.reshape(-1,hidden_states.shape[-1])
        local_this=flat.index_select(0,index).clone()+visual_embeds
        flat.index_copy_(0,index,local_this)
        return hidden_states

    def forward(self,**kwargs):
        # Native SDPA already elides this all-one mask. Only its integer test
        # moves to CPU, with the same no-padding attention semantics.
        if kwargs.get('attention_mask') is not None and bool(self.host_attention.all()):
            kwargs=dict(kwargs,attention_mask=None)
        return type(self.wrapped).forward(self,**kwargs)


class IndexedHostQwen(HostMetadataQwenModel):
    def __init__(self,wrapped,cpu_inputs,resident_inputs):
        super().__init__(wrapped,cpu_inputs,resident_inputs)
        if bool((self.host_metadata['input_ids']==wrapped.config.image_token_id).any()):
            raise ValueError('Video-only candidate')
        # Untimed input-validation only, not precomputed model/position outputs.
        for key in ['input_ids','attention_mask']:
            if not torch.equal(self.host_metadata[key],resident_inputs[key].detach().cpu()):raise ValueError('Host/device metadata differs')
        self.language_model=IndexedQwenText(wrapped.language_model,self.host_metadata['input_ids'],self.host_metadata['attention_mask'],wrapped.config.video_token_id)
        if {id(p) for p in self.parameters()}!={id(p) for p in wrapped.parameters()}:raise RuntimeError('Parameter identity changed')

    def get_placeholder_mask(self,input_ids,inputs_embeds,image_features=None,video_features=None):
        if input_ids is not self.resident_inputs['input_ids'] or image_features is not None:raise ValueError('Unsupported input')
        host=self.host_metadata['input_ids']
        count=int((host==self.wrapped.config.video_token_id).sum())
        if video_features is not None and count*inputs_embeds.shape[-1]!=video_features.numel():raise ValueError('Video token count mismatch')
        image=(input_ids==self.wrapped.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        video=(input_ids==self.wrapped.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        return image,video

    def get_rope_index(self,input_ids=None,image_grid_thw=None,video_grid_thw=None,attention_mask=None):
        if image_grid_thw is not None or input_ids is not self.resident_inputs['input_ids'] or video_grid_thw is not self.host_metadata['video_grid_thw']:
            raise ValueError('Unsupported metadata')
        positions,deltas=self.wrapped.get_rope_index(self.host_metadata['input_ids'],None,self.host_metadata['video_grid_thw'],attention_mask=self.host_metadata['attention_mask'])
        return positions.to(input_ids.device,non_blocking=True),deltas.to(input_ids.device,non_blocking=True)


def clip_explicit_fp16_forward(model,head,scores):
    if model.visual.conv1.weight.dtype!=torch.float16 or head.weight.dtype!=torch.float32:
        raise ValueError('Preserve original FP16 backbone/FP32 head')
    def compute(inputs):
        # Official encode_image explicitly casts to its original FP16 weights.
        # No autocast policy is needed if all real operator dtypes stay equal.
        with torch.autocast('cuda',enabled=False):
            feature=model.encode_image(inputs).float()
        feature=torch.nn.functional.normalize(feature,dim=-1).reshape(-1,2048).half().float()
        return (feature@head.weight.t()).softmax(-1)@scores
    return compute
