"""Selective LoRA on the original Qwen language layers 24:28.

Training may reuse the frozen prefix cache. Actual inference still executes
the complete original Qwen vision/language graph, with these adapters loaded.
"""
import torch
from torch import nn


TARGETS=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']


def inject_language_lora(module,rank):
    from peft import LoraConfig,inject_adapter_in_model
    module.requires_grad_(False)
    inject_adapter_in_model(LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0,
        target_modules=TARGETS,bias='none',init_lora_weights=True),module)
    names=[n for n,p in module.named_parameters() if p.requires_grad]
    if not names or any('.lora_' not in n for n in names):
        raise ValueError('Only the declared Qwen LoRA parameters may train')
    for p in module.parameters():
        if p.requires_grad:p.data=p.data.float()
    return names


class NativeLanguageTail(nn.Module):
    def __init__(self,layers,norm,contexts):
        super().__init__();self.layers=nn.ModuleList(layers);self.norm=norm
        self.contexts=contexts;self._device_context={}
        if len(self.layers)!=4:raise ValueError('Exactly original language layers24:28 required')

    def arguments(self,length,batch,device):
        key=length,str(device)
        if key not in self._device_context:
            source=self.contexts[length]
            if source.get('past_key_values') is not None or source.get('use_cache',False):
                raise ValueError('No autoregressive cache in image conditioning')
            values={}
            for name in ('position_embeddings','attention_mask','position_ids','cache_position'):
                value=source.get(name)
                if isinstance(value,tuple):values[name]=tuple(v.to(device) for v in value)
                elif isinstance(value,torch.Tensor):values[name]=value.to(device)
                else:values[name]=value
            self._device_context[key]=values
        result={}
        for name,value in self._device_context[key].items():
            if isinstance(value,tuple):result[name]=tuple(v.expand(batch,*v.shape[1:]) for v in value)
            elif isinstance(value,torch.Tensor) and name!='cache_position':
                if value.shape[0]!=1:raise ValueError('Expected actual B1 native argument cache')
                result[name]=value.expand(batch,*value.shape[1:])
            else:result[name]=value
        return result

    def forward(self,prefix,valid):
        lengths=valid.sum(1).detach().cpu().tolist()
        output=torch.zeros_like(prefix)
        for length in sorted(set(lengths)):
            ids=[i for i,n in enumerate(lengths) if n==length]
            hidden=prefix[ids,:length]
            kwargs=self.arguments(length,len(ids),prefix.device)
            for layer in self.layers:
                hidden=layer(hidden,**kwargs)
            output[ids,:length]=self.norm(hidden)
        return output

    def adapter_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if '.lora_' in n}

    def load_adapters(self,state):
        params={n:p for n,p in self.named_parameters() if '.lora_' in n}
        if set(params)!=set(state):raise ValueError('Native language adapter scope differs')
        with torch.no_grad():
            for n,p in params.items():p.copy_(state[n].to(p))


class AdaptiveLanguageHead(nn.Module):
    """Training-only prefix replay wrapper around the unchanged image head."""
    def __init__(self,head,tail):
        super().__init__();self.head=head;self.qwen_language_tail=tail
        self._last_input=None;self._last_valid=None;self._last_hidden=None

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

    def transformed(self,hidden,valid):
        if hidden is not self._last_input or valid is not self._last_valid:
            self._last_input,self._last_valid=hidden,valid
            self._last_hidden=self.qwen_language_tail(hidden,valid)
        return self._last_hidden

    def forward(self,hidden,valid,noise):
        return self.head(self.transformed(hidden,valid),valid,noise)

    def forward_with_aux(self,hidden,valid,noise,with_aux=True):
        return self.head.forward_with_aux(self.transformed(hidden,valid),valid,noise,with_aux=with_aux)

    def semantic_conditions(self,hidden,valid):
        return self.head.semantic_conditions(self.transformed(hidden,valid),valid)

    def spatial_inputs(self,hidden,valid):
        return self.head.spatial_inputs(self.transformed(hidden,valid),valid)


def install_in_full_qwen(qwen,state):
    """Load adapted weights into actual original layers, with no replay cache."""
    if state.get('schema')!='qwen_spatial_language_lora_v1':
        if 'qwen_language_lora' in state:raise ValueError('Unrecognized Qwen adapter schema')
        return qwen
    lm=qwen.model.language_model
    tail=NativeLanguageTail(list(lm.layers[24:28]),lm.norm,{})
    inject_language_lora(tail,state['config']['qwen_lora_rank'])
    tail.load_adapters(state['qwen_language_lora'])
    qwen.eval().requires_grad_(False)
    return qwen
