"""Load only three frozen visual stem tensors, never instantiate a language model."""
import hashlib,json,time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn


class VisualStem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.num_grid_per_side=int(config.num_position_embeddings**.5)
        k=(config.temporal_patch_size,config.patch_size,config.patch_size)
        self.patch_embed=nn.Module()
        self.patch_embed.proj=nn.Conv3d(config.in_channels,config.hidden_size,kernel_size=k,stride=k,bias=True)
        self.pos_embed=nn.Embedding(config.num_position_embeddings,config.hidden_size)

    def forward(self,pixel_values,image_grid_thw):
        # Exact frozen native stem before the old replacement's VisionStartBlock.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
        c=self.config;proj=self.patch_embed.proj
        x=pixel_values.reshape(-1,c.in_channels,c.temporal_patch_size,c.patch_size,c.patch_size)
        x=proj(x.to(proj.weight.dtype)).view(-1,c.hidden_size)
        pos=Qwen3VLVisionModel.fast_pos_embed_interpolate(self,image_grid_thw)
        return x+pos


@dataclass
class LoadedVision:
    model: VisualStem
    processor: object
    device: torch.device
    load_time_sec: float
    source_metadata: dict


def load_backbone(settings,device):
    from transformers import AutoImageProcessor
    from safetensors import safe_open
    from experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.modeling import resolve_cached_model_source
    started=time.perf_counter()
    source=Path(resolve_cached_model_source(settings.model_id,settings.cache_dir))
    if not source.is_dir():
        raise RuntimeError('Prepare the local Qwen checkpoint first; this loader reads only cached visual tensors')
    config=json.loads((source/'config.json').read_text())['vision_config']
    with torch.random.fork_rng(devices=[]):
        stem=VisualStem(SimpleNamespace(**config))
    required=set(stem.state_dict());selected={};origins={}
    # safe_open is lazy. Access only visual patch projection and positional embeddings.
    for file in sorted(source.glob('*.safetensors')):
        with safe_open(str(file),framework='pt',device='cpu') as f:
            for key in f.keys():
                for prefix in ('model.visual.','visual.'):
                    short=key.removeprefix(prefix)
                    if key.startswith(prefix) and short in required:
                        if short in selected:raise RuntimeError('Duplicate visual tensor: '+short)
                        selected[short]=f.get_tensor(key);origins[short]=dict(file=file.name,key=key)
    if set(selected)!=required:raise RuntimeError('Missing visual stem tensors: '+str(required-set(selected)))
    stem.load_state_dict(selected,strict=True)
    dtype={'bfloat16':torch.bfloat16,'float16':torch.float16,'float32':torch.float32}[settings.dtype]
    stem.to(device=device,dtype=dtype).requires_grad_(False).eval()
    processor=AutoImageProcessor.from_pretrained(str(source),local_files_only=True,use_fast=True,
        min_pixels=settings.processor_min_pixels,max_pixels=settings.processor_max_pixels)
    sha=hashlib.sha256()
    for n,v in sorted(stem.state_dict().items()):
        sha.update(n.encode());sha.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    metadata=dict(source=str(source),config_sha256=hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
                  frozen_stem_sha256=sha.hexdigest(),loaded_tensors=origins,
                  frozen_parameters=sum(p.numel() for p in stem.parameters()),language_parameters=0,
                  vision_transformer_parameters=0,merger_parameters=0)
    return LoadedVision(stem,processor,device,time.perf_counter()-started,metadata)
