"""Fixed training references -> vision patch grid -> spatial phase decoder.

Qwen's pre-merger tokens are block ordered, not raster ordered. Only the
vision tower runs; no language model or autoregressive generation is used.
"""
import hashlib
import math
import random

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageOps

from .generator import StaticGenerator, install_lora


def select_references(training, seed):
    selected=[]
    for label in sorted({s.sku_index for s in training}):
        candidates=sorted((s for s in training if s.sku_index==label), key=lambda s:s.sample_id)
        selected.append(random.Random(f'{seed}:spatial-reference:{label}').choice(candidates))
    return selected


def reference_images(training, seed):
    selected=select_references(training,seed)
    images=[];records=[]
    for sample in selected:
        with Image.open(sample.image_path) as picture:
            images.append(ImageOps.fit(picture.convert('RGB'),(224,224),method=Image.Resampling.BICUBIC))
        records.append({**sample.manifest_record(),'sha256':hashlib.sha256(sample.image_path.read_bytes()).hexdigest()})
    return images,records


def unmerge_tokens(tokens, grid_thw, merge_size):
    """Undo processor's [H/m,W/m,m,m] order for *pre-merger* tokens."""
    grids=[];offset=0
    for t,h,w in grid_thw.tolist():
        if t!=1 or h%merge_size or w%merge_size:
            raise ValueError('Static image grids must have t=1 and be divisible by merge size')
        count=h*w
        grid=tokens[offset:offset+count].reshape(h//merge_size,w//merge_size,merge_size,merge_size,-1)
        grids.append(grid.permute(4,0,2,1,3).reshape(-1,h,w))
        offset+=count
    if offset!=len(tokens):raise ValueError('Token count differs from image_grid_thw')
    return torch.stack(grids)


class QVLoRA(nn.Module):
    """FP32 rank-r adapters for Q/V slices of Qwen vision's fused QKV."""
    def __init__(self, base, rank):
        super().__init__()
        if base.out_features!=3*base.in_features:raise ValueError('Expected fused square QKV')
        self.base=base.requires_grad_(False);self.width=base.in_features;self.scale=2.
        self.lora_a=nn.Parameter(torch.empty(2,rank,self.width,device=base.weight.device))
        self.lora_b=nn.Parameter(torch.zeros(2,self.width,rank,device=base.weight.device))
        for a in self.lora_a:nn.init.kaiming_uniform_(a,a=math.sqrt(5))

    def forward(self,x):
        base=self.base(x)
        with torch.autocast(x.device.type,enabled=False):
            q=(x.float() @ self.lora_a[0].T) @ self.lora_b[0].T
            v=(x.float() @ self.lora_a[1].T) @ self.lora_b[1].T
            delta=torch.cat((q,torch.zeros_like(q),v),dim=-1)*self.scale
        return base+delta.to(base.dtype)


class SpatialDecoder(nn.Module):
    """Dense convolutions mix neighbors before/after every pixel shuffle."""
    def __init__(self, outputs=8, global_plane=False):
        super().__init__();self.global_plane=global_plane
        self.grid=15 if global_plane else 14
        axis=torch.linspace(-1,1,self.grid);y,x=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('coordinates',torch.stack((x,y))[None])
        self.input=nn.Sequential(nn.Conv2d(514,64,3,padding=1),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GELU())
        channels=[64,32,24,16,8]+([8] if global_plane else [])
        blocks=[]
        for before,after in zip(channels,channels[1:]):
            blocks.extend((nn.Conv2d(before,after*4,3,padding=1),nn.PixelShuffle(2),nn.GELU(),
                           nn.Conv2d(after,after,3,padding=1),nn.GELU()))
        self.upsample=nn.Sequential(*blocks)
        self.head=nn.Conv2d(8,outputs,3,padding=1)
        nn.init.normal_(self.head.weight,std=.02);nn.init.zeros_(self.head.bias)

    def forward(self,features):
        features=F.adaptive_avg_pool2d(features,(self.grid,self.grid))
        z=self.input(torch.cat((features,self.coordinates.expand(len(features),-1,-1,-1)),dim=1))
        out=self.head(self.upsample(z))[0]
        return out[:,1:479,1:479] if self.global_plane else out.reshape(2,4,224,224)


class SpatialGenerator(StaticGenerator):
    def __init__(self, method, source, images, device, rank=8, seed=42):
        # Reuse compact checkpoint handling, not text encoder construction.
        nn.Module.__init__(self)
        self.method=method;self.source=str(source);self.lora_modules=[]
        self.generates_global=method=='qwen_vision_global_lora'
        self.pooled=method=='qwen_vision_pooled_lora'
        if method.startswith('qwen'):
            from transformers import AutoConfig, AutoImageProcessor
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            from safetensors import safe_open
            from pathlib import Path
            config=AutoConfig.from_pretrained(source,local_files_only=True).vision_config
            config._attn_implementation='eager'
            self.encoder=Qwen3VLVisionModel(config)
            state={}
            for file in sorted(Path(source).glob('*.safetensors')):
                with safe_open(file,framework='pt',device='cpu') as weights:
                    for name in weights.keys():
                        if name.startswith('model.visual.'):
                            state[name.removeprefix('model.visual.')]=weights.get_tensor(name)
            self.encoder.load_state_dict(state,strict=True);del state
            self.encoder.to(device=device,dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32).requires_grad_(False)
            # Native forward now returns raw vision tokens, without any language-space merger.
            self.encoder.merger=nn.Identity()
            self.encoder.deepstack_visual_indexes=[]
            self.encoder.deepstack_merger_list=nn.ModuleList()
            for name,module in list(self.encoder.named_modules()):
                if isinstance(module,nn.Linear) and name.endswith('.qkv'):
                    parent,child=name.rsplit('.',1)
                    setattr(self.encoder.get_submodule(parent),child,QVLoRA(module,rank))
                    self.lora_modules.append(name)
            processor=AutoImageProcessor.from_pretrained(source,local_files_only=True)
            encoded=processor(images=images,do_resize=False,return_tensors='pt')
            self.register_buffer('pixels',encoded['pixel_values'].to(device))
            self.register_buffer('grid_thw',encoded['image_grid_thw'].to(device))
            if not torch.all(self.grid_thw==torch.tensor([1,14,14],device=device)):
                raise ValueError(f'Expected fixed 224px / 16px patch grid: {self.grid_thw}')
        else:
            from transformers import CLIPVisionModel
            import numpy as np
            self.encoder=CLIPVisionModel.from_pretrained(source,local_files_only=True,attn_implementation='eager').to(device).requires_grad_(False)
            self.lora_modules=install_lora(self.encoder,rank)
            pixels=torch.stack([torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255 for im in images])
            mean=torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None]
            std=torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None]
            self.register_buffer('pixels',((pixels-mean)/std).to(device))
        if not self.lora_modules:raise RuntimeError('No trainable vision adapters installed')
        torch.manual_seed(seed+71)
        self.decoder=SpatialDecoder().to(device)
        self.register_buffer('anchor',torch.zeros(2,4,224,224,device=device))
        self.register_buffer('initial_reference',torch.zeros_like(self.anchor))
        if self.generates_global:
            torch.manual_seed(seed+72)
            self.global_decoder=SpatialDecoder(outputs=2,global_plane=True).to(device)
            self.register_buffer('global_reference',torch.zeros(2,478,478,device=device))
            self.register_buffer('global_reference_ready',torch.tensor(False,device=device))
        self.global_enabled=False;self.current_global=None
        self._trainable_names={n for n,p in self.named_parameters() if p.requires_grad}
        self.eval()
        with torch.no_grad():self.initial_reference.copy_(self.decode_raw())

    def spatial_features(self):
        self.encoder.eval()
        if self.method.startswith('qwen'):
            tokens=self.encoder(self.pixels.to(self.encoder.dtype),grid_thw=self.grid_thw)[0]
            maps=unmerge_tokens(tokens,self.grid_thw,self.encoder.spatial_merge_size).float()
        else:
            tokens=self.encoder(pixel_values=self.pixels,return_dict=True).last_hidden_state[:,1:]
            maps=tokens.transpose(1,2).reshape(len(tokens),-1,14,14).float()
        # Average references at each corresponding x/y; never average spatial positions here.
        maps=maps.mean(0,keepdim=True)
        b,c,h,w=maps.shape
        maps=F.adaptive_avg_pool1d(maps.permute(0,2,3,1).reshape(-1,1,c),512).reshape(b,h,w,512).permute(0,3,1,2)
        maps=F.layer_norm(maps.permute(0,2,3,1),(512,)).permute(0,3,1,2)
        if self.pooled:maps=maps.mean((-2,-1),keepdim=True).expand(-1,-1,h,w)
        return maps

    def decode_raw(self):return self.decoder(self.spatial_features())

    def set_global_enabled(self,enabled):
        self.global_enabled=bool(enabled)
        if self.generates_global and enabled and not bool(self.global_reference_ready):
            # Shared expert encoder has changed during expert-only training.
            # Anchor global at its unlock boundary to avoid an untrained phase jump.
            with torch.no_grad():self.global_reference.copy_(self.global_decoder(self.spatial_features()))
            self.global_reference_ready.fill_(True)

    def forward(self):
        features=self.spatial_features()
        if self.generates_global:
            self.current_global=(self.global_decoder(features)-self.global_reference if self.global_enabled
                                 else torch.zeros_like(self.global_reference))
        return self.decoder(features)-self.initial_reference
