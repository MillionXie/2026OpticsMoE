"""Qwen instruction caches and paired datasets for style and turntable tasks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .dataset import read_manifest
from .decoder_optical_generation import DecoderOpticalGenerator
from .feature_cache import _encode_caption_rows


STYLE_SPECS = (
    ("matte_black_nylon", ([.03,.035,.04],[.35,.37,.40]), ("make the backpack matte black nylon", "turn it into a black waterproof commuter backpack", "use a dark black technical-fabric style")),
    ("tan_canvas", ([.22,.12,.05],[.82,.62,.34]), ("make the backpack tan canvas", "use a light brown canvas travel style", "change it to a warm khaki fabric backpack")),
    ("cobalt_outdoor", ([.02,.08,.30],[.18,.58,.95]), ("make the backpack cobalt blue for outdoor use", "use a vivid blue waterproof hiking style", "change it to a deep blue technical backpack")),
    ("safety_orange", ([.35,.06,.01],[1.0,.48,.08]), ("make the backpack high visibility orange", "use a bright orange outdoor safety style", "change it to a vivid orange hiking backpack")),
    ("olive_utility", ([.10,.13,.035],[.48,.55,.20]), ("make the backpack olive green utility style", "use a muted green outdoor fabric", "change it to an olive hiking backpack")),
    ("burgundy_leather", ([.18,.025,.035],[.66,.18,.22]), ("make the backpack burgundy leather", "use a dark red premium leather style", "change it to a wine red urban backpack")),
)
VIEW_SPECS = (
    (-1, ("rotate the chair one step counterclockwise", "show the immediately previous view on the left", "turn the object about 36 degrees to the left")),
    (1, ("rotate the chair one step clockwise", "show the immediately next view on the right", "turn the object about 36 degrees to the right")),
)


def prompt_rows(task: str) -> list[dict[str, Any]]:
    rows=[]
    if task=="style":
        for index,(name,_,prompts) in enumerate(STYLE_SPECS):
            rows.extend({"condition":index,"name":name,"variant":variant,"prompt":prompt} for variant,prompt in enumerate(prompts))
    elif task=="view":
        for index,(shift,prompts) in enumerate(VIEW_SPECS):
            rows.extend({"condition":index,"shift":shift,"variant":variant,"prompt":prompt} for variant,prompt in enumerate(prompts))
    else: raise ValueError(task)
    return rows


def build_instruction_cache(task:str,output:Path,qwen_checkpoint:Path,device:torch.device,force:bool=False)->dict[str,Any]:
    if output.exists() and not force: return torch.load(output,map_location="cpu",weights_only=False)["meta"]
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    rows=prompt_rows(task); processor=AutoProcessor.from_pretrained(qwen_checkpoint,local_files_only=True)
    model=Qwen3VLForConditionalGeneration.from_pretrained(qwen_checkpoint,local_files_only=True,torch_dtype=torch.bfloat16 if device.type=="cuda" else torch.float32,attn_implementation="sdpa").to(device).eval().requires_grad_(False)
    features,_=_encode_caption_rows(model,processor,[x["prompt"] for x in rows],device,16,f"{task}-instructions")
    payload={"meta":{"schema_version":1,"task":task,"qwen":str(qwen_checkpoint.resolve()),"text_dim":int(features.shape[1])},"rows":rows,"text":features.to(torch.bfloat16).cpu()}
    output.parent.mkdir(parents=True,exist_ok=True); tmp=output.with_suffix(".tmp"); torch.save(payload,tmp); tmp.replace(output)
    del model,processor
    if device.type=="cuda": torch.cuda.empty_cache()
    return payload["meta"]


def apply_backpack_style(reference:torch.Tensor,condition:torch.Tensor)->torch.Tensor:
    rgb=reference.float().add(1).mul(.5); luminance=.299*rgb[:,:1]+.587*rgb[:,1:2]+.114*rgb[:,2:3]
    dark=reference.new_tensor([x[1][0] for x in STYLE_SPECS])[condition][:,:,None,None]
    light=reference.new_tensor([x[1][1] for x in STYLE_SPECS])[condition][:,:,None,None]
    styled=dark+(light-dark)*luminance.pow(.82); styled=.94*styled+.06*rgb
    mask=DecoderOpticalGenerator.foreground_mask(reference); return (rgb*(1-mask)+styled*mask).mul(2).sub(1).clamp(-1,1)


class ImageManifestDataset(Dataset[dict[str,Any]]):
    def __init__(self,data_dir:Path,split:str,image_size:int)->None:
        self.rows=read_manifest(data_dir/f"{split}.jsonl"); self.image_size=int(image_size)
    def __len__(self)->int:return len(self.rows)
    def __getitem__(self,index:int)->dict[str,Any]:
        row=self.rows[index]
        with Image.open(row.image_path) as handle:
            image=ImageOps.fit(ImageOps.exif_transpose(handle).convert("RGB"),(self.image_size,self.image_size),method=Image.Resampling.LANCZOS)
        value=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().div(127.5).sub(1)
        return {"sample_id":row.sample_id,"image":value}


class BackpackStyleDataset(Dataset[dict[str,Any]]):
    def __init__(self,data_dir:Path,split:str,image_size:int,cache:Path)->None:
        self.base=ImageManifestDataset(data_dir,split,image_size); payload=torch.load(cache,map_location="cpu",weights_only=False)
        self.prompts=payload["rows"]; self.text=payload["text"].float()
    def __len__(self)->int: return len(self.base)*len(self.prompts)
    def __getitem__(self,index:int)->dict[str,Any]:
        prompt_index=index%len(self.prompts); base=self.base[index//len(self.prompts)]; meta=self.prompts[prompt_index]
        reference=base["image"]; condition=int(meta["condition"])
        return {"sample_id":base["sample_id"],"reference":reference,"target":apply_backpack_style(reference[None],torch.tensor([condition]))[0],"text":self.text[prompt_index],"condition":condition,"prompt":meta["prompt"]}


class TurntableViewDataset(Dataset[dict[str,Any]]):
    """Canonical product view to its immediately adjacent left/right render."""
    def __init__(self,data_dir:Path,split:str,image_size:int,cache:Path)->None:
        base=ImageManifestDataset(data_dir,split,image_size); payload=torch.load(cache,map_location="cpu",weights_only=False)
        self.prompts=payload["rows"]; self.text=payload["text"].float(); grouped=defaultdict(list)
        for index,row in enumerate(base.rows): grouped[row.sequence_id].append((int(row.sample_id.rsplit("-",1)[-1]),index))
        self.base=base; self.groups=[]
        for sequence,items in grouped.items():
            ordered=[index for _,index in sorted(items)]
            self.groups.append((sequence,0,ordered))
    def __len__(self)->int: return len(self.groups)*len(self.prompts)
    def __getitem__(self,index:int)->dict[str,Any]:
        prompt_index=index%len(self.prompts); sequence,position,ordered=self.groups[index//len(self.prompts)]; meta=self.prompts[prompt_index]
        source=self.base[ordered[position]]; target=self.base[ordered[(position+int(meta["shift"]))%len(ordered)]]
        return {"sample_id":f"{sequence}:{position}:{meta['shift']}","reference":source["image"],"target":target["image"],"text":self.text[prompt_index],"condition":int(meta["condition"]),"prompt":meta["prompt"]}


__all__=["BackpackStyleDataset","TurntableViewDataset","apply_backpack_style","build_instruction_cache","prompt_rows"]
