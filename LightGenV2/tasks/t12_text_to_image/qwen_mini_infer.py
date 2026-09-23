"""Arbitrary-prompt inference for the sub-50M Qwen-mini optical editors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .feature_cache import _qwen_prompts
from .half_qwen import load_half_qwen_text_encoder
from .qwen_mini_small import QwenMiniConfig, QwenMiniTextEncoder
from .small_fullframe import SmallEditorConfig, SmallFullFrameEditor


def _load_rgb(path: Path, size: int, device: torch.device) -> torch.Tensor:
    with Image.open(path) as handle:
        image=ImageOps.fit(ImageOps.exif_transpose(handle).convert("RGB"),(size,size),method=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().div(127.5).sub(1).unsqueeze(0).to(device)


@torch.inference_mode()
def infer(*,checkpoint:Path,qwen_checkpoint:Path,input_image:Path,prompt:str,output:Path,device:torch.device,seed:int=42)->dict:
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    editor_values=dict(payload["editor_config"]);editor_values["widths"]=tuple(editor_values["widths"])
    editor_config=SmallEditorConfig(**editor_values);text_config=QwenMiniConfig(**payload["qwen_mini_config"])
    model=SmallFullFrameEditor(editor_config);model.text=QwenMiniTextEncoder(text_config);model.load_state_dict(payload["model"]);model=model.to(device).eval()
    language,processor,qwen_report=load_half_qwen_text_encoder(qwen_checkpoint,device,keep_layers=1)
    encoded=_qwen_prompts(processor,[prompt]);ids=encoded["input_ids"][:,-text_config.max_length:].to(device);mask=encoded["attention_mask"][:,-text_config.max_length:].to(device)
    embeddings=language.embed_tokens(ids);reference=_load_rgb(input_image,editor_config.image_size,device)
    generator=torch.Generator(device=device).manual_seed(seed);noise=torch.randn(reference.shape,generator=generator,device=device)
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        generated=model(reference,(embeddings,mask),noise,None)
    array=generated[0].float().add(1).mul(127.5).clamp(0,255).byte().permute(1,2,0).cpu().numpy()
    output.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(array).save(output)
    report={"checkpoint":str(checkpoint),"input":str(input_image),"output":str(output),"prompt":prompt,"seed":seed,"counted_parameters":payload["counted_parameters"],"qwen_transformer_layers":2,"shared_token_embedding_parameters_excluded":qwen_report["token_embedding_parameters_excluded_by_project_convention"],"oracle_control_used":False,"generator_calls":1}
    output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    del model,language,processor
    if device.type=="cuda":torch.cuda.empty_cache()
    return report


def main()->int:
    parser=argparse.ArgumentParser(description="Run a trained Qwen-mini optical image editor")
    parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--qwen-checkpoint",type=Path,required=True);parser.add_argument("--input-image",type=Path,required=True);parser.add_argument("--prompt",required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--seed",type=int,default=42);parser.add_argument("--device",default="cuda")
    args=parser.parse_args();result=infer(checkpoint=args.checkpoint.resolve(),qwen_checkpoint=args.qwen_checkpoint.resolve(),input_image=args.input_image.resolve(),prompt=args.prompt,output=args.output.resolve(),device=torch.device(args.device),seed=args.seed);print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
