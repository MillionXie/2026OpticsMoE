"""Frozen SDXL+IP-Adapter+ControlNet sweep; geometry from reference RGB only."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from .white_reference_scene_probe import (GPU, SCHEMA, checked_tokens, conditions,
    digest, model_manifest, native_caption, render_panels, write)


def outline_from_rgb(reference, resolution, preset):
    """Remove internal appearance edges; preserve aspect and thin structures.

    This binary preprocessing is an input-derived condition, never a GT mask.
    A smaller centered outline leaves room for furniture; it does not preserve
    the reference crop/composition. Original RGB is not modified or pasted back.
    """
    import cv2
    rgb = np.asarray(reference.convert('RGB'))
    h, w = rgb.shape[:2]
    if min(h,w) < 32 or any(rgb[y,x].min()<245 for y,x in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]):
        raise ValueError('White-background reference required')
    mask = (rgb.min(axis=2) < preset['white_threshold']).astype(np.uint8)*255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c)>=preset['minimum_component_area']]
    if not contours: raise ValueError('No visible product silhouette')
    points = np.concatenate(contours,axis=0)
    x,y,bw,bh = cv2.boundingRect(points)
    fraction = float((mask>0).mean())
    if not .005 < fraction < .9: raise ValueError('Invalid white-reference foreground occupancy')
    scale = resolution*preset['maximum_product_extent']/max(bw,bh)
    offset = np.array([resolution*.5-bw*scale*.5,resolution*preset['center_y']-bh*scale*.5])
    transformed = [np.rint((c.astype(np.float32)-[x,y])*scale+offset).astype(np.int32) for c in contours]
    edge = np.zeros((resolution,resolution),np.uint8)
    cv2.drawContours(edge,transformed,-1,255,preset['edge_width'],lineType=cv2.LINE_8)
    return Image.fromarray(edge).convert('RGB'), {
        'source_RGB_size':[w,h],'source_bbox_xywh':[x,y,bw,bh],
        'foreground_fraction':fraction,'external_contours':len(contours),
        'target_composition':'centered input-derived outline with room around it',
        'internal_texture_edges_used':False,'GT_or_segmentation_model_used':False,
        'recipe':preset}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','inputs','model-path','adapter-path','controlnet-path','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text(encoding='utf-8'))
    commit=os.environ.get('T12_SOURCE_COMMIT','')
    if args.output.exists() or len(commit)!=40 or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:
        raise ValueError('New run, published commit, authorized GPU6 required')
    if cfg['schema']!=SCHEMA or cfg['prompt_profile']!='free_lighting_v2':raise ValueError('Reviewed no-lighting protocol required')
    args.output.mkdir(parents=True)
    write(args.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':commit,
        'source_sha256':digest(Path(__file__)),'config':cfg,'config_sha256':digest(args.config),
        'versions':{n:importlib.metadata.version(n) for n in ['torch','diffusers','transformers','opencv-python']}})
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        from transformers import CLIPVisionModelWithProjection
        torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.8,0)
        inputs=json.loads((args.inputs/'reference_manifest.json').read_text(encoding='utf-8'))
        if inputs['schema']!=SCHEMA or len(inputs['references'])!=4 or any(r['split']!='train' for r in inputs['references']):
            raise ValueError('Exactly four reviewed TRAIN references required')
        assets={'base':model_manifest(args.model_path),'adapter':model_manifest(args.adapter_path),
                'controlnet':model_manifest(args.controlnet_path)}
        write(args.output/'model_assets.json',assets)
        preset=cfg['sdxl_ipadapter'];resolution=preset['resolution'];geometry={}
        sheet=Image.new('RGB',(512,4*282),'white');draw=ImageDraw.Draw(sheet)
        for i,ref in enumerate(inputs['references']):
            path=args.inputs/ref['input_PNG']
            if digest(path)!=ref['input_sha256']:raise ValueError('Reference changed')
            with Image.open(path) as im:reference=im.convert('RGB').copy()
            edge,meta=outline_from_rgb(reference,resolution,cfg['outline'])
            name=ref['reference_id']+'_outline.png';edge.save(args.output/name)
            geometry[ref['reference_id']]={**meta,'edge_PNG':name,'edge_sha256':digest(args.output/name),
                                         'reference_sha256':ref['input_sha256']}
            draw.text((4,i*282+3),ref['reference_id']+' | original reference -> derived OUTLINE, no GT',fill='black')
            sheet.paste(reference.resize((256,256)),(0,i*282+24))
            sheet.paste(edge.resize((256,256)),(256,i*282+24))
        sheet.save(args.output/'input_outline_preview.png');write(args.output/'geometry.json',geometry)
        control=ControlNetModel.from_pretrained(args.controlnet_path,variant='fp16',torch_dtype=torch.float16,local_files_only=True)
        encoder=CLIPVisionModelWithProjection.from_pretrained(args.adapter_path/'models/image_encoder',torch_dtype=torch.float16,local_files_only=True)
        pipe=StableDiffusionXLControlNetPipeline.from_pretrained(args.model_path,variant='fp16',torch_dtype=torch.float16,
                  controlnet=control,image_encoder=encoder,local_files_only=True).to('cuda')
        pipe.load_ip_adapter(str(args.adapter_path),subfolder='sdxl_models',weight_name='ip-adapter-plus_sdxl_vit-h.safetensors',
                             image_encoder_folder=None,local_files_only=True)
        pipe.set_ip_adapter_scale(preset['image_scale']);pipe.set_progress_bar_config(disable=True)
        for component in pipe.components.values():
            if isinstance(component,torch.nn.Module):component.eval().requires_grad_(False)
        count=sum(p.numel() for p in {id(p):p for c in pipe.components.values() if isinstance(c,torch.nn.Module) for p in c.parameters()}.values())
        variant_reports={}
        for variant in cfg['variants']:
            out=args.output/variant['name'];out.mkdir();records=[]
            write(out/'execution.json',{'parent':str(args.output),'source_commit':commit,'variant':variant,
                  'shared_model_assets_sha256':digest(args.output/'model_assets.json'),'parameters_in_loaded_pipeline':count})
            for ref in inputs['references']:
                shutil.copyfile(args.inputs/ref['input_PNG'],out/ref['input_PNG'])
                edge_name=geometry[ref['reference_id']]['edge_PNG'];shutil.copyfile(args.output/edge_name,out/edge_name)
                with Image.open(args.inputs/ref['input_PNG']) as im:reference=im.convert('RGB').copy()
                with Image.open(args.output/edge_name) as im:edge=im.convert('RGB').copy()
                for condition in conditions(ref['category'],cfg['prompt_profile']):
                    prompt=native_caption(ref['category'],condition,preset['prompt_style'],cfg['prompt_profile'])
                    token_audit=[checked_tokens(t,prompt) for t in [pipe.tokenizer,pipe.tokenizer_2]]
                    for seed in cfg['seeds']:
                        began=time.time();generator=torch.Generator(device='cuda').manual_seed(seed)
                        noise=torch.randn((1,4,resolution//8,resolution//8),generator=generator,device='cuda',dtype=torch.float16)
                        noise_sha=__import__('hashlib').sha256(noise.cpu().numpy().tobytes()).hexdigest()
                        with torch.inference_mode():
                            result=pipe(prompt=prompt,negative_prompt=cfg['negative_prompt'],ip_adapter_image=reference,
                                image=edge,width=resolution,height=resolution,num_inference_steps=preset['steps'],
                                guidance_scale=preset['text_guidance'],generator=generator,latents=noise,
                                controlnet_conditioning_scale=variant['control_scale'],control_guidance_start=0.,
                                control_guidance_end=cfg['control_guidance_end']).images[0]
                        if result.size!=(resolution,resolution):raise ValueError('Native resolution mismatch')
                        name=f'{ref["reference_id"]}_condition{condition["condition_id"]}_seed{seed}.png';result.save(out/name)
                        records.append({'model':'sdxl_ipadapter_controlnet_outline','reference_id':ref['reference_id'],
                            'condition_id':condition['condition_id'],'category':ref['category'],'prompt':condition['prompt'],'model_prompt':prompt,
                            'requested_attributes':condition['requested_attributes'],'seed':seed,'resolution':resolution,'preset':preset,
                            'control_scale':variant['control_scale'],'input_sha256':ref['input_sha256'],'derived_outline':geometry[ref['reference_id']],
                            'initial_latent_sha256':noise_sha,'full_prompt_tokens':token_audit,'generated_PNG':name,'generated_sha256':digest(out/name),
                            'diagnostic_generation_seconds_not_latency_benchmark':time.time()-began})
                        write(out/'images.json',records);write(args.output/'progress.json',{'variant':variant['name'],'done':len(records),'total':16})
            panels=render_panels(out,records,inputs)
            report={'complete':True,'schema':SCHEMA,'model':'sdxl_ipadapter_controlnet_outline','source_commit':commit,
                'generated_count':len(records),'prompt_profile':cfg['prompt_profile'],'input_resolution':256,'inference_resolution':resolution,
                'training_performed':False,'quality_accepted':False,'independent_TEST_opened':False,
                'generator_input_whitelist':['white reference RGB','full text','seed'],'derived_input_condition':'RGB external outline only, no GT mask',
                'GT_mask_retrieval_paste_or_external_background_used':False,'parameters_in_loaded_pipeline':count,'panels':panels,
                'shared_assets_sha256':digest(args.output/'model_assets.json'),'reference_manifest_sha256':digest(args.inputs/'reference_manifest.json')}
            write(out/'report.json',report);variant_reports[variant['name']]=report
        # All variants must use identical initial noise and the same complete prompts.
        sets=[json.loads((args.output/v['name']/'images.json').read_text()) for v in cfg['variants']]
        for rows in sets[1:]:
            assert [(r['reference_id'],r['condition_id'],r['seed'],r['prompt'],r['initial_latent_sha256']) for r in rows]==[
                (r['reference_id'],r['condition_id'],r['seed'],r['prompt'],r['initial_latent_sha256']) for r in sets[0]]
        write(args.output/'report.json',{'complete':True,'source_commit':commit,'generated_count':sum(r['generated_count'] for r in variant_reports.values()),
            'variants':variant_reports,'shared_noise_and_prompts_verified':True,'quality_accepted':False,'independent_TEST_opened':False,
            'limitations':'Four TRAIN products, frozen development feasibility, no unique GT or model training, all seeds retained; not independent quality or latency benchmark.'})
        del pipe;gc.collect();torch.cuda.empty_cache()
    except Exception:write(args.output/'failure.json',{'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
