"""CPU-only TRAIN ablation: image conditions versus fixed source-specific noise.

No optimization, independent TEST or deployment shortcut. Cached full-Qwen
TRAIN features isolate the generator's two inputs; targets score outputs only.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import time

import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .qwen_vl_hybrid_run import load_data, completed_teacher
from .qwen_vl_spatial import build_spatial_head
from .qwen_vl_spatial_run import masks
from .qwen_vl_turbo_cache import image_float
from .qwen_vl_turbo_data import sha256, write_json
from .qwen_vl_turbo_model import pack_hidden
from .qwen_vl_turbo_run import noise_for, tensor_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','data','sd','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists():
        raise ValueError('CPU-only immutable TRAIN diagnostic required')
    report=completed_teacher(a.parent);a.output.mkdir(parents=True)
    write_json(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'device':'cpu',
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),
        'started_unix':time.time(),'TRAIN_only':True,'optimization':False})
    torch.set_num_threads(4)
    payload=load_data(a.data);groups={}
    for i,row in enumerate(payload['rows']):
        if row['split']=='train' and row['category']=='lamp' and row['mode']=='background':
            groups.setdefault(row['prompt'],[]).append(i)
    pair=None
    for ids in groups.values():
        for i,j in itertools.combinations(ids,2):
            if payload['rows'][i]['source']['product_id']!=payload['rows'][j]['source']['product_id']:
                pair=[i,j];break
        if pair is not None:break
    if pair is None:raise ValueError('Need two TRAIN products with the exact same instruction')
    rows=[payload['rows'][i] for i in pair]
    seeds=[int(r['noise_seed']) for r in rows]+[1042]
    if len(set(seeds))!=3:raise ValueError('Need distinct noise controls')
    state=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    if state['config'].get('semantic_prefix',0):raise ValueError('This probe targets the existing fused readout')
    head=build_spatial_head(a.sd,state,state['config'],state['visual_layout']['positions'],torch.device('cpu'),drop_vae_encoder=True)
    head=head.float().eval().requires_grad_(False)
    import lpips
    metric=lpips.LPIPS(net='alex').float().eval().requires_grad_(False)
    union=torch.maximum(*[masks(r,torch.device('cpu'))[1] for r in rows])
    y,x=torch.where(union[0,0]>.05)
    box=[max(0,int(x.min())-8),max(0,int(y.min())-8),min(256,int(x.max())+9),min(256,int(y.max())+9)]
    def crop(v):return F.interpolate(v[:,:,box[1]:box[3],box[0]:box[2]],(128,128),mode='bilinear',align_corners=False)
    outputs={};scores=[]
    with torch.inference_mode():
        targets=[image_float(payload['target'][i:i+1],torch.device('cpu')) for i in pair]
        for k,i in enumerate(pair):
            h,valid=pack_hidden([payload['hidden'][i]],torch.device('cpu'))
            for s,seed in enumerate(seeds):
                _,rgb=head(h,valid,noise_for([{'noise_seed':seed}],torch.device('cpu')))
                outputs[k,s]=rgb.clamp(-1,1)
                tensor_image(outputs[k,s][0]).save(a.output/f'condition_{k}_seed_{s}.png')
                scores.append({'source_condition':k,'noise_seed_index':s,'noise_seed':seed,
                    'matching_TRAIN_GT_LPIPS':float(metric(outputs[k,s],targets[k]).mean()),
                    'matching_TRAIN_GT_crop_LPIPS':float(metric(crop(outputs[k,s]),crop(targets[k])).mean()),
                    'other_TRAIN_GT_crop_LPIPS':float(metric(crop(outputs[k,s]),crop(targets[1-k])).mean())})
                write_json(a.output/'progress.json',{'done':len(scores),'total':6})
        distance=lambda x,y:float(metric(x,y).mean())
        effects={}
        for name,transform in [('scene',lambda x:x),('product_union_crop',crop)]:
            image=sum(distance(transform(outputs[0,s]),transform(outputs[1,s])) for s in range(3))/3
            noise=sum(distance(transform(outputs[k,0]),transform(outputs[k,1])) for k in range(2))/2
            effects[name]={'change_reference_same_seed_LPIPS':image,'change_source_seed_same_reference_LPIPS':noise,
                'noise_to_reference_sensitivity_ratio':noise/max(image,1e-8)}
    panel=Image.new('RGB',(1280,660),'white');draw=ImageDraw.Draw(panel)
    import textwrap
    draw.text((5,4),'TRAIN-only fixed-noise ablation. No optimization; cached Qwen conditions, not deployment timing.',fill='black')
    draw.multiline_text((5,20),'\n'.join(textwrap.wrap(rows[0]['prompt'],180)),fill='black')
    for k,i in enumerate(pair):
        y=72+k*290
        for j,label in enumerate(['REFERENCE','TRAIN GT (score only)','SOURCE A SEED','SOURCE B SEED','SHARED SEED 1042']):
            draw.text((j*256+5,y),label,fill='black')
        images=[tensor_image(payload['reference'][i]),tensor_image(payload['target'][i])]+[tensor_image(outputs[k,s][0]) for s in range(3)]
        for j,image in enumerate(images):panel.paste(image,(j*256,y+22))
    panel.save(a.output/'TRAIN_noise_condition_panel.png')
    write_json(a.output/'report.json',{'complete':True,'TRAIN_only':True,'gradient_updates':0,
        'parent_checkpoint_sha256':report['checkpoint_sha256'],'source_commit':os.environ.get('T12_SOURCE_COMMIT'),
        'sample_ids':[r['sample_id'] for r in rows],'same_prompt':rows[0]['prompt'],'noise_seeds':seeds,
        'effects':effects,'cases':scores,'offline_metric_union_box':box,'independent_TEST_opened':False,
        'note':'Sensitivity is not correctness. Fixed TRAIN source noise could be a memorization shortcut; this small crossed-input diagnostic can support or weaken that hypothesis, not establish its unique cause.'})


if __name__=='__main__':main()
