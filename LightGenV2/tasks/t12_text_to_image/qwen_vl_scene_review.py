"""Raw RGB+text DEV export with full prompts and explicit model identities.

Targets/masks enter the offline scorer and figure layout only. This entry never
opens the independent evaluation set or supplies a target to the generator.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import textwrap
import time

import torch
from PIL import Image,ImageDraw

from .qwen_vl_hybrid import apply_configured_hybrid,load_macro_state,restore_head_overlay
from .qwen_vl_hybrid_run import completed_teacher,load_data,load_qwen,raw_hidden
from .qwen_vl_spatial import build_spatial_head
from .qwen_vl_spatial_run import evaluate
from .qwen_vl_turbo_data import sha256,write_json
from .qwen_vl_turbo_model import pack_hidden
from .qwen_vl_turbo_run import noise_for,tensor_image


def prompt_panels(output,metrics,label):
    for mode in ('background','object','joint'):
        chosen=[]
        for category in ('lamp','pillow'):
            candidates=[(i,r) for i,r in enumerate(metrics['images']) if r['category']==category and r['mode']==mode]
            seen=set()
            for i,row in candidates:
                # Figure diversity only: first three distinct reference IDs,
                # independent of quality scores, never cherry-pick the best.
                reference=row['sample_id'].split(':'+mode+':')[0]
                if reference not in seen:chosen.append((i,row));seen.add(reference)
                if len(seen)==3:break
        height=360;panel=Image.new('RGB',(768,height*len(chosen)),'white');draw=ImageDraw.Draw(panel)
        for j,(i,row) in enumerate(chosen):
            y=j*height
            draw.text((4,y+2),f"{label} | {row['category']}/{mode} | DEV | PSNR {row['psnr']:.2f} LPIPS {row['lpips']:.3f}",fill='black')
            draw.multiline_text((4,y+18),'\n'.join(textwrap.wrap(row['prompt'],110)),fill='black')
            draw.text((4,y+78),'REFERENCE                         TARGET (offline scoring)           GENERATED (raw RGB+text)',fill='black')
            for k,role in enumerate(('reference','target','generated')):
                with Image.open(output/'images'/'raw_dev'/role/f'sample_{i:04d}.png') as im:panel.paste(im,(k*256,y+100))
        panel.save(output/(mode+'_full_prompts.png'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','data','qwen','sd','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--hybrid',type=Path);parser.add_argument('--tail',type=Path)
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--label',required=True)
    a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid or a.output.exists():raise ValueError('Unauthorized GPU or immutable existing review')
    completed_teacher(a.parent);a.output.mkdir(parents=True)
    write_json(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256':sha256(Path(__file__)),'started_unix':time.time(),'gpu_uuid':a.gpu_uuid})
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.85,0);device=torch.device('cuda:0')
    state=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    payload=load_data(a.data);ids=[i for i,r in enumerate(payload['rows']) if r['split']=='val']
    qwen,processor=load_qwen(a.qwen,device)
    head=build_spatial_head(a.sd,state,state['config'],state['visual_layout']['positions'],device,drop_vae_encoder=True).eval().requires_grad_(False)
    if a.hybrid:
        hybrid_report=json.loads((a.hybrid/'report.json').read_text())
        if not hybrid_report['complete'] or sha256(a.hybrid/'best_checkpoint.pt')!=hybrid_report['checkpoint_sha256']:raise ValueError('Hybrid identity differs')
        hybrid=torch.load(a.hybrid/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        macros=apply_configured_hybrid(qwen,hybrid['config']);load_macro_state(macros,hybrid['macros'])
        restore_head_overlay(head,hybrid);qwen.eval().requires_grad_(False)
    if a.tail:
        from .qwen_vl_tail import build_tail_student
        tail=torch.load(a.tail/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        if not a.hybrid or tail['hybrid_checkpoint_sha256']!=sha256(a.hybrid/'best_checkpoint.pt'):raise ValueError('Tail/hybrid identity differs')
        unet,_=build_tail_student(head.unet,tail['profile']['channels'],tail['profile']['heads'])
        unet.load_state_dict(tail['unet']);head.unet=unet.to(device).eval().requires_grad_(False)
    temporary=dict(payload);temporary['hidden']=list(payload['hidden']);routes=[];checks=[];checked_cells=set()
    with torch.no_grad():
        for number,i in enumerate(ids):
            hidden=raw_hidden(qwen,processor,payload,i,state['visual_layout'],device)
            temporary['hidden'][i]=hidden.cpu()
            if a.hybrid:routes.append({'sample_id':payload['rows'][i]['sample_id'],**{m:v.diagnostics() for m,v in macros.items()}})
            cell=payload['rows'][i]['category'],payload['rows'][i]['mode']
            if not a.hybrid and cell not in checked_cells:
                checked_cells.add(cell)
                h,valid=pack_hidden([hidden],device);_,raw=head(h,valid,noise_for([payload['rows'][i]],device))
                h,valid=pack_hidden([payload['hidden'][i]],device);_,cached=head(h,valid,noise_for([payload['rows'][i]],device))
                checks.append({'sample_id':payload['rows'][i]['sample_id'],'raw_cached_same_png':tensor_image(raw[0]).tobytes()==tensor_image(cached[0]).tobytes(),
                    'max_abs_hidden':float((hidden.float()-payload['hidden'][i].to(device).float()).abs().max()),'max_abs_RGB':float((raw-cached).abs().max())})
            if (number+1)%12==0:write_json(a.output/'progress.json',{'stage':'raw_DEV_features','done':number+1,'total':len(ids)})
    import lpips
    metric=lpips.LPIPS(net='alex').to(device).eval().requires_grad_(False)
    metrics=evaluate(head,temporary,ids,device,a.output,'raw_dev',metric,export=True)
    prompt_panels(a.output,metrics,a.label)
    write_json(a.output/'routing.json',routes);write_json(a.output/'raw_cached_identity.json',checks)
    write_json(a.output/'report.json',{'complete':True,'model_label':a.label,'count':len(ids),'metrics':metrics,
        'parent_checkpoint_sha256':sha256(a.parent/'best_checkpoint.pt'),
        'hybrid_checkpoint_sha256':sha256(a.hybrid/'best_checkpoint.pt') if a.hybrid else None,
        'tail_checkpoint_sha256':sha256(a.tail/'best_checkpoint.pt') if a.tail else None,
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),
        'evaluation_label':'development_test','independent_test_opened':False,
        'generator_input':'Reference RGB and full instruction through actual Qwen, no cached hidden bypass and no GT/mask/CLIP inference branch',
        'completed_unix':time.time(),'quality_accepted':False})


if __name__=='__main__':main()
