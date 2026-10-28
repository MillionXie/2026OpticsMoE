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


def review_ids(rows,cohort):
    if cohort=='development':return [i for i,r in enumerate(rows) if r['split']=='val']
    if cohort!='train_probe':raise ValueError('Unknown review cohort')
    chosen={}
    for category in ('lamp','pillow'):
        sources={}
        for row in rows:
            if row['split']!='train' or row['category']!=category:continue
            source=row['source']
            if source['product_id'] not in sources and len(sources)<2:
                sources[source['product_id']]=source['sample_id']
        if len(sources)!=2:raise ValueError('Need two distinct TRAIN reference products per category')
        chosen[category]=set(sources.values())
    ids=[i for i,r in enumerate(rows) if r['split']=='train' and r['source']['sample_id'] in chosen[r['category']]]
    if {rows[i]['category'] for i in ids}!={'lamp','pillow'}:raise ValueError('Incomplete TRAIN probe')
    for category in chosen:
        if {rows[i]['mode'] for i in ids if rows[i]['category']==category}!={'background','object','joint'}:
            raise ValueError('Incomplete TRAIN task probe')
    return ids


def prompt_panels(output,metrics,label,folder='raw_dev',cohort_label='DEV'):
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
            draw.text((4,y+2),f"{label} | {row['category']}/{mode} | {cohort_label} | PSNR {row['psnr']:.2f} LPIPS {row['lpips']:.3f}",fill='black')
            draw.multiline_text((4,y+18),'\n'.join(textwrap.wrap(row['prompt'],110)),fill='black')
            draw.text((4,y+78),'REFERENCE                         TARGET (offline scoring)           GENERATED (raw RGB+text)',fill='black')
            for k,role in enumerate(('reference','target','generated')):
                with Image.open(output/'images'/folder/role/f'sample_{i:04d}.png') as im:panel.paste(im,(k*256,y+100))
        panel.save(output/(mode+'_full_prompts.png'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','data','qwen','sd','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--hybrid',type=Path);parser.add_argument('--tail',type=Path)
    parser.add_argument('--visual-cache',type=Path,help='Verified native merger cache for raw/cache identity only')
    parser.add_argument('--language-context',type=Path)
    parser.add_argument('--teacher-features',type=Path)
    parser.add_argument('--cohort',choices=('development','train_probe'),default='development')
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--label',required=True)
    a=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid or a.output.exists():raise ValueError('Unauthorized GPU or immutable existing review')
    completed_teacher(a.parent);a.output.mkdir(parents=True)
    write_json(a.output/'execution.json',{'argv':sys.argv,'pid':os.getpid(),'source_commit':os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256':sha256(Path(__file__)),'started_unix':time.time(),'gpu_uuid':a.gpu_uuid})
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.85,0);device=torch.device('cuda:0')
    state=torch.load(a.parent/'best_checkpoint.pt',map_location='cpu',weights_only=False)
    payload=load_data(a.data);ids=review_ids(payload['rows'],a.cohort)
    if state['config'].get('semantic_prefix',0):
        if a.visual_cache is None:raise ValueError('Native readout requires its explicit cache for identity checks')
        ready=json.loads((a.visual_cache/'report.json').read_text());path=a.visual_cache/'native_vision.pt'
        if not ready['complete'] or ready['sha256']!=sha256(path) or ready['source_cache_sha256']!=sha256(a.data/'frozen_features.pt'):
            raise ValueError('Native visual cache identity changed')
        native=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if native['sample_ids']!=[r['sample_id'] for r in payload['rows']]:raise ValueError('Native feature row identities differ')
        payload['hidden']=[torch.cat([v,h],0) for v,h in zip(native['vision'],payload['hidden'])]
    qwen,processor=load_qwen(a.qwen,device)
    if state.get('schema')=='qwen_spatial_language_lora_v1':
        from .qwen_vl_language_adapt import install_in_full_qwen,NativeLanguageTail
        from .qwen_vl_hybrid_run import checked_features
        install_in_full_qwen(qwen,state)
        if a.language_context is None or a.teacher_features is None:raise ValueError('Adapted Qwen raw replay requires its verified frozen-prefix contexts')
        ready=json.loads((a.language_context/'report.json').read_text());path=a.language_context/'native_contexts.pt'
        if not ready['complete'] or ready['sha256']!=sha256(path) or ready['sha256']!=state['config']['language_context_sha256']:
            raise ValueError('Adapted Qwen native contexts differ')
        contexts=torch.load(path,map_location='cpu',weights_only=False)
        features=checked_features(a.teacher_features,payload)
        if contexts['prefix_feature_sha256']!=state['config']['language_prefix_sha256']:raise ValueError('Adapted prefix SHA differs')
        lm=qwen.model.language_model
        replay=NativeLanguageTail(list(lm.layers[24:28]),lm.norm,contexts['contexts']).eval().requires_grad_(False)
        with torch.no_grad():
            for i in ids:
                prefix=features[i]['language_23'].to(device)
                payload['hidden'][i]=replay(prefix,torch.ones(prefix.shape[:2],dtype=torch.bool,device=device))[0].cpu()
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
    folder='raw_dev' if a.cohort=='development' else 'raw_train'
    metrics=evaluate(head,temporary,ids,device,a.output,folder,metric,export=True)
    prompt_panels(a.output,metrics,a.label,folder,'DEV' if a.cohort=='development' else 'TRAIN FIT ONLY')
    write_json(a.output/'routing.json',routes);write_json(a.output/'raw_cached_identity.json',checks)
    write_json(a.output/'report.json',{'complete':True,'model_label':a.label,'count':len(ids),'metrics':metrics,
        'parent_checkpoint_sha256':sha256(a.parent/'best_checkpoint.pt'),
        'hybrid_checkpoint_sha256':sha256(a.hybrid/'best_checkpoint.pt') if a.hybrid else None,
        'tail_checkpoint_sha256':sha256(a.tail/'best_checkpoint.pt') if a.tail else None,
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),
        'evaluation_label':'development_test' if a.cohort=='development' else 'TRAIN_resubstitution_diagnostic',
        'cohort':a.cohort,'independent_test_opened':False,
        'sample_selection':'All DEV rows' if a.cohort=='development' else 'First view of first two distinct TRAIN reference products/category; all original counterfactual instructions; no quality filtering',
        'generator_input':'Reference RGB and full instruction through actual Qwen, no cached hidden bypass and no GT/mask/CLIP inference branch',
        'completed_unix':time.time(),'quality_accepted':False})


if __name__=='__main__':main()
