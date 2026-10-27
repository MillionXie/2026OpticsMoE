"""TRAIN/DEV rich reference-and-text product scenes; independent TEST sealed."""
import argparse
import collections
import json
import os
from pathlib import Path
import random
import re
import textwrap

from PIL import Image,ImageDraw

from .qwen_vl_scene_render import render_pair
from .qwen_vl_turbo_data import _pixels_hash,seed,sha256,validate_roles,write_json


def description(category,text):
    text=text.replace(', entirely visible','').replace('entirely visible','').strip().rstrip('.')
    if category=='lamp' and ' with ' in text and ' and ' in text:
        lead,detail=text.split(' with ',1)
        first=detail.split(',',1)[0];last=detail.rsplit(' and ',1)[1]
        if 'shade' in last and 'shade' not in first:
            text=lead+' with '+last+' and '+first
        elif 'shade' in first and 'base' in last:
            text=lead+' with '+first+' and '+last
    text=re.sub(r'\b(?:detailed|intricate|subtle|repeating)\b\s*','',text,flags=re.I)
    return text[0].lower()+text[1:]


def scene_text(bg):
    return f"a {bg['brightness']}, {bg['tone']} {bg['room']}; window light from the {bg['direction']}"


def pick_conditions(source,pool,count,key):
    fields=('room','tone','brightness','direction')
    choices=[p for p in pool if any(p[f]!=source[f] for f in fields)]
    rng=random.Random(seed(key,'rich-scenes'));rng.shuffle(choices)
    choices.sort(key=lambda p:sum(p[f]!=source[f] for f in fields))
    picked=[];seen=set()
    # Four nearby counterfactuals followed by varied full scene changes.
    for field in fields:
        candidates=[p for p in choices if p[field]!=source[field] and all(p[f]==source[f] for f in fields if f!=field)]
        if not candidates:raise ValueError('Missing single-attribute scene counterfactual')
        candidate=candidates[0];identity=tuple(candidate[f] for f in fields)
        picked.append(candidate);seen.add(identity)
        if len(picked)==min(4,count):break
    if len(picked)==count:return picked
    rest=[p for p in choices if tuple(p[f] for f in fields) not in seen];rng.shuffle(rest)
    for candidate in rest:
        identity=tuple(candidate[f] for f in fields)
        if identity not in seen:
            picked.append(candidate);seen.add(identity)
        if len(picked)==count:break
    if len(picked)!=count:raise ValueError('Insufficient distinct scene conditions')
    return picked


def table_for(asset,pool):
    matches=[p for p in pool if p['split']==asset['split'] and p['room']=='modern study'
             and all(p[f]==asset[f] for f in ('tone','brightness','direction','variant'))]
    if len(matches)!=1:raise ValueError('Missing same-split tabletop texture')
    return matches[0]


def prepare(a):
    if a.output.exists():raise FileExistsError('Immutable rich data: inspect instead')
    cfg=json.loads(a.review.read_text(encoding='utf-8'))
    if sha256(a.source/'annotated_pairs.json')!=cfg['parent_reviewed_pairs_sha256']:
        raise ValueError('Original reviewed view identity differs')
    bg_report=json.loads((a.backgrounds/'report.json').read_text())
    if not bg_report['complete'] or bg_report.get('preview_only') or bg_report.get('layout')!='tabletop':
        raise ValueError('Use a complete reviewed tabletop pool, not a probe/wide room')
    if sha256(a.backgrounds/'background_manifest.json')!=bg_report['manifest_sha256']:
        raise ValueError('Background manifest bytes differ')
    bgs=json.loads((a.backgrounds/'background_manifest.json').read_text())
    if {b['split'] for b in bgs}!={'train','val'}:raise ValueError('Heldout backgrounds are forbidden here')
    original=json.loads((a.source/'annotated_pairs.json').read_text())
    old_desc=json.loads((a.source/'product_descriptions.json').read_text())
    selected={s:{} for s in ('train','val')};desc={}
    for row in original:
        selected[row['split']].setdefault(row['source']['sample_id'],row['source'])
    for spec in cfg['additional_TRAIN_products']:
        raw={r['sample_id']:r for line in (a.raw/'train.jsonl').read_text().splitlines() if (r:=json.loads(line))}
        for view in spec['views']:
            sid=f"{spec['category']}-{spec['product_id']}-{view:02d}";p=dict(raw[sid])
            if p['license']!='CC BY 4.0':raise ValueError('Wrong ABO license')
            p.update(product_id=p['sequence_id'],family_id=p.get('parent_id') or p.get('family_id') or p['sequence_id'])
            for field in ('image_path','mask_path'):p[field]=str((a.raw/p[field]).resolve())
            p['rgba_sha256']=_pixels_hash(p);selected['train'][sid]=p;old_desc[sid]=spec['description']
    for group in selected.values():
        for sid,p in group.items():desc[sid]=description(p['category'],old_desc[sid])
    rows=[]
    for split,products in selected.items():
        pool=[p for p in bgs if p['split']==split];canonical={}
        for product in sorted(products.values(),key=lambda p:p['sample_id']):canonical.setdefault(product['product_id'],product)
        count=8 if split=='train' else 2
        for source in sorted(products.values(),key=lambda p:p['sample_id']):
            sbg=pool[seed(source['sample_id'],'source-scene')%len(pool)];source_table=table_for(sbg,pool)
            generator_seed=seed(1042,split,source['sample_id'],'latent-input')%(2**31)
            scenes=pick_conditions(sbg,pool,count,source['sample_id'])
            candidates=[p for p in canonical.values() if p['category']==source['category'] and p['product_id']!=source['product_id']]
            rng=random.Random(seed(source['sample_id'],'target-products'));rng.shuffle(candidates)
            seen=set();targets=[]
            for p in candidates:
                if desc[p['sample_id']] not in seen:targets.append(p);seen.add(desc[p['sample_id']])
            if len(targets)<count:raise ValueError('Not enough distinct target descriptions')
            if split=='train' and source['category']=='lamp':
                greens=[p for p in candidates if p['product_id']=='B076VF4KRN']
                if greens:
                    targets=[greens[0],greens[0]]+[p for p in targets if p['product_id']!='B076VF4KRN']
            for j in range(count):
                for mode in ('background','object','joint'):
                    target=source if mode=='background' else targets[j]
                    tbg=sbg if mode=='object' else scenes[j];table=table_for(tbg,pool)
                    changes=[f for f in ('room','tone','brightness','direction') if tbg[f]!=sbg[f]]
                    if mode!='object' and len(changes)==1 and changes[0] in ('tone','brightness'):
                        # Controlled exposure/white-balance edit: same room and
                        # tabletop pixels, changed only by the declared field.
                        field=changes[0];tbg={**sbg,field:tbg[field],'render_adjustment':field,
                            'parent_teacher_prompt':sbg['prompt']};table=source_table
                    target_desc=desc[target['sample_id']];scene=scene_text(tbg)
                    variant=None
                    if split=='train' and mode!='background':
                        if target['product_id']=='B076VF4KRN' and j%3!=2:
                            color='red' if j%3==0 else 'blue'
                            variant={'part':'shade','color':color};target_desc=target_desc.replace('green',color)
                        elif target['product_id']=='B075X2Y3VK' and j%3==0:
                            variant={'part':'base','color':'blue'};target_desc=target_desc.replace('teal','blue')
                    if mode=='background':prompt=f"Keep this {source['category']} and replace the background with {scene}."
                    elif mode=='object':prompt=f"Replace this {source['category']} with {target_desc}. Keep the background."
                    else:prompt=f"Generate a new product scene with {target_desc} in {scene}."
                    if re.search(r'\bB0[A-Z0-9]{8}\b|\bSKU\b|\bAmazon\b',prompt,re.I):raise ValueError('Product IDs are not model conditions')
                    rows.append({'sample_id':f"{split}:rich:{source['sample_id']}:{mode}:{j}",'split':split,
                        'mode':mode,'category':source['category'],'source':source,'target':target,
                        'target_description':target_desc,'target_color_variant':variant,
                        'prompt':prompt,'teacher_caption':f"Full view of {target_desc}, centered in {scene}.",
                        'render_protocol':'rich_product_scene_v1','edit_count':2 if mode=='joint' else 1,
                        'source_background_asset':sbg,'source_table_asset':source_table,
                        'target_background_asset':tbg,'target_table_asset':table,'noise_seed':generator_seed,
                        'source_scene':[sbg['room'].replace(' ','_'),sbg['tone'],sbg['brightness'],sbg['direction']],
                        'target_scene':[tbg['room'].replace(' ','_'),tbg['tone'],tbg['brightness'],tbg['direction']]})
    audit=validate_roles(rows)
    if audit['pairs']!={'train':1824,'val':96}:raise ValueError('Rich reviewed data budget differs')
    a.output.mkdir(parents=True);files={};identities={}
    for i,row in enumerate(rows):
        before,after=render_pair(row)
        for role,im in [('reference',before),('target',after)]:
            path=a.output/'images'/row['split']/role/f'{i:05d}.png';path.parent.mkdir(parents=True,exist_ok=True);im.save(path);files[str(path.relative_to(a.output))]=sha256(path)
        key=sha256(a.output/'images'/row['split']/'reference'/f'{i:05d}.png'),row['prompt'],row['noise_seed']
        value=sha256(a.output/'images'/row['split']/'target'/f'{i:05d}.png')
        if key in identities and identities[key]!=value:raise ValueError('Same RGB/prompt/noise has conflicting GT')
        identities[key]=value
    audit.update(protocol='rich_product_scene_v1',expected_counts={'train':1824,'val':96},
        review_sha256=sha256(a.review),background_manifest_sha256=bg_report['manifest_sha256'],
        profile_sha256=sha256(a.review),test_selection_authorized=True,evaluation_label='development_test',
        train_products=44,train_views=76,development_products=8,development_views=16,
        source_RGB_fixed_within_counterfactual_group=True,heldout_opened=False,
        generator_seed_independent_of_target=True,
        TRAIN_synthetic_component_colour_cases=sum(r['split']=='train' and r['target_color_variant'] is not None for r in rows),
        pixel_GTs_are_synthetic_composites=True,reference_or_GT_VAE_inference_branch=False,
        note='ABO genuine alpha and frozen-teacher room assets with a foreground tabletop and known tone/exposure/directional fields. Target images supervise TRAIN only. No compositing or GT access at inference. Independent 540 are not created here.')
    for name,obj in [('annotated_pairs.json',rows),('pilot_pairs.json',rows),('product_descriptions.json',desc),
                     ('data_audit.json',audit),('pairs_manifest.json',{'config':cfg,'audit':audit,'pairs':rows}),('image_sha256.json',files)]:write_json(a.output/name,obj)
    write_json(a.output/'inference_dev_inputs.json',[
        {'image':str((a.output/'images'/'val'/'reference'/f'{i:05d}.png').resolve()),
         'prompt':r['prompt'],'sample_id':r['sample_id'],'category':r['category'],'mode':r['mode']}
        for i,r in enumerate(rows) if r['split']=='val'])
    for cat in ('lamp','pillow'):
        for mode in ('background','object','joint'):
            picked=[];seen=set()
            for row in rows:
                if row['split']=='val' and row['category']==cat and row['mode']==mode and row['source']['product_id'] not in seen:
                    picked.append(row);seen.add(row['source']['product_id'])
                if len(picked)==3:break
            panel=Image.new('RGB',(520,348*len(picked)),'white');draw=ImageDraw.Draw(panel)
            for j,row in enumerate(picked):
                y=j*348;draw.text((4,y+2),'RICH DATA PAIRS ONLY | '+cat+'/'+mode,fill='black')
                draw.multiline_text((4,y+18),'\n'.join(textwrap.wrap(row['prompt'],82)),fill='black')
                draw.text((4,y+76),'REFERENCE',fill='black');draw.text((264,y+76),'TARGET (not editor output)',fill='black')
                for k,im in enumerate(render_pair(row)):panel.paste(im,(4+k*260,y+92))
            panel.save(a.output/f'{cat}_{mode}_data.png')
    write_json(a.output/'report.json',{'complete':True,'stage':'TRAIN_DEV_rich_data_preparation',
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'source_sha256':sha256(Path(__file__)),
        'pair_sha256':sha256(a.output/'annotated_pairs.json'),'profile_sha256':sha256(a.review),
        'audit':audit,'no_heldout_inference_or_training':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','raw','review','backgrounds','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=('prepare','cache'),default='prepare')
    p.add_argument('--qwen',type=Path);p.add_argument('--sd',type=Path);p.add_argument('--gpu-uuid')
    a=p.parse_args()
    if a.stage=='prepare':prepare(a);return
    report=json.loads((a.output/'report.json').read_text())
    if not report['complete'] or sha256(a.review)!=report['profile_sha256'] or sha256(a.output/'annotated_pairs.json')!=report['pair_sha256']:
        raise ValueError('Rich prepared dataset identity differs')
    if not a.qwen or not a.sd or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid:
        raise ValueError('Approved GPU and public assets required')
    import torch,sys,time
    from .qwen_vl_turbo_run import cache
    torch.set_num_threads(4);torch.manual_seed(1042);torch.cuda.set_per_process_memory_fraction(.65,0)
    write_json(a.output/'cache_execution.json',{'pid':os.getpid(),'argv':sys.argv,'started_unix':time.time(),
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'gpu_uuid':a.gpu_uuid,'reference_RGB_and_text_only':True})
    rows=json.loads((a.output/'annotated_pairs.json').read_text())
    digest=cache(a,{'image_size':256,'use_all_pairs':True,'uint8_image_cache':True,'clip_cache_batch_size':16},rows,torch.device('cuda:0'))
    write_json(a.output/'cache_complete.json',{'complete':True,'sha256':digest,'samples':len(rows),
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'reference_RGB_and_text_only':True})


if __name__=='__main__':main()
