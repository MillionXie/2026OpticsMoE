"""CPU-only correction of declared TRAIN color targets, with verified cache reuse.

Every reference pixel, prompt, caption and frozen Qwen input is unchanged. Only
the 150 synthetic TRAIN color targets receive the explicit HSV component recipe.
Original data/checkpoints remain immutable; no DEV/TEST labels are modified.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import textwrap

import numpy as np
from PIL import Image,ImageDraw
import torch

from .qwen_vl_scene_render import compose,render_pair
from .qwen_vl_turbo_cache import save_with_digest
from .qwen_vl_turbo_data import sha256,write_json,validate_roles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('Immutable CPU-only repair required')
    torch.set_num_threads(4)
    ready=json.loads((a.source/'cache_complete.json').read_text());parent_sha=sha256(a.source/'frozen_features.pt')
    if not ready['complete'] or parent_sha!=ready['sha256']:raise ValueError('Original cache identity differs')
    payload=torch.load(a.source/'frozen_features.pt',map_location='cpu',weights_only=False,mmap=True)
    if payload['reference'].dtype!=torch.uint8 or payload['target'].dtype!=torch.uint8:raise ValueError('Require audited uint8 cache')
    rows=copy.deepcopy(payload['rows']);old_files=json.loads((a.source/'image_sha256.json').read_text())
    audit=json.loads((a.source/'data_audit.json').read_text());report=json.loads((a.source/'report.json').read_text())
    if not report['complete'] or not audit['role_disjoint'] or audit['test_opened']:raise ValueError('Data split identity differs')
    a.output.mkdir(parents=True);target_tensor=payload['target'].clone();new_files={};changed=[];examples={}
    for i,row in enumerate(rows):
        variant=row.get('target_color_variant')
        affected=row['split']=='train' and variant is not None
        if variant is not None and row['split']!='train':raise ValueError('Synthetic color variant outside TRAIN')
        if affected:row['target_color_variant']={**variant,'recipe':'hsv_component_hull_v2'}
        for role in ('reference','target'):
            name=f"images/{row['split']}/{role}/{i:05d}.png";old=a.source/name;new=a.output/name
            if sha256(old)!=old_files[name]:raise ValueError('Original PNG changed')
            with Image.open(old) as im:original=np.asarray(im.convert('RGB')).copy()
            if not torch.equal(torch.from_numpy(original).permute(2,0,1),payload[role][i]):raise ValueError('Original cached RGB differs from its exported PNG')
            new.parent.mkdir(parents=True,exist_ok=True)
            if affected and role=='target':
                before,after=render_pair(row)
                if not np.array_equal(np.asarray(before),payload['reference'][i].permute(1,2,0).numpy()):raise ValueError('Repair changed reference RGB')
                after.save(new);target_tensor[i]=torch.from_numpy(np.asarray(after).copy()).permute(2,0,1)
                if np.array_equal(np.asarray(after),original):raise ValueError('Declared target correction changed no pixels')
                changed.append({'sample_id':row['sample_id'],'old_target_sha256':old_files[name],'new_target_sha256':sha256(new)})
                examples.setdefault((variant['part'],variant['color']),(i,row))
            else:shutil.copyfile(old,new)
            new_files[name]=sha256(new)
    if len(changed)!=150:raise ValueError('Unexpected TRAIN repair budget')
    validate_roles(rows)
    audit.update(color_target_repair='hsv_component_hull_v2',corrected_TRAIN_targets=len(changed),
        reference_pixels_changed=0,prompts_changed=0,DEV_targets_changed=0,TEST_opened=False,
        parent_data_report_sha256=sha256(a.source/'report.json'),parent_cache_sha256=parent_sha)
    for name,value in [('annotated_pairs.json',rows),('pilot_pairs.json',rows),('data_audit.json',audit),('image_sha256.json',new_files),('target_corrections.json',changed)]:write_json(a.output/name,value)
    shutil.copyfile(a.source/'product_descriptions.json',a.output/'product_descriptions.json')
    write_json(a.output/'pairs_manifest.json',{'audit':audit,'pairs':rows,'parent':str(a.source.resolve())})
    write_json(a.output/'inference_dev_inputs.json',[{'image':str((a.output/f"images/val/reference/{i:05d}.png").resolve()),
        'prompt':row['prompt'],'sample_id':row['sample_id'],'category':row['category'],'mode':row['mode']}
        for i,row in enumerate(rows) if row['split']=='val'])
    for old,new in zip(payload['rows'],rows):
        if any(old[key]!=new[key] for key in ('sample_id','split','source','prompt','teacher_caption','noise_seed')):raise ValueError('Cache reuse input/text identity differs')
    repaired={**payload,'rows':rows,'target':target_tensor,'pilot_sha256':sha256(a.output/'pilot_pairs.json'),
        'target_only_rebind':{'parent_cache_sha256':parent_sha,'reference_RGB_text_caption_noise_identical':True}}
    digest=save_with_digest(repaired,a.output/'frozen_features.pt')
    write_json(a.output/'cache_complete.json',{'complete':True,'sha256':digest,'samples':len(rows),
        'source_commit':os.environ.get('T12_SOURCE_COMMIT'),'reference_RGB_and_text_unchanged':True,
        'frozen_frontend_reused_from_verified_cache':parent_sha,'new_Qwen_forward_or_TEST_cache':False})
    panel=Image.new('RGB',(768,340*len(examples)),'white');draw=ImageDraw.Draw(panel)
    for j,((part,color),(i,row)) in enumerate(examples.items()):
        y=j*340;draw.text((4,y+2),f'TRAIN DATA REPAIR ONLY | {part}/{color} | not model outputs',fill='black')
        draw.multiline_text((4,y+18),'\n'.join(textwrap.wrap(row['prompt'],110)),fill='black')
        draw.text((4,y+62),'ORIGINAL ABO COLOR               OLD TRAIN AUGMENT                 CORRECTED TRAIN AUGMENT',fill='black')
        plain=compose(row['target'],row['target_background_asset'],row['target_table_asset'])
        for k,image in enumerate((plain,Image.open(a.source/f'images/train/target/{i:05d}.png'),Image.open(a.output/f'images/train/target/{i:05d}.png'))):panel.paste(image,(k*256,y+84))
    panel.save(a.output/'TRAIN_color_repair_QA.png')
    write_json(a.output/'report.json',{'complete':True,'stage':'TRAIN_color_target_correction','source_commit':os.environ.get('T12_SOURCE_COMMIT'),
        'source_sha256':sha256(Path(__file__)),'renderer_sha256':sha256(Path(__file__).with_name('qwen_vl_scene_render.py')),
        'profile_sha256':report['profile_sha256'],'pair_sha256':sha256(a.output/'annotated_pairs.json'),'cache_sha256':digest,
        'audit':audit,'parent_cache_sha256':parent_sha,'TRAIN_targets_corrected':150,
        'reference_pixels_prompts_captions_noise_changes':0,'DEV_targets_changed':0,'independent_TEST_opened':False,
        'reason':'Absolute RGB-spread threshold and clipped luminance rescaling left patchy green/red color labels. Explicit HSV component support repairs dark surfaces, keeps saturation/value and protects brass/highlights. Historical recipe remains available for exact old-data reproduction.'})


if __name__=='__main__':main()
