"""Target-free scene requests, product-disjoint, before synthetic TRAIN labels.

Only original white ABO RGB is copied. Old rendered scene targets/masks are never
read. TRAIN/DEV roles come from the reviewed cohort; held-out products inherit
the previous ancestor-exposure audit and remain prediction-free at preparation.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image,ImageDraw
from .white_reference_scene_probe import digest,write

SCHEMA='abo_white_scene_requests_v1'
COLORS=['deep blue','terracotta','olive green','charcoal grey']
LAMP_SCENES=[('modern study','on an oak desk','bookshelves, curtains and a small plant'),
 ('reading corner','on a pale stone side table','an armchair, bookshelves and framed abstract shapes'),
 ('quiet bedroom','on a wooden bedside table','a linen bed, curtains and a small plant'),
 ('warm living room','on a wooden console','a sofa, books and framed abstract shapes'),
 ('home office','on a walnut desk','wall shelves, books and curtains'),
 ('bright sunroom','on a stone console','large windows, leafy plants and a woven chair'),
 ('compact apartment','on a wooden side table','a sofa, a rug and open shelves'),
 ('cozy library','on a wooden reading desk','bookcases, an armchair and a leafy plant')]
PILLOW_SCENES=[('modern living room','on a beige sofa','an oak coffee table, curtains and a leafy plant'),
 ('reading nook','on a grey armchair','a stone side table, bookshelves and a leafy plant'),
 ('quiet bedroom','on a linen bed','a bedside table, curtains and framed abstract shapes'),
 ('warm lounge','on a fabric sofa','a wooden coffee table, a rug and bookshelves'),
 ('home office lounge','on a small sofa','a desk, wall shelves and a leafy plant'),
 ('bright sunroom','on a wooden bench','large windows, plants and a woven rug'),
 ('compact apartment','on a grey loveseat','open shelves, curtains and a side table'),
 ('cozy library','on a reading chair','bookcases, a wooden table and a leafy plant')]


def bank(category):
    if category not in ['lamp','pillow']:raise ValueError('Only approved categories')
    rows=[]
    for c,d,s in itertools.product(range(4),range(2),range(8)):
        scene,support,props=(LAMP_SCENES if category=='lamp' else PILLOW_SCENES)[s]
        detail=['fabric','matte metal'][d] if category=='lamp' else ['plain woven fabric','broad cream stripes'][d]
        subject=(f'Give the lampshade a {COLORS[c]} {detail} finish.' if category=='lamp' else
                 f'Give the cushion {COLORS[c]} '+('woven fabric with a plain surface.' if d==0 else 'fabric with broad cream stripes.'))
        prompt=(f'Generate a realistic {scene} using the reference {"lamp" if category=="lamp" else "cushion"}. '
          f'Retain its overall shape. {subject} Place it {support}, with {props} in the room. Other small details may vary naturally.')
        rows.append({'condition_id':f'{category}-c{c}d{d}s{s}','category':category,'color_index':c,'detail_index':d,'scene_index':s,
          'prompt':prompt,'attributes':{'product_color':COLORS[c],'product_detail':detail,'scene':scene,'support':support,'background_elements':props}})
    return rows


def condition_subset(category,split):
    rows=bank(category)
    if split=='train':return [r for r in rows if r['scene_index'] in [(2*r['color_index']+r['detail_index']+4*k)%8 for k in range(2)]]
    if split=='dev':return [r for r in rows if r['scene_index']==(2*r['color_index']+r['detail_index']+2)%8]
    if split=='test':return random.Random('scene-v1-'+category).sample(rows,50)
    raise ValueError('Unknown split')


def reference_catalog(rows,heldout):
    pool=defaultdict(dict)
    def insert(split,item):
        if not isinstance(item,dict) or item.get('category') not in ['lamp','pillow']:return
        previous=pool[split].get(item['sample_id'])
        if previous is not None and any(previous.get(k)!=item.get(k) for k in ['product_id','family_id','category','image_path']):
            raise ValueError('One source sample has inconsistent identity metadata')
        pool[split][item['sample_id']]=item
    for row in rows:
        split={'train':'train','val':'dev'}[row['split']]
        for role in ['source','target']:
            item=row[role]
            insert(split,item)
    for row in heldout:
        for role in ['source','target']:
            item=row[role]
            insert('test',item)
    # One fixed reviewed original view per TEST product, 50 distinct requests.
    unique={}
    for item in sorted(pool['test'].values(),key=lambda r:r['sample_id']):unique.setdefault(item['product_id'],item)
    pool['test']={r['sample_id']:r for r in unique.values()}
    ids={s:{r['product_id'] for r in items.values()} for s,items in pool.items()}
    families={s:{r.get('family_id',r['product_id']) for r in items.values()} for s,items in pool.items()}
    for a,b in itertools.combinations(ids,2):
        if ids[a]&ids[b] or families[a]&families[b]:raise ValueError('Cross-role product/family leakage')
    if Counter(r['category'] for r in pool['test'].values())!={'lamp':5,'pillow':5}:raise ValueError('Reviewed five+five TEST products required')
    return pool


def request_seed(reference_sha,seed_index):
    return int(hashlib.sha256((reference_sha+'|scene-noise|'+str(seed_index)).encode()).hexdigest()[:8],16)%(2**31)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-run','heldout-run','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();commit=os.environ.get('T12_SOURCE_COMMIT','')
    if a.output.exists() or len(commit)!=40:raise ValueError('Published source/new run required')
    report=json.loads((a.source_run/'report.json').read_text());audit=json.loads((a.source_run/'data_audit.json').read_text())
    held_report=json.loads((a.heldout_run/'report.json').read_text())
    if not report['complete'] or not audit['role_disjoint'] or audit['test_opened'] or not held_report['complete'] or held_report['model_inference']:
        raise ValueError('Source role and unopened historical heldout audit required')
    if held_report['evaluation_pairs_sha256']!=digest(a.heldout_run/'evaluation_pairs.json'):raise ValueError('Heldout source inventory changed')
    pool=reference_catalog(json.loads((a.source_run/'annotated_pairs.json').read_text()),json.loads((a.heldout_run/'evaluation_pairs.json').read_text()))
    a.output.mkdir(parents=True);(a.output/'references').mkdir();refs=[];requests=[];image_hash_roles={};source_data=[];input_keys=set();duplicates=[]
    for split in ['train','dev','test']:
        for item in sorted(pool[split].values(),key=lambda r:(r['category'],r['product_id'],r['sample_id'])):
            source=Path(item['image_path'])
            if item['license']!='CC BY 4.0':raise ValueError('ABO source license')
            with Image.open(source) as image:
                image=image.convert('RGB');rgb=np.asarray(image);h,w=rgb.shape[:2]
                if any(rgb[y,x].min()<245 for y,x in [(0,0),(0,w-1),(h-1,0),(h-1,w-1)]):raise ValueError('Reference not white')
                raw_sha=hashlib.sha256(image.tobytes()).hexdigest()
                if raw_sha in image_hash_roles and image_hash_roles[raw_sha]!=split:raise ValueError('Reference pixels cross splits')
                image_hash_roles[raw_sha]=split
                rid='ref-'+hashlib.sha256(item['sample_id'].encode()).hexdigest()[:12];name='references/'+rid+'.png';image.save(a.output/name)
            ref={'reference_id':rid,'split':split,'category':item['category'],'image':name,'image_sha256':digest(a.output/name),
                 'RGB_sha256':raw_sha,'source_product_id':item['product_id'],'source_family_id':item.get('family_id',item['product_id']),'source_sample_id':item['sample_id'],
                 'license':item['license'],'source_url':item['source_url'],'source_member':item['source_member'],
                 'original_image_sha256':digest(source),'source_path':str(source),'input_resolution':[w,h]}
            refs.append(ref);source_data.append({k:ref[k] for k in ['reference_id','source_product_id','source_sample_id','license','source_member']})
            for condition in condition_subset(item['category'],split):
                for seed_index in ([0,1] if split=='train' else [0]):
                    sample_id=rid+'-'+condition['condition_id']+'-seed'+str(seed_index)
                    key=(raw_sha,condition['prompt'],request_seed(raw_sha,seed_index))
                    if key in input_keys:duplicates.append(sample_id);continue
                    input_keys.add(key)
                    requests.append({'sample_id':sample_id,'split':split,'reference_id':rid,'reference':name,
                       'reference_sha256':ref['image_sha256'],'prompt':condition['prompt'],'condition_id':condition['condition_id'],
                       'category':item['category'],'attributes':condition['attributes'],'seed_index':seed_index,
                       'seed':request_seed(raw_sha,seed_index),'training_label_status':'not_generated' if split=='train' else 'not_applicable_unique_GT_not_required'})
    counts=Counter(r['split'] for r in requests);assert counts['test']==500
    for split in ['train','dev','test']:
        selected=[r for r in requests if r['split']==split]
        # The model-facing whitelist deliberately excludes SKU, masks and GT.
        inputs=[{k:r[k] for k in ['sample_id','reference','prompt','seed']} for r in selected]
        write(a.output/(split+'_inputs.json'),inputs)
    write(a.output/'references.json',refs);write(a.output/'requests.json',requests)
    # Pilot: first four products/category, one view/product, four colours, two seeds.
    pilot_refs=[]
    for category in ['lamp','pillow']:
        first={}
        for ref in refs:
            if ref['split']=='train' and ref['category']==category and any(r['reference_id']==ref['reference_id'] for r in requests):first.setdefault(ref['source_product_id'],ref)
        pilot_refs.extend(list(first.values())[:4])
    pilot=[]
    for ref in pilot_refs:
        available=[r for r in requests if r['reference_id']==ref['reference_id']]
        for i,color in enumerate(COLORS):
            detail=(['fabric','matte metal'] if ref['category']=='lamp' else ['plain woven fabric','broad cream stripes'])[i%2]
            cond=next(r['condition_id'] for r in available if r['attributes']['product_color']==color and r['attributes']['product_detail']==detail)
            pilot.extend(r for r in available if r['condition_id']==cond)
    assert len(pilot)==64
    write(a.output/'pilot_train_requests.json',pilot)
    # Source contact sheets are dataset preparation, never TEST model outputs.
    for split in ['train','dev','test']:
        shown={}
        for ref in refs:
            if ref['split']==split:shown.setdefault(ref['source_product_id'],ref)
        images=list(shown.values());sheet=Image.new('RGB',(8*144,((len(images)+7)//8)*166),'white');d=ImageDraw.Draw(sheet)
        for i,ref in enumerate(images):
            x=(i%8)*144;y=(i//8)*166;d.text((x+3,y+3),ref['reference_id'][-8:]+' '+ref['category'],fill='black')
            with Image.open(a.output/ref['image']) as im:sheet.paste(im.resize((144,144)),(x,y+20))
        sheet.save(a.output/(split+'_white_sources.png'))
    audit={'role_disjoint':True,'reference_pixels_disjoint':True,'generation_or_features_performed':False,
       'unique_GT_required_at_TEST':False,'counts':dict(counts),'inventories':{s:{'products':len({r['source_product_id'] for r in refs if r['split']==s}),
         'views':sum(r['split']==s for r in refs)} for s in ['train','dev','test']},
       'TEST_interpretation':'500 distinct reference+prompt requests from10 heldout products, not500 products; no predictions/selection',
       'test_exposure_audit_inherited_from':str(a.heldout_run/'report.json'),'test_exposure_report_sha256':digest(a.heldout_run/'report.json'),
       'train_prompt_product_support':{c:min(len({r['source_product_id'] for r in refs if r['split']=='train' and r['category']==c}),999) for c in ['lamp','pillow']},
       'only_white_ABO_sources_read':True,'old_scene_target_or_mask_pixels_read':False}
    audit['duplicate_forward_inputs_removed']=duplicates
    write(a.output/'data_audit.json',audit)
    write(a.output/'report.json',{'complete':True,'schema':SCHEMA,'stage':'request_preparation_only','source_commit':commit,'argv':sys.argv,
       'source_sha256':digest(Path(__file__)),'request_sha256':digest(a.output/'requests.json'),'audit':audit,
       'accepted_training_targets':0,'pilot_candidates_to_generate':64,'heldout_source_metadata_read':True,'TEST_model_inference':False})
    print(json.dumps(audit))


if __name__=='__main__':main()
