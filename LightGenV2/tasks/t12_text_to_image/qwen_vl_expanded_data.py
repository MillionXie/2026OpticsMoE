"""Clear-view screening and Qwen annotations for the revised 960/96 protocol.

Original TRAIN and VAL inventories stay immutable. The latter supplies a small
development TEST used for selection; it is not an independent test claim.
"""
import argparse
import collections
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .qwen_vl_turbo_data import _pixels_hash, normalized_product, sha256, write_json


def view_quality(row):
    with Image.open(row['image_path']) as im:rgb=np.asarray(im.convert('RGB'),dtype=np.float32)
    with Image.open(row['mask_path']) as im:alpha=np.asarray(im.convert('L'),dtype=np.float32)/255
    support=alpha>.1;y,x=np.where(support)
    if not len(x):return None
    h,w=alpha.shape;x0,x1=int(x.min()),int(x.max())+1;y0,y1=int(y.min()),int(y.max())+1
    bw,bh=x1-x0,y1-y0;fill=float(alpha[y0:y1,x0:x1].mean());area=float(alpha.mean())
    border=float(np.concatenate([alpha[0],alpha[-1],alpha[:,0],alpha[:,-1]]).mean())
    if area<.015 or min(bw,bh)<24 or border>.002 or min(x0,y0,w-x1,h-y1)<2:return None
    aspect=bw/bh
    if row['category']=='lamp' and (not .4<aspect<1.2 or fill<.5):return None
    if row['category']=='pillow' and not .7<aspect<2.3:return None
    gray=rgb.mean(2);edge_x=np.abs(np.diff(gray,axis=1));edge_y=np.abs(np.diff(gray,axis=0))
    interior_x=support[:,:-1]&support[:,1:];interior_y=support[:-1]&support[1:]
    edges=[edge_x[interior_x],edge_y[interior_y]]
    sharp=float(np.concatenate(edges).mean()) if any(len(v) for v in edges) else 0.
    # Lamps need a visible base, so prefer tall frontal projections rather than
    # the high-fill overhead shade views rejected in the first visual review.
    score=(.7*fill-.65*abs(math.log(aspect/.55))+.03*math.log1p(sharp) if row['category']=='lamp' else
           .7*fill+.1*min(area/.3,1)+.18*math.log1p(sharp)-.08*abs(math.log(max(aspect,.01))))
    histogram=np.concatenate([np.histogram(rgb[:,:,c][support],bins=24,range=(0,256))[0] for c in range(3)]).astype(float)
    histogram/=max(histogram.sum(),1)
    return {'score':score,'fill':fill,'alpha_area':area,'aspect':aspect,'foreground_edge':sharp,
            'border_alpha':border,'bbox':[x0,y0,x1,y1],'foreground_color_histogram':histogram.tolist()}


def screen(data,out,parent_data,excluded=()):
    if out.exists():raise ValueError('Existing screen must be inspected, not overwritten')
    source={};hashes={}
    for split in ('train','val'):
        path=data/(split+'.jsonl');hashes[split]=sha256(path);groups=collections.defaultdict(list)
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['category'] not in ('lamp','pillow'):continue
            if r['sequence_id'] in excluded:continue
            if r['license']!='CC BY 4.0':raise ValueError('Unreviewed source license')
            for key in ('image_path','mask_path'):r[key]=str((data/r[key]).resolve())
            r['product_id']=r['sequence_id'];r['family_id']=r.get('parent_id') or r.get('family_id') or r['product_id']
            quality=view_quality(r)
            if quality is not None:
                r['rgba_sha256']=_pixels_hash(r);r['quality']=quality;groups[r['category'],r['product_id']].append(r)
        source[split]=groups
    selected={};quality_rows=[]
    for split,groups in source.items():
        selected[split]=[]
        for cat in ('lamp','pillow'):
            products=[]
            for (category,sku),views in groups.items():
                if category!=cat:continue
                ordered=sorted(views,key=lambda r:(-r['quality']['score'],r['sample_id']))
                unique=[];seen=set()
                for row in ordered:
                    if row['rgba_sha256'] not in seen:unique.append(row);seen.add(row['rgba_sha256'])
                if unique and cat=='pillow':
                    anchor=unique[0]['quality'];threshold=anchor['foreground_edge']*.5 if anchor['foreground_edge']>.5 else 0
                    hist=np.asarray(anchor['foreground_color_histogram'])
                    unique=[r for r in unique if r['quality']['foreground_edge']>=threshold and
                            np.sqrt(hist*np.asarray(r['quality']['foreground_color_histogram'])).sum()>=.92]
                if len(unique)<2:continue
                chosen=unique[:2];products.append((min(r['quality']['score'] for r in chosen),sku,chosen))
            need=16 if split=='train' else 4
            if len(products)<need:raise ValueError(f'Only {len(products)} eligible {split}/{cat} products; need {need}; never pad with clipped views')
            # Both roles use the same input-only quality rule, never predictions
            # or target metrics. A human reviews all chosen views before caching.
            products=sorted(products,key=lambda p:(-p[0],p[1]))[:need]
            for _,sku,chosen in products:
                selected[split].append({'category':cat,'product_id':sku,
                    'views':[int(r['sample_id'].rsplit('-',1)[1]) for r in chosen],
                    'description':'','source_rows':chosen})
                quality_rows.extend({'split':split,'sample_id':r['sample_id'],**r['quality']} for r in chosen)
    from .qwen_vl_curated import prior_exposure,reviewed_pairs
    descriptions={r['sample_id']:'a clear complete '+r['category'] for v in selected.values() for p in v for r in p['source_rows']}
    profile={'version':'qwen_vl_expanded_lamp_pillow_960_96_20261027','seed':1042,
      'categories':['lamp','pillow'],'image_size':256,'requires_parent_exposure_audit':True,
      'products':{s:[{k:v for k,v in p.items() if k!='source_rows'} for p in products] for s,products in selected.items()},
      'excluded_products':list(excluded),'edits_per_mode':{'train':5,'val':2},'manual_prompt_descriptions':False,
      'selection_basis':'Alpha boundary/completeness and foreground face/edge screening followed by contact-sheet visual review; never ranked by model predictions.',
      'target_view_policy':'First screened view is canonical target; both clear views may be references. Different input scenes per repeated target.',
      'evaluation_label':'development_test','evaluation_source_role':'val',
      'test_selection_authorized':True,'independent_test_claim':False}
    rows=reviewed_pairs({s:[r for p in v for r in p['source_rows']] for s,v in selected.items()},descriptions,profile)
    prior=prior_exposure(rows,json.loads(parent_data.read_text(encoding='utf-8')))
    if dict(collections.Counter(r['split'] for r in rows))!={'train':960,'val':96}:raise ValueError('Revised pair budget differs')
    out.mkdir(parents=True)
    write_json(out/'screened_products.json',selected);write_json(out/'screen_profile.json',profile)
    for split,products in selected.items():
        for cat in ('lamp','pillow'):
            group=[p for p in products if p['category']==cat]
            for start in range(0,len(group),16):
                page=group[start:start+16];sheet=Image.new('RGB',(1280,190*math.ceil(len(page)/4)),'white');draw=ImageDraw.Draw(sheet)
                for i,p in enumerate(page):
                    x=(i%4)*320;y=(i//4)*190;draw.text((x+3,y+3),p['product_id'],fill='black')
                    for j,r in enumerate(p['source_rows']):
                        fg,mask=normalized_product(r,256);im=Image.composite(fg,Image.new('RGB',(256,256),'white'),mask);im.thumbnail((155,155))
                        sheet.paste(im,(x+j*160,y+30));draw.text((x+j*160+3,y+17),r['sample_id'].rsplit('-',1)[1],fill='black')
                sheet.save(out/f'{split}_{cat}_screen_{start//16:02d}.png')
    write_json(out/'screen_report.json',{'complete':True,'source_sha256':sha256(Path(__file__)),
      'raw_manifest_sha256':hashes,'parent_manifest_sha256':sha256(parent_data),'parent_exposure':prior,
      'products':{s:len(v) for s,v in selected.items()},'views':{s:sum(len(p['views']) for p in v) for s,v in selected.items()},
      'pairs':dict(collections.Counter(r['split'] for r in rows)),'view_quality':quality_rows,
      'evaluation_scope':'Original VAL subset used as development TEST for selection; no independent TEST claim',
      'visual_review_complete':False})
    print('Screen prepared; visual contact-sheet review required before annotation',flush=True)


def annotate(screen_dir,qwen,gpu):
    review=json.loads((screen_dir/'visual_review.json').read_text())
    if not review['passed'] or review['screen_profile_sha256']!=sha256(screen_dir/'screen_profile.json'):raise ValueError('Visual review identity mismatch')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('Unauthorized GPU')
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(4);torch.manual_seed(1042);torch.cuda.set_per_process_memory_fraction(.65,0)
    processor=AutoProcessor.from_pretrained(qwen,local_files_only=True,min_pixels=65536,max_pixels=65536)
    model=Qwen3VLForConditionalGeneration.from_pretrained(qwen,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
    profile=json.loads((screen_dir/'screen_profile.json').read_text());selected=json.loads((screen_dir/'screened_products.json').read_text())
    annotations={};instruction='Describe only this central product in one noun phrase of at most 32 words. State category, shape, colors and visible material or pattern. No brand, model number, background, speculation or camera instruction. Do not describe missing or invisible details.'
    for split,products in selected.items():
        for index,p in enumerate(products):
            r=p['source_rows'][0];fg,mask=normalized_product(r,256);rgb=Image.composite(fg,Image.new('RGB',(256,256),'white'),mask)
            messages=[{'role':'user','content':[{'type':'image','image':rgb},{'type':'text','text':instruction}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            inputs=processor(text=[text],images=[rgb],return_tensors='pt').to('cuda')
            with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=64,do_sample=False)
            desc=processor.batch_decode(tokens[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip().strip('"').rstrip('.')
            if not desc or re.search(r'\bB0[A-Z0-9]{8}\b|\bSKU\b|\bAmazon\b',desc,re.I):raise ValueError('Unusable product annotation '+p['product_id'])
            # Explicit category avoids ambiguous nouns without catalog identifiers.
            if p['category']=='lamp' and 'lamp' not in desc.lower():desc='a lamp, '+desc
            if p['category']=='pillow' and not any(w in desc.lower() for w in ('pillow','cushion')):desc='a cushion, '+desc
            match=next(x for x in profile['products'][split] if x['product_id']==p['product_id']);match['description']=desc
            annotations[p['product_id']]={'description':desc,'canonical_sample_id':r['sample_id'],'source_rgba_sha256':r['rgba_sha256']}
            write_json(screen_dir/'annotation_progress.json',{'stage':split,'done':index+1,'total':len(products),'updated_unix':time.time()})
            write_json(screen_dir/'product_annotations.json',annotations)
    profile['annotation_method']='Frozen full public Qwen3-VL-2B, canonical product RGB only; greedy descriptions, no brand/SKU; used only for dataset prompt construction'
    write_json(screen_dir/'annotated_profile.json',profile)
    write_json(screen_dir/'annotation_complete.json',{'complete':True,'qwen':str(qwen),'screen_profile_sha256':sha256(screen_dir/'screen_profile.json'),
      'profile_sha256':sha256(screen_dir/'annotated_profile.json'),'product_count':len(annotations),'source_sha256':sha256(Path(__file__))})


def revise(data,out,revisions):
    if (out/'manual_revision.json').exists():raise ValueError('Reviewed revision already applied')
    rules=json.loads(revisions.read_text());profile=json.loads((out/'screen_profile.json').read_text());products=json.loads((out/'screened_products.json').read_text())
    original=sha256(out/'screen_profile.json')
    for name in ('screen_profile.json','screened_products.json','screen_report.json'):
        (out/('automatic_'+name)).write_bytes((out/name).read_bytes())
    for split,groups in rules['views'].items():
        raw=[json.loads(s) for s in (data/(split+'.jsonl')).read_text().splitlines()];lookup={r['sample_id']:r for r in raw}
        products[split]=[p for p in products[split] if p['product_id'] not in rules.get('remove_products',{}).get(split,[])]
        for sku,views in groups.items():
            rows=[]
            for view in views:
                r=dict(lookup[f'lamp-{sku}-{view:02d}'])
                for k in ('image_path','mask_path'):r[k]=str((data/r[k]).resolve())
                r.update(product_id=sku,family_id=r.get('family_id') or r.get('parent_id') or sku)
                r['rgba_sha256']=_pixels_hash(r);r['quality']=view_quality(r) or {'manual_complete_frontal_view_override':True}
                with Image.open(r['mask_path']) as im:mask=np.asarray(im.convert('L'))>.1*255
                y,x=np.where(mask)
                if not len(x) or min(x.min(),y.min(),mask.shape[1]-1-x.max(),mask.shape[0]-1-y.max())<2:raise ValueError('Manual view is empty/clipped')
                rows.append(r)
            products[split]=[p for p in products[split] if p['product_id']!=sku]
            products[split].append({'category':'lamp','product_id':sku,'views':views,'description':'','source_rows':rows})
        products[split].sort(key=lambda p:(p['category'],p['product_id']))
    for split,group in products.items():
        expected=16 if split=='train' else 4
        if any(sum(p['category']==cat for p in group)!=expected for cat in ('lamp','pillow')):raise ValueError('Reviewed product quota changed')
        profile['products'][split]=[{k:v for k,v in p.items() if k!='source_rows'} for p in group]
        for cat in ('lamp','pillow'):
            chosen=[p for p in group if p['category']==cat];sheet=Image.new('RGB',(1280,190*math.ceil(len(chosen)/4)),'white');draw=ImageDraw.Draw(sheet)
            for i,p in enumerate(chosen):
                x=i%4*320;y=i//4*190;draw.text((x+3,y+3),p['product_id'],fill='black')
                for j,r in enumerate(p['source_rows']):
                    fg,alpha=normalized_product(r,256);im=Image.composite(fg,Image.new('RGB',(256,256),'white'),alpha);im.thumbnail((155,155));sheet.paste(im,(x+j*160,y+30));draw.text((x+j*160+3,y+17),str(p['views'][j]),fill='black')
            sheet.save(out/f'{split}_{cat}_reviewed.png')
    profile['manual_view_overrides_sha256']=sha256(revisions);profile['selection_basis']+=' '+rules['basis']
    write_json(out/'screen_profile.json',profile);write_json(out/'screened_products.json',products)
    write_json(out/'manual_revision.json',{'original_profile_sha256':original,'revisions_sha256':sha256(revisions),'revised_profile_sha256':sha256(out/'screen_profile.json'),'basis':rules['basis']})


def review_captions(out,revisions):
    profile=json.loads((out/'annotated_profile.json').read_text());rules=json.loads(revisions.read_text())
    annotations=json.loads((out/'product_annotations.json').read_text())
    products={p['product_id']:p for group in profile['products'].values() for p in group}
    if set(rules['descriptions'])-set(products):raise ValueError('Caption override names unselected product')
    for sku,description in rules['descriptions'].items():
        if re.search(r'\bB0[A-Z0-9]{8}\b|\bSKU\b|\bAmazon\b',description,re.I):raise ValueError('Identity leaked into caption')
        products[sku]['description']=description
    profile['annotation_method']+='; lamp descriptions corrected by visual review of original selected views; cushion descriptions retained'
    profile['caption_review_sha256']=sha256(revisions)
    write_json(out/'reviewed_profile.json',profile)
    write_json(out/'caption_review.json',{'passed':True,'original_profile_sha256':sha256(out/'annotated_profile.json'),
        'reviewed_profile_sha256':sha256(out/'reviewed_profile.json'),'overrides_sha256':sha256(revisions),'basis':rules['basis'],
        'changed_products':list(rules['descriptions']),'source_annotation_count':len(annotations),'model_outputs_used':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('screen','revise','annotate','review-captions'),required=True)
    for k in ('data','output','parent-data','qwen','revisions'):p.add_argument('--'+k,type=Path)
    p.add_argument('--gpu-uuid');p.add_argument('--exclude-product',action='append',default=[]);a=p.parse_args()
    if a.stage=='screen':screen(a.data,a.output,a.parent_data,a.exclude_product)
    elif a.stage=='revise':revise(a.data,a.output,a.revisions)
    elif a.stage=='review-captions':review_captions(a.output,a.revisions)
    else:annotate(a.output,a.qwen,a.gpu_uuid)


if __name__=='__main__':main()
