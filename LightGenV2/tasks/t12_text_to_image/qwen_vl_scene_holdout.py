"""Prepare sealed rich TEST540; no model inference, training or feature cache.

18 original product views appear in 10 independently seeded reference scenes
each. Each reference scene supports three balanced edits: 180 cases per mode.
This is 10 products/18 views/180 reference scenes, not 540 independent products.
"""
import argparse
import collections
import json
from pathlib import Path
import random

from .qwen_vl_holdout import exposure_inventory,reject_exposure,raw_products
from .qwen_vl_scene_compose import description,pick_conditions,scene_text,table_for
from .qwen_vl_scene_render import render_pair
from .qwen_vl_turbo_data import seed,sha256,write_json


def contexts(pool,source_id,count=10):
    if len(pool)<count or {r['split'] for r in pool}!={'test'}:raise ValueError('Sealed TEST assets required')
    return random.Random(seed(source_id,'heldout-reference-contexts')).sample(pool,count)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','baseline','training-data','review','backgrounds','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite a heldout cohort')
    cfg=json.loads(a.review.read_text());root=Path.cwd()
    original=root/'LightGenV2/tasks/t12_text_to_image'/cfg['retain_original_TEST_profile']
    if sha256(original)!=cfg['original_TEST_profile_sha256']:raise ValueError('Reviewed original TEST identities differ')
    selection=json.loads(original.read_text())['products']+cfg['additional_TEST_products']
    inventory={r['sample_id']:r for r in raw_products(a.raw)};products=[];descriptions={}
    for spec in selection:
        for view in spec['views']:
            sid=f"{spec['category']}-{spec['product_id']}-{view:02d}";product=inventory[sid]
            products.append(product);descriptions[sid]=description(spec['category'],spec['description'])
    if len(products)!=18 or len({r['product_id'] for r in products})!=10:raise ValueError('Reviewed TEST budget differs')
    previous,exposure=exposure_inventory(a.baseline,root)
    prior_data=json.loads((a.training_data/'annotated_pairs.json').read_text());previous+=prior_data
    excluded=reject_exposure(products,previous)
    bg_report=json.loads((a.backgrounds/'report.json').read_text());manifest=a.backgrounds/'background_manifest.json'
    if not bg_report['complete'] or not bg_report['heldout_generated'] or sha256(manifest)!=bg_report['manifest_sha256']:raise ValueError('Incomplete independent background assets')
    pool=json.loads(manifest.read_text())
    if {r['split'] for r in pool}!={'test'}:raise ValueError('TRAIN/DEV background prohibited')
    exposed_bg={row[role]['seed'] for row in prior_data for role in ('source_background_asset','target_background_asset')}
    if {row['seed'] for row in pool}&exposed_bg:raise ValueError('TEST background seed overlap')
    canonical={}
    for product in sorted(products,key=lambda r:r['sample_id']):canonical.setdefault(product['product_id'],product)
    rows=[]
    for source in sorted(products,key=lambda r:r['sample_id']):
        targets=[r for r in canonical.values() if r['category']==source['category'] and r['product_id']!=source['product_id']]
        random.Random(seed(source['sample_id'],'heldout-target-order')).shuffle(targets)
        for j,sbg in enumerate(contexts(pool,source['sample_id'])):
            tbg=pick_conditions(sbg,pool,10,source['sample_id']+':'+str(j))[j]
            noise=seed(1042,source['sample_id'],j,'heldout-generator')%(2**31)
            for mode in ('background','object','joint'):
                target=source if mode=='background' else targets[j%len(targets)]
                background=sbg if mode=='object' else tbg;desc=descriptions[target['sample_id']]
                scene=scene_text(background)
                if mode=='background':prompt=f"Keep this {source['category']} and replace the background with {scene}."
                elif mode=='object':prompt=f"Replace this {source['category']} with {desc}. Keep the background."
                else:prompt=f"Generate a new product scene with {desc} in {scene}."
                rows.append({'sample_id':f"test:rich:{source['sample_id']}:{mode}:{j}",'split':'test','category':source['category'],
                    'mode':mode,'source':source,'target':target,'target_description':desc,'prompt':prompt,
                    'render_protocol':'rich_product_scene_v1','source_background_asset':sbg,'source_table_asset':table_for(sbg,pool),
                    'target_background_asset':background,'target_table_asset':table_for(background,pool),'noise_seed':noise})
    assert len(rows)==540 and collections.Counter(r['mode'] for r in rows)=={'background':180,'object':180,'joint':180}
    a.output.mkdir(parents=True);files={};inputs=[];reference_hashes=set();identities=set()
    for i,row in enumerate(rows):
        for role,image in zip(('reference','target'),render_pair(row)):
            path=a.output/'images'/role/f'{i:04d}.png';path.parent.mkdir(exist_ok=True,parents=True);image.save(path)
            files[str(path.relative_to(a.output))]=sha256(path)
        reference_path=a.output/'images'/'reference'/f'{i:04d}.png';reference_digest=sha256(reference_path)
        reference_hashes.add(reference_digest);identity=(reference_digest,row['prompt'])
        if identity in identities:raise ValueError('Duplicate RGB+prompt test condition')
        identities.add(identity)
        inputs.append({'image':str(reference_path.resolve()),'prompt':row['prompt'],'sample_id':row['sample_id'],'category':row['category'],'mode':row['mode']})
    if len(reference_hashes)!=180:raise ValueError('TEST reference-context diversity differs')
    write_json(a.output/'evaluation_pairs.json',rows);write_json(a.output/'inference_inputs.json',inputs)
    write_json(a.output/'image_sha256.json',files)
    write_json(a.output/'report.json',{'complete':True,'TRAIN':0,'DEV':0,'TEST':540,'modes':{'background':180,'object':180,'joint':180},
        'products':10,'original_views':18,'unique_reference_scenes':180,'unique_RGB_prompt_conditions':len(identities),
        'only_asset_preparation':True,'feature_cache_created':False,'model_inference':False,
        'no_gradients_or_selection':True,'evaluation_pairs_sha256':sha256(a.output/'evaluation_pairs.json'),
        'input_manifest_sha256':sha256(a.output/'inference_inputs.json'),'review_sha256':sha256(a.review),
        'background_manifest_sha256':sha256(manifest),'training_pairs_sha256':sha256(a.training_data/'annotated_pairs.json'),
        'exposure_inventory':exposure,'exposed_product_family_pixel_counts':excluded,
        'limitations':'Repeated product views across independently seeded scenes are clustered observations. Not 540 independent goods. Public pretraining exposure unknown. Freeze final models/protocol before inference; never use this cohort for selection.'})


if __name__=='__main__':main()
