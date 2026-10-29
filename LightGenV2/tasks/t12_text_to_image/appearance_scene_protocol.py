"""Appearance-preserving furnished scenes: new no-recolor protocol, not rescoring old requests."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from .scene_generation_data import LAMP_SCENES,PILLOW_SCENES,request_seed
from .white_reference_scene_probe import digest,write

PROFILE='appearance_preserving_room_v1'
COARSE_PROFILE='coarse_shape_furnished_room_v2'


def conditions(category,profile=PROFILE):
    if category not in ['lamp','pillow']:raise ValueError('Approved categories only')
    subject='table lamp' if category=='lamp' else 'cushion';out=[]
    for i,(scene,support,props) in enumerate(LAMP_SCENES if category=='lamp' else PILLOW_SCENES):
        support=support.replace('a beige sofa','a sofa').replace('a grey armchair','an armchair').replace('a grey loveseat','a loveseat')
        prompt=(f'Generate a realistic product photograph of one reference {subject} {support} in a furnished {scene}. '
                f'Retain the product\'s overall shape and original appearance. Include {props} behind it. '
                f'Keep the {subject} clearly visible as the main foreground product.')
        caption=(f'Close-up product photograph of one {subject} {support} in a furnished {scene}, with {props}. '
                 'The reference product is the main foreground subject, with its original shape and appearance.')
        compact=(f'A {subject} {support} in a furnished {scene} with {props}.').replace('framed abstract shapes','art').replace('a small plant','a plant')
        out.append({'condition_id':str(i),'prompt_profile':PROFILE,'category':category,'prompt':prompt,'native_caption':caption,
          'reward_caption':compact,'requested_attributes':{'retain_coarse_shape':True,'retain_original_appearance':True,
           'change_product_color':False,'scene':scene,'support':support,'background_elements':props}})
    if profile==COARSE_PROFILE:
        for row in out:
            row['prompt_profile']=profile
            row['prompt']=row['prompt'].replace("Retain the product's overall shape and original appearance.",
                "Retain the product's overall shape; fine details may vary naturally.")
            row['native_caption']=row['native_caption'].replace('with its original shape and appearance.',
                'with its overall shape retained; fine details may vary naturally.')
            row['requested_attributes']['retain_original_appearance']=False
            row['requested_attributes']['fine_details_free']=True
    elif profile!=PROFILE:raise ValueError('Unreviewed scene contract')
    return out


def seed_indices(split,scene_index):
    if split=='train':return [0,1]
    if split=='dev':return [0]
    if split=='test':return list(range(7 if scene_index<2 else 6))
    raise ValueError('Unknown split')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    commit=os.environ.get('T12_SOURCE_COMMIT','')
    if a.output.exists() or len(commit)!=40:raise ValueError('New dataset/published source required')
    old=json.loads((a.source_data/'report.json').read_text());audit=json.loads((a.source_data/'data_audit.json').read_text())
    assert old['complete'] and audit['role_disjoint'] and not audit['generation_or_features_performed']
    refs=json.loads((a.source_data/'references.json').read_text());a.output.mkdir(parents=True);(a.output/'references').mkdir()
    requests=[]
    for ref in refs:
        assert digest(a.source_data/ref['image'])==ref['image_sha256'];shutil.copyfile(a.source_data/ref['image'],a.output/ref['image'])
        for scene in conditions(ref['category']):
            for index in seed_indices(ref['split'],int(scene['condition_id'])):
                requests.append({**scene,'sample_id':ref['reference_id']+'-room'+scene['condition_id']+'-seed'+str(index),
                  'split':ref['split'],'reference_id':ref['reference_id'],'reference':ref['image'],'reference_sha256':ref['image_sha256'],
                  'seed_index':index,'seed':request_seed(ref['RGB_sha256'],index)})
    assert len({(r['reference'],r['prompt'],r['seed']) for r in requests})==len(requests)
    counts=Counter(r['split'] for r in requests);assert counts=={'train':1216,'dev':128,'test':500},counts
    old_pilot={r['reference_id'] for r in json.loads((a.source_data/'pilot_train_requests.json').read_text())}
    pilot=[r for r in requests if r['split']=='train' and r['reference_id'] in old_pilot]
    assert len(pilot)==128
    write(a.output/'references.json',refs);write(a.output/'requests.json',requests);write(a.output/'pilot_train_requests.json',pilot)
    for split in ['train','dev','test']:
        write(a.output/(split+'_inputs.json'),[{k:r[k] for k in ['sample_id','reference','prompt','seed']} for r in requests if r['split']==split])
    new_audit={'role_disjoint_inherited':True,'source_data_report_sha256':digest(a.source_data/'report.json'),
      'counts':dict(counts),'TEST_distinct_reference_prompt_contexts':80,'TEST_stochastic_draws':500,
      'TEST_products':10,'TEST_model_inference':False,'unique_GT_used':False,'new_color_requests':False,
      'old_color_protocol_preserved':True,'meaning':'500 draws from80 held-out reference+scene contexts, not500 distinct products/prompts.'}
    write(a.output/'data_audit.json',new_audit)
    write(a.output/'report.json',{'complete':True,'prompt_profile':PROFILE,'source_commit':commit,'source_sha256':digest(Path(__file__)),
      'request_sha256':digest(a.output/'requests.json'),'audit':new_audit,'stage':'request_preparation_only','accepted_training_targets':0})
    print(json.dumps(new_audit))


if __name__=='__main__':main()
