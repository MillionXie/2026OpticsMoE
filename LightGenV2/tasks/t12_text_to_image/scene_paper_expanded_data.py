"""Single-seed TRAIN expansion and a user-authorized TEST-for-selection cohort.

No validation split. TEST has no target image and never enters gradients.
Request preparation is not target generation, semantic approval, or training.
Historical v7 datasets and runs remain immutable.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from .scene_generation_data import request_seed
from .scene_instruction_data import SAFE, sha
from .scene_paper_prompt_protocol import condition as original_condition, write
from .scene_paper_paired_data import SOURCE_CATALOG_SHA

PROFILE = 'two_sentence_product_scene_expanded_v8'
SCHEMA = 'scene_paper_train_test_single_seed_v1'


def conditions(category):
    """Twenty different support/anchor scenes, not extra draws or paraphrases."""
    rows = [original_condition(category, i) for i in range(3)]
    if category == 'lamp':
        supports = [('a wooden desk', 'wooden desk'), ('a bedside table', 'bedside table'),
                    ('a wooden console', 'wooden console'), ('a side table', 'side table')]
        anchors = [('a bookcase against the wall', 'bookcase'), ('a bed in the background', 'bed'),
                   ('a curtained window in the background', 'curtained window'),
                   ('a fabric sofa in the background', 'fabric sofa'),
                   ('an armchair beside the table', 'armchair')]
        subject = 'a table lamp'
    elif category == 'pillow':
        supports = [('a fabric sofa', 'fabric sofa'), ('an armchair', 'armchair'),
                    ('a bed', 'bed'), ('a wooden bench', 'wooden bench')]
        anchors = [('a leafy floor plant beside the seating area', 'leafy floor plant'),
                   ('a bookcase against the wall', 'bookcase'),
                   ('a curtained window in the background', 'curtained window'),
                   ('a wooden side table beside the furniture', 'wooden side table'),
                   ('a framed picture on the wall', 'framed picture')]
        subject = 'a cushion'
    else:
        raise ValueError('Only the original lamp/pillow task is authorized')
    for support, support_fact in supports:
        for anchor, anchor_fact in anchors:
            prompt = f'Generate an interior photograph with {subject} resting on {support}. Include {anchor}.'
            rows.append(dict(prompt=prompt, native_caption=prompt, reward_caption=f'{subject} on {support} with {anchor}.',
                             required_visible_facts=[subject, support_fact+' supports subject', anchor_fact]))
    rows = rows[:20]
    for i, row in enumerate(rows):
        row.update(category=category, condition_id=str(i), prompt_profile=PROFILE)
        if row['prompt'].count('.') != 2 or len(row['prompt'].split()) > 32:
            raise ValueError('Two concise informative sentences required')
    if len({r['prompt'] for r in rows}) != 20:
        raise ValueError('Repeated prompts are not additional data')
    return rows


def make_requests(references):
    owners = {}
    for ref in references:
        if ref['split'] not in ['train', 'test']:
            raise ValueError('No validation split in this user-approved protocol')
        for key in ['source_product_id', 'source_family_id', 'reference_sha256']:
            if owners.setdefault((key, ref[key]), ref['split']) != ref['split']:
                raise ValueError('TRAIN/TEST product, family or pixel leakage')
    if Counter((r['split'], r['category']) for r in references) != {
            ('train','lamp'):36, ('train','pillow'):40, ('test','lamp'):13, ('test','pillow'):13}:
        raise ValueError('Use all original76 TRAIN views and26 non-TRAIN views')
    groups = defaultdict(list)
    for ref in sorted(references, key=lambda r:(r['split'],r['category'],r['reference_id'])):
        for definition in conditions(ref['category']):
            row = {k:ref[k] for k in ['reference_id','reference','reference_sha256','category']}
            row.update({k:definition[k] for k in ['prompt','native_caption','reward_caption','condition_id','prompt_profile']})
            row.update(sample_id=ref['reference_id']+'-expanded-scene'+definition['condition_id']+'-seed0',
                       seed=request_seed(ref['reference_sha256'],0))
            if set(row) != set(SAFE): raise ValueError('Generator input whitelist changed')
            groups[ref['split']].append(row)
    # 260/category ->250/category, uniformly omit one scene for ten sources.
    # Deterministic choice precedes any predictions; no quality-based exclusions.
    test=[]
    for category in ['lamp','pillow']:
        refs=sorted(r['reference_id'] for r in references if r['split']=='test' and r['category']==category)
        omissions={(rid,str(i%20)) for i,rid in enumerate(refs[:10])}
        test.extend(r for r in groups['test'] if r['category']==category and (r['reference_id'],r['condition_id']) not in omissions)
    if len(groups['train'])!=1520 or len(test)!=500:
        raise ValueError('Expected1520 TRAIN candidates and500 unique TEST requests')
    return groups['train'],test


def prepare(sources,white,original,output):
    if output.exists(): raise ValueError('Never overwrite a prior dataset')
    if sha(sources)!=SOURCE_CATALOG_SHA: raise ValueError('Use the fixed TRAIN/oldDEV source inventory')
    catalog=json.loads(sources.read_text(encoding='utf8'))
    oldrefs=json.loads((original/'references.json').read_text(encoding='utf8'))
    by_sample={r['source_sample_id']:r for r in oldrefs}
    references=[];paths={}
    for row in catalog:
        prior=by_sample[row['sample_id']]
        if prior['split']!=row['split'] or prior['source_product_id']!=row['product_id'] or row['license']!='CC BY 4.0':
            raise ValueError('Original role or attribution changed')
        rid=row['sample_id'];name='references/'+rid+'.png';path=white/row['white_reference']
        if sha(path)!=row['white_reference_sha256']: raise ValueError('Original white RGB changed')
        references.append(dict(reference_id=rid, reference=name, reference_sha256=sha(path), category=row['category'],
            split='train' if row['split']=='train' else 'test', original_role=row['split'],
            source_product_id=row['product_id'], source_family_id=row['family_id'], source_RGBA_sha256=row['rgba_file_sha256'],
            license=row['license'], source_url=row['source_url'], source_member=row['source_member']))
        paths[name]=path
    for row in oldrefs:
        if row['split']!='test': continue
        path=original/row['image']
        if sha(path)!=row['image_sha256'] or row['license']!='CC BY 4.0': raise ValueError('Historical TEST source bytes changed')
        references.append(dict(reference_id=row['reference_id'],reference=row['image'],reference_sha256=sha(path),category=row['category'],
            split='test',original_role='historical_test_already_evaluated_by_older_models',source_product_id=row['source_product_id'],
            source_family_id=row['source_family_id'],license=row['license'],source_url=row['source_url'],source_member=row['source_member']))
        paths[row['image']]=path
    train,test=make_requests(references)
    output.mkdir(parents=True);(output/'references').mkdir()
    for name,path in paths.items(): shutil.copyfile(path,output/name)
    write(output/'references.json',references)
    write(output/'train_inputs.json',train);write(output/'test_inputs.json',test);write(output/'inference_inputs.json',test)
    write(output/'prompt_definitions.json',[r for category in ['lamp','pillow'] for r in conditions(category)])
    selection=dict(schema='scene_test_PT_selection_rule_v1',validation_set=False,TEST_used_for_model_selection=True,
        independent_generalization_claim=False,gradient_roles=['train'],test_count=500,
        primary='macro_mean_ImageReward_compact',tie_breaker='macro_mean_CLIP_T_then_checkpoint_SHA',
        complete_cohort_required=True,all_candidates_share_same_requests=True,no_unique_test_target_required=True,
        test_request_sha256=sha(output/'test_inputs.json'),rule_frozen_before_new_predictions=True)
    write(output/'selection_protocol.json',selection)
    write(output/'report.json',dict(schema=SCHEMA,complete=True,prompt_profile=PROFILE,TRAIN_candidate_requests=1520,
        TRAIN_views=76,TRAIN_products=44,TEST_requests=500,TEST_views=26,TEST_products=18,TEST_categories={'lamp':250,'pillow':250},
        validation_set=False,TEST_used_for_model_selection=True,independent_generalization_claim=False,role_disjoint=True,
        one_seed_per_reference=True,one_output_per_RGB_prompt=True,source_catalog_sha256=sha(sources),
        original_reference_manifest_sha256=sha(original/'references.json'),TRAIN_request_sha256=sha(output/'train_inputs.json'),
        request_sha256=sha(output/'test_inputs.json'),accepted_training_pairs=0,targets_created=False,eligible_for_training=False,
        generation_or_scoring_performed=False,former_DEV_sources_now_test=True,historical_TEST_sources_previously_evaluated=True,
        inputs_white_ABO_only=True,external_background_photographs_read=False,old_dataset_or_results_relabelled=False))


def select_checkpoint(requests,scored_candidates):
    """Use every500 request for each PT; no unique GT and no cherry-picking."""
    if len(requests)!=500 or Counter(r['category'] for r in requests)!={'lamp':250,'pillow':250}:
        raise ValueError('Complete fixed balanced TEST500 required')
    expected={r['sample_id']:r for r in requests}
    if len(expected)!=500: raise ValueError('Repeated test requests')
    results=[]
    for checkpoint_sha,rows in scored_candidates.items():
        if len(checkpoint_sha)!=64 or any(c not in '0123456789abcdef' for c in checkpoint_sha):raise ValueError('Actual PT SHA required')
        if len(rows)!=500 or len({r['sample_id'] for r in rows})!=500 or set(expected)!={r['sample_id'] for r in rows}:
            raise ValueError('Score every request for every candidate checkpoint')
        for r in rows:
            e=expected[r['sample_id']]
            if (r['input_sha256'],r['prompt'],r['seed'],r['category'])!=(e['reference_sha256'],e['prompt'],e['seed'],e['category']):
                raise ValueError('Changed source/prompt/seed in PT comparison')
            if r['checkpoint_sha256']!=checkpoint_sha or any(not math.isfinite(float(r[k])) for k in ['ImageReward_compact','CLIP_T']):
                raise ValueError('Unbound checkpoint or nonfinite scores')
        def macro(metric):
            return sum(sum(float(r[metric]) for r in rows if r['category']==c)/250 for c in ['lamp','pillow'])/2
        results.append(dict(checkpoint_sha256=checkpoint_sha,macro_ImageReward_compact=macro('ImageReward_compact'),macro_CLIP_T=macro('CLIP_T')))
    if not results: raise ValueError('No scored candidate checkpoints')
    results.sort(key=lambda r:(-r['macro_ImageReward_compact'],-r['macro_CLIP_T'],r['checkpoint_sha256']))
    return dict(selected=results[0],candidates=results,TEST_used_for_model_selection=True,independent_generalization_claim=False,
                validation_set=False,proxies_not_attribute_success_rates=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['sources','white','original','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.sources,a.white,a.original,a.output)


if __name__=='__main__':main()
