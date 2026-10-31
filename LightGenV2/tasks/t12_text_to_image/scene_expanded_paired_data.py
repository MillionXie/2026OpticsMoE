"""Reviewed expanded TRAIN targets with target-free TEST500 and no validation.

TRAIN target admission and TEST PT selection are distinct. Missed difficult
scene facts can remain explicitly downweighted TRAIN examples; broken subjects
cannot silently enter gradients. No TEST target is generated or loaded.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from .scene_instruction_data import SAFE,sha
from .scene_paper_expanded_data import PROFILE,SCHEMA,conditions
from .scene_paper_expanded_targets import checked_plan
from .scene_paper_paired_data import accepted_ids
from .scene_paper_prompt_protocol import write

PAIRED_SCHEMA='scene_expanded_train_only_pairs_v1'


def checked_candidate_rows(report,rows,expected,reviewed_prefix=False):
    ids={r['sample_id'] for r in rows}
    if len(ids)!=len(rows):raise ValueError('Repeated candidate input')
    selected=[r for r in expected if r['sample_id'] in ids]
    if reviewed_prefix:
        if(not report.get('partial_TRAIN_prefix') or report.get('original_TRAIN_request_count')!=len(expected)
           or report.get('candidate_count')!=len(rows) or not 0<len(rows)<len(expected)):
            raise ValueError('Explicit stopped partial TRAIN pool required')
    elif len(rows)!=len(expected):raise ValueError('Complete original TRAIN candidate plan required')
    if len(selected)!=len(rows) or any({k:r[k] for k in SAFE}!=e for r,e in zip(rows,selected)):
        raise ValueError('Actual candidates must retain their original public input identities and order')


def checked_fast_coverage(rows,refs):
    training=[r for r in refs if r['split']=='train'];catalog={r['reference_id']:r for r in training}
    expected_groups={(r['category'],str(i)) for r in training for i in range(len(conditions(r['category'])))}
    if(len(rows)<600 or len({r['sample_id']for r in rows})!=len(rows) or {r['reference_id'] for r in rows}!=set(catalog)
       or {(r['category'],r['condition_id']) for r in rows}!=expected_groups):
        raise ValueError('Fast admission needs at least600 usable TRAIN pairs, all source views and all public template groups')
    return dict(usable_TRAIN_pairs=len(rows),TRAIN_views=len(catalog),TRAIN_products=len({r['source_product_id']for r in training}),
                template_groups=len(expected_groups),all_source_views_and_template_groups_covered=True)


def checked_dataset(data,verify_pixels=True):
    report=json.loads((data/'report.json').read_text());review=json.loads((data/'visual_review.json').read_text())
    rows=json.loads((data/'pairs.json').read_text());test=json.loads((data/'inference_inputs.json').read_text())
    refs=json.loads((data/'references.json').read_text());catalog={r['reference_id']:r for r in refs}
    if(report.get('schema')!=PAIRED_SCHEMA or not report['complete'] or not report['eligible_for_training']
       or report['prompt_profile']!=PROFILE or report['validation_set'] or not report['TEST_used_for_model_selection']
       or not report['role_disjoint'] or report['pairs_sha256']!=sha(data/'pairs.json')
       or report['request_sha256']!=sha(data/'inference_inputs.json') or not review['approved_for_training']
       or review['pairs_sha256']!=report['pairs_sha256'] or not review['all_TRAIN_candidates_reviewed']):
        raise ValueError('Actual reviewed expanded TRAIN-only dataset required')
    if not rows or any(r['split']!='train' for r in rows) or len(test)!=500:
        raise ValueError('TRAIN targets only and complete target-free TEST500 required')
    owners={}
    for ref in refs:
        for key in ['source_product_id','source_family_id','reference_sha256']:
            if owners.setdefault((key,ref[key]),ref['split'])!=ref['split']:raise ValueError('Cross-role source identity')
    if Counter(r['category'] for r in test)!={'lamp':250,'pillow':250} or any(set(r)!=set(SAFE) for r in test):
        raise ValueError('Complete balanced public TEST500 whitelist required')
    for row in rows+test:
        source=catalog[row['reference_id']];training='target' in row
        if source['split']!=('train' if training else 'test') or row['reference_sha256']!=source['reference_sha256']:
            raise ValueError('Wrong role or RGB identity')
        definition=conditions(row['category'])[int(row['condition_id'])]
        if any(row[k]!=definition[k] for k in ['prompt','prompt_profile','native_caption','reward_caption']):
            raise ValueError('Changed canonical prompt')
        if verify_pixels and sha(data/row['reference'])!=row['reference_sha256']:raise ValueError('Changed reference pixels')
        if training:
            if(not row['QA_accepted'] or row['training_admission_policy']!='basic_subject_quality_v2'
               or row['supervision_weight'] not in [.5,1.] or row['source_product_id']!=source['source_product_id']
               or row['source_RGBA_sha256']!=source['source_RGBA_sha256']):raise ValueError('TRAIN admission identity missing')
            if verify_pixels and sha(data/row['target'])!=row['target_sha256']:raise ValueError('Changed reviewed target pixels')
    if report.get('TRAIN_plan_mode')=='reviewed_prefix_fast_v1':
        marker=data/'generation_stop_receipt.json';stop=json.loads(marker.read_text())
        if(sha(marker)!=report['generation_stop_receipt_sha256'] or not stop['own_generator_stopped']
           or not stop['GPU_released'] or stop['candidate_manifest_sha256']!=report['candidate_manifest_sha256']):
            raise ValueError('Stopped TRAIN pool identity and resource receipt required')
        checked_fast_coverage(rows,refs)
    return rows,test,report


def curate(plan,candidates,review_path,parent,output,reviewed_prefix=False):
    if output.exists():raise ValueError('Keep every historical dataset unchanged')
    pr,expected=checked_plan(plan);cr=json.loads((candidates/'report.json').read_text())
    rows=json.loads((candidates/'candidates.json').read_text());review=json.loads(review_path.read_text())
    if(not cr['complete'] or cr.get('schema')!='scene_paper_expanded_TRAIN_candidates_v1'
       or cr['TRAIN_request_sha256']!=pr['TRAIN_request_sha256'] or cr['TEST_targets_or_predictions_created']
       or sha(candidates/'candidates.json')!=cr['candidate_manifest_sha256']):
        raise ValueError('Completed exact TRAIN candidate pool required')
    checked_candidate_rows(cr,rows,expected,reviewed_prefix)
    if reviewed_prefix:
        marker=candidates/'generation_stop_receipt.json';stop=json.loads(marker.read_text())
        if(not stop['own_generator_stopped'] or not stop['GPU_released']
           or stop['candidate_manifest_sha256']!=cr['candidate_manifest_sha256']
           or cr['generation_stop_receipt_sha256']!=sha(marker)):
            raise ValueError('Actual stopped own generation and pixel-bound prefix receipt required')
    if review.get('admission_policy')!='basic_subject_quality_v2':raise ValueError('Explicit partial TRAIN admission required')
    admitted=accepted_ids(rows,review,sha(candidates/'candidates.json'))
    if not admitted:raise ValueError('No usable TRAIN targets')
    refs=json.loads((plan/'references.json').read_text());catalog={r['reference_id']:r for r in refs}
    annotations={r['sample_id']:r for r in review['rows']};pairs=[]
    parent_report=json.loads((parent/'report.json').read_text())
    if not parent_report['complete'] or parent_report['pairs_sha256']!=sha(parent/'pairs.json'):raise ValueError('Completed parent dataset required')
    output.mkdir(parents=True);(output/'references').mkdir();(output/'targets').mkdir()
    for ref in refs:
        if sha(plan/ref['reference'])!=ref['reference_sha256']:raise ValueError('Source changed')
        shutil.copyfile(plan/ref['reference'],output/ref['reference'])
    for row in rows:
        if row['sample_id'] not in admitted:continue
        source=catalog[row['reference_id']];a=annotations[row['sample_id']]
        if sha(candidates/row['target'])!=row['target_sha256']:raise ValueError('Candidate pixels changed')
        shutil.copyfile(candidates/row['target'],output/row['target'])
        full=a['facts']=='YYY' and a['coarse_form']=='Y' and a['quality']=='Y'
        pairs.append(dict({k:row[k] for k in SAFE},split='train',source_product_id=source['source_product_id'],
            source_RGBA_sha256=source['source_RGBA_sha256'],target=row['target'],target_sha256=row['target_sha256'],
            QA_accepted=True,training_admission_policy='basic_subject_quality_v2',supervision_weight=1. if full else .5,
            all_requested_facts_and_form_satisfied=full,teacher_candidate_manifest_sha256=cr['candidate_manifest_sha256'],
            teacher_revision=row['teacher_revision'],teacher_kind=row.get('teacher_kind','frozen_official_OmniGen'),
            teacher_generation_steps=row.get('teacher_generation_steps',50),teacher_inference_batch_size=row.get('teacher_inference_batch_size',1)))
    coverage=checked_fast_coverage(pairs,refs) if reviewed_prefix else None
    if reviewed_prefix:shutil.copyfile(marker,output/'generation_stop_receipt.json')
    write(output/'pairs.json',pairs)
    for name in ['references.json','test_inputs.json','inference_inputs.json','selection_protocol.json','prompt_definitions.json']:
        shutil.copyfile(plan/name,output/name)
    pairsha=sha(output/'pairs.json')
    write(output/'visual_review.json',dict(approved_for_training=True,all_TRAIN_candidates_reviewed=True,
        pairs_sha256=pairsha,review_sha256=sha(review_path),partial_examples_not_full_instruction_success=True))
    write(output/'report.json',dict(schema=PAIRED_SCHEMA,complete=True,eligible_for_training=True,prompt_profile=PROFILE,
        pairs_sha256=pairsha,request_sha256=sha(output/'inference_inputs.json'),source_plan_request_sha256=pr['TRAIN_request_sha256'],
        source_parent_pairs_sha256=sha(parent/'pairs.json'),candidate_manifest_sha256=cr['candidate_manifest_sha256'],
        candidate_run=str(candidates),
        TRAIN_plan_mode='reviewed_prefix_fast_v1' if reviewed_prefix else 'full_candidate_plan',
        original_TRAIN_candidate_plan=len(expected),actual_TRAIN_candidates_reviewed=len(rows),
        full_original_TRAIN_candidate_plan_generated=not reviewed_prefix,
        fast_admission_coverage=coverage,
        generation_stop_receipt_sha256=sha(marker) if reviewed_prefix else None,
        TRAIN_pairs=len(pairs),TRAIN_products=len({r['source_product_id'] for r in pairs}),TRAIN_views=len({r['reference_id'] for r in pairs}),
        partial_TRAIN_pairs=sum(r['supervision_weight']==.5 for r in pairs),TEST_requests=500,TEST_target_images=0,
        validation_set=False,TEST_used_for_model_selection=True,role_disjoint=True,independent_generalization_claim=False,
        TEST_model_predictions_created=False))
    checked_dataset(output)


def checked_parent_transfer(data,parent,parent_sha,new_sha):
    rows,test,report=checked_dataset(data)
    if report['source_parent_pairs_sha256']!=parent_sha or sha(parent/'pairs.json')!=parent_sha or report['pairs_sha256']!=new_sha:
        raise ValueError('Explicit actual parent and new reviewed TRAIN identity required')
    old=json.loads((parent/'pairs.json').read_text());old_train={r['source_product_id'] for r in old if r['split']=='train'}
    sources=json.loads((data/'references.json').read_text());test_products={r['source_product_id'] for r in sources if r['split']=='test'}
    if old_train & test_products:raise ValueError('Parent TRAIN product leaked into TEST')
    return dict(schema='scene_expanded_reviewed_TRAIN_transfer_v1',parent_pair_sha256=parent_sha,new_pair_sha256=new_sha,
                validation_set=False,TEST_used_for_model_selection=True,independent_generalization_claim=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','candidates','review','parent','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reviewed-prefix',action='store_true',help='Time-priority reviewed subset; at least600 usable pairs and all source/template groups')
    a=p.parse_args();curate(a.plan,a.candidates,a.review,a.parent,a.output,a.reviewed_prefix)


if __name__=='__main__':main()
