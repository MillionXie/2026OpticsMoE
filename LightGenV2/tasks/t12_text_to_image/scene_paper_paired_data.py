"""Curate newly generated v7 targets; preserve old pixels, labels and scores.

Only individually approved new TRAIN images enter gradients. DEV labels select
noise loss only; raw generation always retains all48 original DEV requests.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from .scene_instruction_data import SAFE,sha
from .scene_paper_prompt_protocol import PROFILE,PARENT_PAIRS,condition,write
from .scene_omnigen_peer import REVISION
SOURCE_CATALOG_SHA='96f687e146d3c76bd9320402cd67cb4b9cbcc52244ddf75c5298523c457fa2be'


def source_inventory(path,parent_rows):
    if sha(path)!=SOURCE_CATALOG_SHA:raise ValueError('Fixed pre-existing TRAIN/DEV source inventory required')
    inventory=json.loads(Path(path).read_text(encoding='utf8'));catalog={r['sample_id']:dict(split=r['split'],source_product_id=r['product_id'],source_RGBA_sha256=r['rgba_file_sha256'],reference_sha256=r['white_reference_sha256'])for r in inventory}
    if len(catalog)!=92 or any(r['split']not in['train','dev']for r in inventory):raise ValueError('No TEST source inventory')
    for r in parent_rows:
        if any(r[k]!=catalog[r['reference_id']][k]for k in ['split','source_product_id','source_RGBA_sha256','reference_sha256']):raise ValueError('Original source identity changed')
    return catalog


def accepted_ids(rows,review,manifest_sha):
    if review.get('schema')!='scene_paper_new_target_review_v1'or review.get('all_candidates_reviewed')is not True or review.get('candidate_manifest_sha256')!=manifest_sha:
        raise ValueError('Complete pixel-bound new-target review required')
    annotations={r['sample_id']:r for r in review['rows']}
    if len(annotations)!=len(rows)or set(annotations)!={r['sample_id']for r in rows}:
        raise ValueError('Every candidate, including failures, needs an annotation')
    accepted=set()
    policy=review.get('admission_policy','all_facts_form_quality_v1')
    if policy not in ['all_facts_form_quality_v1','basic_subject_quality_v2']:raise ValueError('Unknown explicit training-label admission')
    for row in rows:
        a=annotations[row['sample_id']];target_sha=row.get('target_sha256',row.get('generated_sha256'))
        if a['target_sha256']!=target_sha or len(a['facts'])!=3 or len(a['coarse_form'])!=1 or len(a['quality'])!=1 or any(x not in'YNU'for x in a['facts']+a['coarse_form']+a['quality']):
            raise ValueError('Changed target pixels or invalid review states')
        valid=a['facts']=='YYY'and a['coarse_form']=='Y'and a['quality']=='Y'
        if policy=='basic_subject_quality_v2':valid=a['facts'][0]=='Y'and a['quality']=='Y'and a.get('gross_subject_failure')is False
        if a['accepted']is not valid:raise ValueError('Acceptance must follow the recorded training-label policy, not silently change evaluation success')
        if valid:accepted.add(row['sample_id'])
    return accepted


def curate(plan,parent,sources,train,train_review,dev,dev_review,output):
    if output.exists():raise ValueError('Preserve all historical datasets')
    p=json.loads((plan/'report.json').read_text());old=json.loads((parent/'report.json').read_text())
    if not p['complete']or p['prompt_profile']!=PROFILE or p['source_pairs_sha256']!=PARENT_PAIRS or sha(parent/'pairs.json')!=PARENT_PAIRS or not old['role_disjoint']or p['TEST_opened']:
        raise ValueError('Explicit original role-disjoint v7 plan and R2 parent required')
    requests=json.loads((plan/'train_inputs.json').read_text());devrequests=json.loads((plan/'dev_inputs.json').read_text())
    if len(requests)!=264 or len(devrequests)!=48 or sha(plan/'train_inputs.json')!=p['TRAIN_request_sha256']or sha(plan/'dev_inputs.json')!=p['request_sha256']:raise ValueError('Keep fixed complete input cohorts')
    tr=json.loads((train/'report.json').read_text());dr=json.loads((dev/'report.json').read_text())
    draws=tr.get('selected_draw_indices',[0,1])
    if draws not in [[0],[0,1]]:raise ValueError('Only full44-source three-scene first-draw or original two-draw plans')
    expected_train=[r for r in requests if any(r['sample_id'].endswith('-draw'+str(j))for j in draws)]
    if not tr['complete']or not dr['complete']or tr.get('schema')!='scene_paper_TRAIN_candidates_v1'or tr['DEV_or_TEST_generated']or tr['TRAIN_request_sha256']!=p['TRAIN_request_sha256']:
        raise ValueError('Actual completed new TRAIN candidates required')
    if dr['prompt_profile']!=PROFILE or dr.get('independent_TEST_opened')or dr['generated_count']!=48 or dr.get('official_weights_revision')!=REVISION:raise ValueError('Full new-prompt frozen OmniGen DEV peer required')
    trainrows=json.loads((train/'candidates.json').read_text());devrows=json.loads((dev/'images.json').read_text())
    if any(r.get('teacher_revision')!=REVISION or r.get('GT_masks_or_targets_as_inputs')is not False for r in trainrows):raise ValueError('Frozen native teacher and target-free input evidence required')
    if sha(train/'candidates.json')!=tr['candidate_manifest_sha256']:raise ValueError('Changed TRAIN candidate manifest')
    for expected,actual in [(expected_train,trainrows),(devrequests,devrows)]:
        if len(expected)!=len(actual)or any({k:r[k]for k in SAFE}!={k:a[k]for k in SAFE}for r,a in zip(expected,actual)):raise ValueError('Candidate input/prompt/source/seed changed')
    reviews=[json.loads(train_review.read_text()),json.loads(dev_review.read_text())]
    chosen=[accepted_ids(trainrows,reviews[0],sha(train/'candidates.json')),accepted_ids(devrows,reviews[1],sha(dev/'images.json'))]
    if not chosen[0]or not chosen[1]:raise ValueError('At least one reviewed TRAIN and DEV target required')
    catalog=source_inventory(sources,json.loads((parent/'pairs.json').read_text()));inference_refs={r['reference_id']for r in devrequests}
    output.mkdir();(output/'references').mkdir();(output/'targets').mkdir();pairs=[];owners={}
    for split,root,rows,ids,review in [('train',train,trainrows,chosen[0],reviews[0]),('dev',dev,devrows,chosen[1],reviews[1])]:
        annotations={r['sample_id']:r for r in review['rows']}
        for r in rows:
            if r['sample_id']not in ids:continue
            source=catalog[r['reference_id']]
            if (r['reference_id']in inference_refs)!=(split=='dev')or r['reference_sha256']!=source['reference_sha256']:raise ValueError('Source role changed')
            for field in ['source_product_id','source_RGBA_sha256','reference_sha256']:
                value=source[field]
                if owners.setdefault((field,value),split)!=split:raise ValueError('Cross-role identity')
            original=root/(r['target']if split=='train'else r['generated_PNG']);digest=r['target_sha256']if split=='train'else r['generated_sha256']
            if sha(original)!=digest or sha(plan/r['reference'])!=r['reference_sha256']:raise ValueError('Reviewed image bytes changed')
            dst=output/r['reference']
            if not dst.exists():shutil.copyfile(plan/r['reference'],dst)
            target='targets/'+r['sample_id']+'.png';shutil.copyfile(original,output/target)
            a=annotations[r['sample_id']];full=a['facts']=='YYY'and a['coarse_form']=='Y'and a['quality']=='Y'
            pairs.append(dict({k:r[k]for k in SAFE},split=split,source_product_id=source['source_product_id'],source_RGBA_sha256=source['source_RGBA_sha256'],target=target,target_sha256=digest,QA_accepted=True,training_admission_policy=review.get('admission_policy','all_facts_form_quality_v1'),all_requested_facts_and_form_satisfied=full,supervision_weight=1.0 if full else 0.5,teacher='frozen official OmniGen-v1; newly generated v7 target',teacher_candidate_run=str(root),teacher_candidate_manifest_sha256=sha(root/('candidates.json'if split=='train'else'images.json')),scene_GT_is_one_valid_training_example=full,old_target_not_recaptioned=True,required_visible_facts=condition(r['category'],r['condition_id'])['required_visible_facts']))
    for r in devrequests:
        dst=output/r['reference']
        if not dst.exists():shutil.copyfile(plan/r['reference'],dst)
    write(output/'pairs.json',pairs);write(output/'inference_inputs.json',devrequests);write(output/'inference_dev_inputs.json',devrequests)
    shutil.copyfile(sources,output/'source_catalog.json')
    count=Counter(r['split']for r in pairs);groups={(r['reference_id'],r['condition_id'])for r in pairs if r['split']=='train'}
    missing=sorted({(r['reference_id'],r['condition_id'])for r in requests}-groups)
    write(output/'report.json',dict(schema='scene_paper_paired_dataset_v1',complete=True,eligible_for_training=True,prompt_profile=PROFILE,role_disjoint=True,TEST_opened=False,source_pairs_sha256=PARENT_PAIRS,pairs_sha256=sha(output/'pairs.json'),source_TRAIN_request_sha256=p['TRAIN_request_sha256'],selected_TRAIN_draws=draws,candidate_TRAIN_count=len(trainrows),partial_TRAIN_labels=sum(r['split']=='train'and not r['all_requested_facts_and_form_satisfied']for r in pairs),request_sha256=p['request_sha256'],accepted_counts=dict(count),TRAIN_source_condition_groups=len(groups),missing_TRAIN_source_condition_groups=missing,all44_TRAIN_sources_fully_balanced=not missing,full48_DEV_inference_preserved=True,unique_GT_required_at_inference=False,old_targets_not_relabelled=True))
    write(output/'visual_review.json',dict(approved_for_training=True,pairs_sha256=sha(output/'pairs.json'),all_new_candidates_reviewed=True,TRAIN_admission_policy=reviews[0].get('admission_policy','all_facts_form_quality_v1'),TRAIN_review_sha256=sha(train_review),DEV_review_sha256=sha(dev_review),partial_training_labels_not_counted_as_complete_instruction_success=True))


def checked_parent_transfer(new_data,parent_data,parent_sha,new_sha):
    r=json.loads((new_data/'report.json').read_text());v=json.loads((new_data/'visual_review.json').read_text())
    if r.get('schema')!='scene_paper_paired_dataset_v1'or not r['complete']or r['prompt_profile']!=PROFILE or not r['role_disjoint']or r['TEST_opened']or parent_sha!=PARENT_PAIRS or r['source_pairs_sha256']!=parent_sha or sha(parent_data/'pairs.json')!=parent_sha or sha(new_data/'pairs.json')!=new_sha or r['pairs_sha256']!=new_sha or v['pairs_sha256']!=new_sha or not v['approved_for_training']or not v['all_new_candidates_reviewed']:raise ValueError('Only explicit reviewed new-target transfer from R2 required')
    sources=source_inventory(new_data/'source_catalog.json',json.loads((parent_data/'pairs.json').read_text()))
    for row in json.loads((new_data/'pairs.json').read_text()):
        source=sources[row['reference_id']]
        if any(row[k]!=source[k]for k in ['split','source_product_id','source_RGBA_sha256','reference_sha256'])or not row['QA_accepted']or not row['old_target_not_recaptioned']or sha(new_data/row['reference'])!=row['reference_sha256']or sha(new_data/row['target'])!=row['target_sha256']:raise ValueError('Preserve original roles and bind reviewed new pixels')
    return dict(schema='scene_reviewed_new_v7_target_transfer_v1',parent_pair_sha256=parent_sha,new_pair_sha256=new_sha,new_targets_and_new_prompts_not_recaptioned_old_pixels=True,full48_DEV_requests_preserved=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','parent','sources','train','train-review','dev','dev-review','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();curate(a.plan,a.parent,a.sources,a.train,a.train_review,a.dev,a.dev_review,a.output)


if __name__=='__main__':main()
