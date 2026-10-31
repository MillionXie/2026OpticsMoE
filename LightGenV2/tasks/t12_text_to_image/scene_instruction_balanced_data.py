"""Curate factual TRAIN candidates into complete same-reference scene groups.

Teacher candidates are not accepted automatically. Every candidate is reviewed
against its pixels; incomplete reference/scene coverage writes a deficit report
and cannot produce a training-eligible dataset. Old evidence is never replaced.
"""
import argparse
from collections import defaultdict,Counter
import json
import os
from pathlib import Path
import shutil
from .scene_instruction_data import PROFILE,condition,sha,write
from .scene_instruction_balanced import SCENE_FIRST_PROFILE,BEDSIDE_PROFILE,scene_first_condition,bedside_condition,primary_codes
from .scene_omnigen_peer import REVISION
from .scene_instruction_native_prior import visible_caption


def checked_pool(root,review_path,allowed):
    report=json.loads((root/'report.json').read_text(encoding='utf8'));rows=json.loads((root/'candidates.json').read_text(encoding='utf8'));review=json.loads(review_path.read_text(encoding='utf8'))
    if not report.get('complete')or report.get('DEV_or_TEST_forwarded')or sha(root/'candidates.json')!=report['candidate_manifest_sha256']or review['candidate_manifest_sha256']!=report['candidate_manifest_sha256']:raise ValueError('Changed or non-TRAIN teacher cohort')
    if json.loads((root/'model_assets.json').read_text(encoding='utf8'))['official_revision']!=REVISION:raise ValueError('Unapproved teacher identity')
    annotations={r['sample_id']:r for r in review['rows']}
    if len(annotations)!=len(rows)or set(annotations)!={r['sample_id']for r in rows}:raise ValueError('Review every candidate, including failures')
    result=[]
    for row in rows:
        source=allowed.get(row['reference_id']);annotation=annotations[row['sample_id']]
        if source is None or row['split']!='train'or any(row[k]!=source[k]for k in ['reference_sha256','source_product_id','source_RGBA_sha256','category']):raise ValueError('Candidate outside the fixed44 TRAIN sources')
        code=row['anchor_code'];profile=row['prompt_profile']
        if code not in primary_codes(row['category']):raise ValueError('Not a primary counterfactual condition')
        expected=condition(row['category'],code)if profile==PROFILE else scene_first_condition(row['category'],code)if profile==SCENE_FIRST_PROFILE else bedside_condition()if profile==BEDSIDE_PROFILE and row['category']=='lamp'and code=='D'else None
        if expected is None or row['prompt']!=expected['prompt']:raise ValueError('Unrecorded teacher prompt')
        if sha(root/row['reference'])!=row['reference_sha256']or sha(root/row['target'])!=row['target_sha256']or annotation['target_sha256']!=row['target_sha256']:raise ValueError('Review pixel identity changed')
        accepted=annotation['status']=='usable_TRAIN_example'
        if annotation['accepted_for_training']!=accepted:raise ValueError('Uncertain/rejected is not usable')
        if accepted:result.append((root,row,review_path))
    return result


def curate(plan,dev,pools,output):
    if output.exists():raise ValueError('Preserve every prior curation')
    plan_report=json.loads((plan/'report.json').read_text(encoding='utf8'));requests=json.loads((plan/'requests.json').read_text(encoding='utf8'))
    if plan_report['schema']!='scene_balanced_visible_anchor_plan_v1'or sha(plan/'requests.json')!=plan_report['request_sha256']or len(requests)!=264:raise ValueError('Original fixed264 TRAIN plan required')
    allowed={r['reference_id']:r for r in requests}
    if len(allowed)!=44 or any(r['split']!='train'for r in requests):raise ValueError('All44 prespecified TRAIN products required')
    accepted=[]
    for spec in pools:accepted.extend(checked_pool(Path(spec['run']),Path(spec['review']),allowed))
    grouped=defaultdict(list)
    for item in accepted:grouped[item[1]['reference_id'],item[1]['anchor_code']].append(item)
    selected=[];missing=[];used=defaultdict(set)
    for rid,source in sorted(allowed.items()):
        for code in primary_codes(source['category']):
            choices=[];seeds=set()
            # Specification order fixes precedence; never select by generated
            # scores. Same public RGB/prompt/seed cannot bind conflicting PNGs.
            for item in grouped[rid,code]:
                row=item[1]
                if row['seed']in seeds or row['target_sha256']in used[rid]:continue
                seeds.add(row['seed']);used[rid].add(row['target_sha256']);choices.append(item)
                if len(choices)==2:break
            if not choices:missing.append(dict(reference_id=rid,anchor_code=code,reference=source['reference'],reference_sha256=source['reference_sha256'],prompt=condition(source['category'],code)['prompt']))
            selected.extend(choices)
    output.mkdir()
    provenance=[dict(run=str(Path(s['run'])),review=str(Path(s['review'])),candidate_manifest_sha256=sha(Path(s['run'])/'candidates.json'),review_sha256=sha(Path(s['review'])))for s in pools]
    write(output/'candidate_provenance.json',provenance);write(output/'missing_reference_conditions.json',missing)
    if missing:
        write(output/'readiness.json',dict(complete=True,ready_for_training=False,missing_reference_conditions=len(missing),covered_reference_conditions=132-len(missing),accepted_candidate_slots=len(selected),all44_sources_preserved=True,TEST_opened=False))
        return
    old=json.loads((dev/'report.json').read_text(encoding='utf8'));old_rows=json.loads((dev/'pairs.json').read_text(encoding='utf8'))
    if old['schema']!='scene_visible_anchor_dataset_v1'or not old['complete']or old['TEST_opened']or sha(dev/'pairs.json')!=old['pairs_sha256']:raise ValueError('Fixed factual DEV labels required')
    (output/'references').mkdir();(output/'targets').mkdir();pairs=[]
    def copy_pixels(root,row):
        for key in ['reference','target']:
            src=root/row[key];dst=output/row[key]
            if sha(src)!=row[key+'_sha256']:raise ValueError('Changed dataset pixels')
            if not dst.exists():shutil.copyfile(src,dst)
            elif sha(dst)!=row[key+'_sha256']:raise ValueError('Filename collision')
    for root,candidate,review_path in selected:
        row=dict(candidate,**condition(candidate['category'],candidate['anchor_code']))
        row['native_caption']=visible_caption(row['category'],row['anchor_code'])
        generated_prompt=candidate['prompt'];reuse=candidate.get('reused_reviewed_label')
        if reuse is not None:
            earlier=Path(reuse['data_root']);earlier_report=json.loads((earlier/'report.json').read_text(encoding='utf8'))
            if sha(earlier/'pairs.json')!=earlier_report['pairs_sha256']:raise ValueError('Earlier reviewed provenance changed')
            source={r['sample_id']:r for r in json.loads((earlier/'pairs.json').read_text(encoding='utf8'))}[reuse['original_sample_id']]
            if source['target_sha256']!=candidate['target_sha256']or source['reference_sha256']!=candidate['reference_sha256']:raise ValueError('Earlier target provenance changed')
            generated_prompt=source.get('original_prompt',source['prompt'])
        row.update(QA_accepted=True,teacher_candidate_run=str(root),teacher_candidate_sample_id=candidate['sample_id'],reviewed_candidate_caption=candidate['prompt'],target_generation_prompt=generated_prompt,
            teacher_candidate_manifest_sha256=sha(root/'candidates.json'),factual_review_sha256=sha(review_path),
            recaption_is_training_data_not_rescored_model_output=candidate['prompt']!=row['prompt'])
        copy_pixels(root,row);pairs.append(row)
    train_products={r['source_product_id']for r in pairs}
    for row in old_rows:
        if row['split']=='dev':
            row=dict(row,previous_native_caption=row['native_caption'],native_caption=visible_caption(row['category'],row['anchor_code']))
            if row['source_product_id']in train_products:raise ValueError('Role leakage')
            copy_pixels(dev,row);pairs.append(row)
    inputs=json.loads((dev/'inference_inputs.json').read_text(encoding='utf8'))
    if len(inputs)!=48 or len({r['reference_id']for r in inputs})!=16:raise ValueError('All original16 DEV views/48 requests required')
    for row in inputs:
        code=({'0':'B','1':'D','2':'W'}if row['category']=='lamp'else{'0':'S','1':'A','2':'D'})[row['condition_id']]
        row['native_caption']=visible_caption(row['category'],code)
        src=dev/row['reference'];dst=output/row['reference']
        if sha(src)!=row['reference_sha256']:raise ValueError('DEV reference changed')
        if not dst.exists():shutil.copyfile(src,dst)
    owners={}
    for row in pairs:
        for field in ['source_product_id','source_RGBA_sha256','reference_sha256']:
            if owners.setdefault((field,row[field]),row['split'])!=row['split']:raise ValueError('Cross-role source')
    write(output/'pairs.json',pairs);write(output/'inference_inputs.json',inputs);write(output/'inference_dev_inputs.json',inputs)
    pairsha=sha(output/'pairs.json');write(output/'visual_review.json',dict(approved_for_training=True,pairs_sha256=pairsha,per_candidate_review_provenance=provenance,full_reference_condition_coverage=True,not_a_model_success_claim=True))
    write(output/'report.json',dict(complete=True,schema='scene_balanced_visible_anchor_dataset_v1',source_commit=os.environ.get('T12_SOURCE_COMMIT'),prompt_profile=PROFILE,
        counts=dict(Counter(r['split']for r in pairs)),pairs_sha256=pairsha,request_sha256=sha(output/'inference_inputs.json'),source_pairs_sha256=old['source_pairs_sha256'],
        factual_parent_pairs_sha256=old['pairs_sha256'],role_disjoint=True,TRAIN_products=44,TRAIN_reference_views=44,TRAIN_condition_groups=132,
        all_reference_conditions_covered=True,eligible_for_training=True,target_resolution=512,preview_only=False,TEST_opened=False,
        native_caption_style='public_fact_equivalent_description_v1',native_caption_has_no_target_or_SKU_attributes=True,
        old_TEST_scores_do_not_apply=True,baseline_instruction_success_certified=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['plan','dev','pools','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();curate(a.plan,a.dev,json.loads(a.pools.read_text(encoding='utf8')),a.output)


if __name__=='__main__':main()
