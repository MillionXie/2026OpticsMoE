"""Explicitly recaption reviewed TRAIN/DEV examples; never relabel old results.

One visible background anchor replaces the former multi-object room lists.
Per-image facts, pixels, parent acceptance and role identities are mandatory.
New inference requests have no target and keep the full original DEV cohort.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil

PROFILE='two_sentence_visible_anchor_v5'
SAFE=('sample_id','reference','prompt','seed','reference_id','reference_sha256','category','condition_id','native_caption','reward_caption','prompt_profile')
LAMP={'B':('bookshelves','Bookshelves are visible in the background.'),
      'D':('bed','A bed is visible in the background.'),
      'W':('window','A window is visible in the background.'),
      'S':('sofa','A sofa is visible in the background.'),
      'A':('armchair','An armchair is visible in the background.')}
PILLOW={'S':('sofa','Generate an indoor photograph of a cushion on a sofa. Show the cushion and sofa together in the room.'),
        'A':('armchair','Generate an indoor photograph of a cushion on an armchair. Show the cushion and armchair together in the room.'),
        'D':('bed','Generate an indoor photograph of a cushion on a bed. Show the cushion and bed together in the room.'),
        'W':('window_surface','Generate an indoor photograph of a cushion on a wooden surface. A window is visible in the background.')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')


def condition(category,code):
    if category=='lamp' and code in LAMP:
        noun,second=LAMP[code];prompt='Generate an indoor photograph of a table lamp on a table. '+second
        checks=['recognizable table lamp','lamp supported by a table','visible '+noun,'coherent indoor photograph']
    elif category=='pillow' and code in PILLOW:
        noun,prompt=PILLOW[code]
        checks=['recognizable separate cushion','cushion supported by '+noun.replace('_',' '),'coherent indoor photograph']
        if code=='W':checks.insert(2,'visible window')
    else:raise ValueError('Unknown category/visible anchor')
    if prompt.count('.')!=2 or any(x in prompt.lower()for x in ['reference','fine-tun','lighting direction']):raise ValueError('Public two-sentence contract changed')
    cid=str(list(LAMP if category=='lamp'else PILLOW).index(code))
    return dict(condition_id=cid,anchor_code=code,prompt=prompt,native_caption=prompt,reward_caption=prompt,
                prompt_profile=PROFILE,category=category,required_visible_facts=checks)


def checked_annotations(candidates,review):
    if review.get('schema')!='scene_visible_facts_review_v1' or not review.get('all_candidates_reviewed'):
        raise ValueError('Complete explicit image review required')
    indexed={r['sample_id']:r for r in candidates}
    if len(indexed)!=len(candidates):raise ValueError('Duplicate source IDs')
    annotations=review['annotations']
    if len(annotations)!=len(indexed) or {r['sample_id']for r in annotations}!=set(indexed):raise ValueError('Review denominator changed')
    result={}
    for entry in annotations:
        row=indexed[entry['sample_id']]
        if entry['target_sha256']!=row['target_sha256']:raise ValueError('Review is not bound to target pixels')
        if entry['code']!='X':condition(row['category'],entry['code'])
        result[entry['sample_id']]=entry
    return result


def prepare(corpus,paired,review_file,output):
    if output.exists():raise ValueError('Never overwrite a prior protocol')
    old=json.loads((paired/'report.json').read_text(encoding='utf8'))
    if not old['complete'] or not old['role_disjoint'] or old['TEST_opened'] or old['pairs_sha256']!=sha(paired/'pairs.json'):raise ValueError('Invalid reviewed parent')
    candidates=json.loads((corpus/'candidates.json').read_text(encoding='utf8'));review=json.loads(review_file.read_text(encoding='utf8'))
    if review['candidate_manifest_sha256']!=sha(corpus/'candidates.json'):raise ValueError('Different image review')
    annotations=checked_annotations(candidates,review)
    accepted={r['sample_id']:r for r in json.loads((paired/'pairs.json').read_text(encoding='utf8'))}
    output.mkdir(parents=True);(output/'references').mkdir();(output/'targets').mkdir()
    rows=[];rejected=[];seen=set();owners={}
    for original in sorted(candidates,key=lambda r:r['sample_id']):
        annotation=annotations[original['sample_id']]
        if original['sample_id']not in accepted or annotation['code']=='X':
            rejected.append(dict(sample_id=original['sample_id'],reason='prior_rejection_preserved'if original['sample_id']not in accepted else'no_verified_usable_subject_and_anchor'));continue
        row=dict(accepted[original['sample_id']]);definition=condition(row['category'],annotation['code'])
        signature=(row['reference_sha256'],definition['prompt'],row['seed'])
        if signature in seen:
            rejected.append(dict(sample_id=row['sample_id'],reason='same_RGB_prompt_seed_duplicate_after_recaption'));continue
        seen.add(signature)
        for key in ['source_product_id','reference_sha256','source_RGBA_sha256']:
            token=(key,row[key]);prior=owners.setdefault(token,row['split'])
            if prior!=row['split']:raise ValueError('Cross-role input')
        for key,hashkey in [('reference','reference_sha256'),('target','target_sha256')]:
            src=paired/row[key]
            if sha(src)!=row[hashkey]:raise ValueError('Changed reviewed pixels')
            dst=output/row[key]
            if not dst.exists():shutil.copyfile(src,dst)
        old_prompt=row['prompt'];row.update(definition)
        row.update(original_sample_id=original['sample_id'],original_prompt=old_prompt,
            requested_attributes=dict(visible_anchor=definition['anchor_code'],fine_details_free=True,change_product_color=False),
            QA_accepted=True,acceptance_note='New factual two-sentence caption: subject/support/one visible anchor checked against these exact pixels. Not a generation success claim.',
            prompt_recaption_is_new_training_data=True,old_outputs_not_rescored=True)
        rows.append(row)
    original_inputs=json.loads((paired/'inference_dev_inputs.json').read_text(encoding='utf8'));refs={r['reference_id']:r for r in original_inputs}
    if len(refs)!=16:raise ValueError('Keep all original sixteen DEV views; do not hide hard views')
    requests=[]
    for rid,ref in sorted(refs.items()):
        codes=['B','D','W']if ref['category']=='lamp'else['S','A','D']
        for code in codes:
            d=condition(ref['category'],code);row={k:ref[k]for k in SAFE};row.update({k:d[k]for k in ['condition_id','prompt','native_caption','reward_caption','prompt_profile','category']})
            row['sample_id']=rid+'-visible-'+code+'-draw2';requests.append(row)
        dst=output/ref['reference'];src=paired/ref['reference']
        if sha(src)!=ref['reference_sha256']:raise ValueError('DEV input changed')
        if not dst.exists():shutil.copyfile(src,dst)
    write(output/'pairs.json',rows);write(output/'inference_dev_inputs.json',requests)
    write(output/'inference_inputs.json',requests)
    # Checklists live outside the generator whitelist, keyed to the full public prompt.
    checklists={}
    for category,codes in [('lamp',LAMP),('pillow',PILLOW)]:
        for code in codes:
            c=condition(category,code);checklists[c['prompt']]=c['required_visible_facts']
    write(output/'required_facts.json',checklists)
    write(output/'rejected_or_deduplicated.json',rejected)
    by_prompt=defaultdict(set)
    for r in rows:
        if r['split']=='train':by_prompt[r['prompt']].add(r['source_product_id'])
    pairsha=sha(output/'pairs.json');counts=dict(Counter(r['split']for r in rows))
    write(output/'data_audit.json',dict(counts=counts,all16_DEV_views_retained=True,DEV_target_free_requests=48,
        TRAIN_prompt_distinct_products={p:len(ids)for p,ids in by_prompt.items()},old_targets_and_original_prompts_preserved=True,
        same_input_conflicting_targets_disallowed=True,TEST_opened=False,parent_pairs_sha256=old['pairs_sha256']))
    write(output/'visual_review.json',dict(approved_for_training=True,pairs_sha256=pairsha,
        basis='Per-image visible facts, prior coarse-shape rejection inherited, duplicate conflicting inputs removed.',review_sha256=sha(review_file),not_a_test_accuracy_evaluation=True))
    write(output/'report.json',dict(complete=True,schema='scene_visible_anchor_dataset_v1',source_commit=os.environ.get('T12_SOURCE_COMMIT'),
        prompt_profile=PROFILE,counts=counts,pairs_sha256=pairsha,request_sha256=sha(output/'inference_inputs.json'),
        role_disjoint=True,TEST_opened=False,preview_only=False,eligible_for_training=True,target_resolution=512,
        source_pairs_sha256=old['pairs_sha256'],review_sha256=sha(review_file),
        old_test_scores_do_not_apply=True,new_baseline_adherence_validation_required=True,accepted_as_baseline_or_LG_success=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['corpus','paired','review','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();prepare(a.corpus,a.paired,a.review,a.output)


if __name__=='__main__':main()
