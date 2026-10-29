"""Accept explicitly reviewed whole-scene candidates; keep all DEV requests.

Rejected teacher outputs remain in their original run. No rejected DEV request
is removed from generation evaluation. Targets are TRAIN/DEV denoising examples,
not a uniquely correct test image. SKU/hash metadata never enters Qwen text.
"""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import shutil
from .white_reference_scene_probe import digest,write

SAFE_INPUTS=['sample_id','reference','prompt','seed','reference_id','reference_sha256','category',
             'condition_id','native_caption','reward_caption','prompt_profile']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['candidates','review','output']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or len(os.environ.get('T12_SOURCE_COMMIT',''))!=40:raise ValueError('New dataset and published source required')
    report=json.loads((a.candidates/'report.json').read_text(encoding='utf8'))
    review=json.loads(a.review.read_text(encoding='utf8'))
    rows=json.loads((a.candidates/'candidates.json').read_text(encoding='utf8'))
    requests=json.loads((a.candidates/'requests.json').read_text(encoding='utf8'))
    assert report['complete'] and not report['pilot_only'] and not report['TEST_opened']
    assert review['candidate_manifest_sha256']==digest(a.candidates/'candidates.json')
    annotations={r['sample_id']:r for r in review['annotations']}
    assert len(annotations)==len(rows) and set(annotations)=={r['sample_id'] for r in rows}
    assert all(isinstance(r['approved'],bool) for r in annotations.values())
    selected=[{**r,'QA_accepted':True,'acceptance_note':annotations[r['sample_id']].get('reason','')}
              for r in rows if annotations[r['sample_id']]['approved']]
    products={s:{r['source_product_id'] for r in rows if r['split']==s} for s in ['train','dev']}
    assert not products['train']&products['dev']
    # Every TRAIN product must have usable supervision. DEV generation keeps
    # its complete input denominator even when a teacher failed every target
    # for one product: never force a wrong image into the noise-loss labels.
    for split in ['train']:
        accepted={r['source_product_id'] for r in selected if r['split']==split}
        if accepted!=products[split]:raise ValueError('Need valid additional candidates for '+split+': '+str(sorted(products[split]-accepted)))
    a.output.mkdir(parents=True);(a.output/'references').mkdir();(a.output/'targets').mkdir()
    refs={r['reference_id']:r for r in requests}
    for row in refs.values():
        src=a.candidates/row['reference'];assert digest(src)==row['reference_sha256']
        shutil.copyfile(src,a.output/row['reference'])
    for row in selected:
        assert row['split'] in ['train','dev']
        src=a.candidates/row['target'];assert digest(src)==row['target_sha256']
        shutil.copyfile(src,a.output/row['target'])
    write(a.output/'pairs.json',selected)
    write(a.output/'inference_dev_inputs.json',[{k:r[k] for k in SAFE_INPUTS} for r in requests if r['split']=='dev'])
    by_prompt=defaultdict(set);same_input=Counter()
    for row in selected:
        if row['split']=='train':
            by_prompt[row['prompt']].add(row['source_product_id']);same_input[(row['reference_id'],row['prompt'])]+=1
    write(a.output/'data_audit.json',{'counts':dict(Counter(r['split'] for r in selected)),
        'products':{s:len(p) for s,p in products.items()},'TRAIN_prompt_to_distinct_products':
        {prompt:len(ids) for prompt,ids in by_prompt.items()},
        'TRAIN_reference_prompt_groups_with_multiple_targets':sum(n>1 for n in same_input.values()),
        'all_DEV_inference_requests_retained':True,'TEST_opened':False,'teacher_rejections_retained_in_parent':True})
    shutil.copyfile(a.review,a.output/'candidate_review.json')
    pair_sha=digest(a.output/'pairs.json')
    write(a.output/'visual_review.json',{'approved_for_training':True,'pairs_sha256':pair_sha,
        'basis':'Explicit per-candidate coarse-shape/artifact screening; all TRAIN products and full DEV generation inputs retained; rejected DEV targets are not forced into noise supervision.',
        'review_sha256':digest(a.review),'not_a_test_accuracy_evaluation':True})
    write(a.output/'report.json',{'complete':True,'source_commit':os.environ['T12_SOURCE_COMMIT'],
        'source_sha256':digest(Path(__file__)),'pairs_sha256':pair_sha,'counts':dict(Counter(r['split'] for r in selected)),
        'role_disjoint':True,'TEST_opened':False,'target_resolution':report['resolution'],
        'prompt_profile':report['prompt_profile'],'preview_only':False,'eligible_for_training':True,
        'parent_candidates_manifest_sha256':digest(a.candidates/'candidates.json'),
        'original_synthetic_outputs_are_training_examples_not_unique_test_GT':True,
        'ABO_reference_attribution':'Amazon Berkeley Objects (ABO), CC BY 4.0',
        'target_provenance':'Own frozen public SDXL/IP-Adapter/Canny generations, separate from ABO original images.',
        'RGB_compositing_at_inference':False,'image_feedback':False})


if __name__=='__main__':main()
