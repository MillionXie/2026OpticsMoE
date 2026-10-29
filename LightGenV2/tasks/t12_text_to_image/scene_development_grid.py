"""Target-free eight-scene DEV grid over the ORIGINAL sixteen DEV views.

The small 32-request review covered only two scenes. Before a final TEST freeze,
evaluate all eight canonical training scene types on these same DEV subjects.
No held-out TEST asset, scene GT, feature tensor or model is opened here.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from .appearance_scene_protocol import SHORT_PROFILE,conditions
from .white_reference_scene_probe import digest,write

SAFE={'sample_id','reference','prompt','seed','reference_id','reference_sha256',
      'category','condition_id','native_caption','reward_caption','prompt_profile'}


def expand(original):
    if len(original)!=32 or any(set(r)!=SAFE for r in original):raise ValueError('Require original target-free DEV32 requests')
    refs={}
    for row in original:
        old=refs.setdefault(row['reference_id'],{})
        if row['condition_id'] in old:raise ValueError('Duplicate source condition')
        old[row['condition_id']]=row
    if len(refs)!=16:raise ValueError('Require all sixteen existing DEV views')
    if len({rid.rsplit('-',1)[0] for rid in refs})!=8:raise ValueError('Require the original eight DEV products')
    result=[]
    for rid,rows in refs.items():
        if set(rows)!={'0','1'}:raise ValueError('Original two-condition cohort changed')
        row=rows['0']
        for key in ['reference','reference_sha256','category','seed','prompt_profile']:
            if rows['1'][key]!=row[key]:raise ValueError('Source/noise differs across counterfactual prompts')
        if row['prompt_profile']!=SHORT_PROFILE:raise ValueError('Unexpected canonical prompt contract')
        suffix=row['sample_id'].split('-draw')[-1]
        for scene in conditions(row['category'],SHORT_PROFILE):
            cid=scene['condition_id'];entry=dict(row)
            for key in ['prompt','condition_id','category','native_caption','reward_caption','prompt_profile']:entry[key]=scene[key]
            entry['sample_id']=rid+'-scene'+cid+'-draw'+suffix
            if cid in rows and entry!=rows[cid]:raise ValueError('Existing DEV requests must remain byte-for-field identical')
            result.append(entry)
    if len(result)!=128 or len({r['sample_id'] for r in result})!=128:raise ValueError('Incomplete canonical grid')
    return result


def checked_grid(folder,data,state,original):
    report=json.loads((folder/'report.json').read_text())
    if not (report['complete'] and report['schema']=='scene_expanded_development_v1' and not report['TEST_opened']):raise ValueError('Explicit completed DEV-grid preparation required')
    if report['source_DEV_request_sha256']!=digest(data/'inference_dev_inputs.json'):raise ValueError('Grid source cohort changed')
    if report['pairs_sha256']!=state['pair_sha256']:raise ValueError('Grid belongs to another paired training protocol')
    path=folder/'inference_inputs.json';actual=json.loads(path.read_text())
    if report['request_sha256']!=digest(path) or actual!=expand(original):raise ValueError('Grid contains noncanonical or non-DEV requests')
    return actual,dict(manifest_sha256=digest(folder/'report.json'),request_sha256=report['request_sha256'],
        source_DEV_request_sha256=report['source_DEV_request_sha256'],contexts=128,reference_views=16,TEST_opened=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    report=json.loads((a.data/'report.json').read_text())
    assert report['complete'] and report['role_disjoint'] and not report['TEST_opened']
    assert digest(a.data/'pairs.json')==report['pairs_sha256']
    original=json.loads((a.data/'inference_dev_inputs.json').read_text());rows=expand(original)
    for row in original:assert digest(a.data/row['reference'])==row['reference_sha256']
    a.output.mkdir(parents=True);write(a.output/'inference_inputs.json',rows)
    write(a.output/'report.json',dict(complete=True,schema='scene_expanded_development_v1',source_commit=os.environ['T12_SOURCE_COMMIT'],
        source_sha256=digest(Path(__file__)),argv=sys.argv,prompt_profile=SHORT_PROFILE,request_sha256=digest(a.output/'inference_inputs.json'),
        source_DEV_request_sha256=digest(a.data/'inference_dev_inputs.json'),pairs_sha256=report['pairs_sha256'],
        seeded_requests=128,reference_views=16,products=8,canonical_scenes=8,original_32_requests_unchanged=True,
        same_source_noise_across_counterfactuals=True,model_inference_performed=False,targets_or_feature_tensors_opened=False,
        TEST_opened=False,only_original_DEV_references=True,not_independent_generalization=True))


if __name__=='__main__':main()
