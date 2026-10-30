"""Target-free, audited natural-language prompt revision on existing DEV inputs."""
import argparse
import json
import os
from pathlib import Path
from .scene_development_grid import expand
from .appearance_scene_protocol import SHORT_PROFILE
from .white_reference_scene_probe import digest,write

PROFILE='two_sentence_natural_room_v4'
SUFFIX='. Keep its overall shape visible; other details may vary.'


def revise(row):
    prompt=row['prompt']
    if row['prompt_profile']!=SHORT_PROFILE or not prompt.endswith(SUFFIX):
        raise ValueError('Require the frozen previous two-sentence request')
    content=prompt[:-len(SUFFIX)]
    if content.count('the reference ')!=1 or ' with ' not in content:
        raise ValueError('Unexpected source wording; do not silently drop attributes')
    content=content.replace('the reference ','a ',1)
    scene,details=content.rsplit(' with ',1)
    new=scene+'. Include '+details+' in the scene.'
    if any(s in new.lower() for s in ['reference','overall shape','details may vary','fine-tun','finetun']):
        raise ValueError('Reference/implementation instructions must not appear in the public prompt')
    return dict(row,prompt=new,native_caption=new,prompt_profile=PROFILE)


def checked_revision(folder,data,state,original):
    report=json.loads((folder/'report.json').read_text());path=folder/'inference_inputs.json'
    if not (report['complete'] and report['schema']=='scene_natural_prompt_DEV_v1'
        and not report['TEST_opened'] and report['reference_RGB_still_input']):
        raise ValueError('Require the explicit completed natural-prompt DEV preparation')
    if report['source_DEV_request_sha256']!=digest(data/'inference_dev_inputs.json'):
        raise ValueError('DEV source changed')
    if report['pairs_sha256']!=state['pair_sha256']:
        raise ValueError('Different paired training provenance')
    actual=json.loads(path.read_text());expected=[revise(q) for q in expand(original)]
    if digest(path)!=report['request_sha256'] or actual!=expected:
        raise ValueError('Natural revision changed reference/noise/split or lost scene attributes')
    for q in actual:
        if digest(data/q['reference'])!=q['reference_sha256']:raise ValueError('Reference pixels changed')
    return actual,dict(manifest_sha256=digest(folder/'report.json'),request_sha256=digest(path),
        source_DEV_request_sha256=report['source_DEV_request_sha256'],contexts=128,reference_views=16,
        prompt_profile=PROFILE,reference_RGB_still_input=True,prompt_revision_only=True,TEST_opened=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and len(os.environ.get('T12_SOURCE_COMMIT',''))==40
    previous=json.loads((a.data/'report.json').read_text());assert previous['complete'] and previous['role_disjoint']
    original=json.loads((a.data/'inference_dev_inputs.json').read_text());rows=[revise(q) for q in expand(original)]
    for q in rows:assert digest(a.data/q['reference'])==q['reference_sha256']
    a.output.mkdir(parents=True);write(a.output/'inference_inputs.json',rows)
    write(a.output/'report.json',dict(complete=True,schema='scene_natural_prompt_DEV_v1',source_commit=os.environ['T12_SOURCE_COMMIT'],
        prompt_profile=PROFILE,source_DEV_request_sha256=digest(a.data/'inference_dev_inputs.json'),
        pairs_sha256=previous['pairs_sha256'],request_sha256=digest(a.output/'inference_inputs.json'),
        reference_RGB_still_input=True,reference_pixels_and_seeds_unchanged=True,contexts=128,reference_views=16,
        products=8,canonical_scenes=8,targets_or_features_opened=False,model_inference_performed=False,TEST_opened=False,
        inference_only_revision=True,old_training_prompts_and_weights_preserved=True))


if __name__=='__main__':main()
