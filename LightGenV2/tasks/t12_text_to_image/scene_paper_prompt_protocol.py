"""Two informative sentences: product/support first, a distinct scene fact second.

This prepares requests, not a training corpus or a relabeling of existing images.
New targets require new pixel-bound review before any paired training.
"""
import argparse
import json
from pathlib import Path
import shutil
from .scene_instruction_data import SAFE,sha,write
from .scene_generation_data import request_seed

PROFILE='two_sentence_product_scene_v7'
PARENT_PAIRS='536dac05bb15d7226ad79e7573c886f1af39982c75d9beaa7197b3102f1d3970'
DEFINITIONS={
 'lamp':[
  ('Generate an interior photograph of a table lamp on a wooden desk. Place a bookcase against the wall behind the desk.', 'A table lamp on a wooden desk with a bookcase behind the desk.', ['table lamp','wooden desk supports lamp','bookcase behind desk']),
  ('Generate a bedroom photograph with a table lamp on a bedside table. Include the bed in the background.', 'A table lamp on a bedside table with a bed in the background.', ['table lamp','bedside table supports lamp','bed behind lamp']),
  ('Generate an interior photograph of a table lamp on a table. Include a curtained window in the background.', 'A table lamp on a table with a curtained window in the background.', ['table lamp','table supports lamp','curtained window in background'])],
 'pillow':[
  ('Generate a living-room photograph with a cushion resting on a fabric sofa. Place a leafy floor plant beside the sofa.', 'A cushion on a fabric sofa with a leafy floor plant beside the sofa.', ['separate cushion','fabric sofa supports cushion','leafy floor plant beside sofa']),
  ('Generate a reading-corner photograph with a cushion resting on an armchair. Place a bookcase against the wall behind the chair.', 'A cushion on an armchair with a bookcase behind the chair.', ['separate cushion','armchair with armrest supports cushion','bookcase behind chair']),
  ('Generate a bedroom photograph with a cushion resting on a bed. Place a wooden bedside table next to the bed.', 'A cushion on a bed with a wooden bedside table next to the bed.', ['separate cushion','bed supports cushion','wooden bedside table next to bed'])]}


def condition(category,index):
    prompt,caption,facts=DEFINITIONS[category][int(index)]
    if prompt.count('.')!=2 or len(prompt.split())>30 or any(s in prompt.lower()for s in ['reference','fine-tun','together','lighting direction']):
        raise ValueError('Two informative public sentences required')
    return dict(category=category,condition_id=str(index),prompt=prompt,native_caption=caption,reward_caption=caption,prompt_profile=PROFILE,required_visible_facts=facts)


def prepare(data,output):
    if output.exists():raise ValueError('Keep every prior request profile unchanged')
    report=json.loads((data/'report.json').read_text());assert report['complete']and report['role_disjoint']and sha(data/'pairs.json')==PARENT_PAIRS
    pairs=json.loads((data/'pairs.json').read_text());dev=json.loads((data/'inference_inputs.json').read_text());trainrefs={r['reference_id']:r for r in pairs if r['split']=='train'};devrefs={r['reference_id']:r for r in dev}
    if len(trainrefs)!=44 or len(devrefs)!=16 or {r['reference_sha256']for r in trainrefs.values()}&{r['reference_sha256']for r in devrefs.values()}:
        raise ValueError('Retain the role-disjoint original44TRAIN/16DEV views')
    output.mkdir(parents=True);requests={}
    for split,refs,draws in [('train',trainrefs,[0,1]),('dev',devrefs,[2])]:
        rows=[]
        for rid,ref in sorted(refs.items()):
            src=data/ref['reference'];assert sha(src)==ref['reference_sha256'];dst=output/ref['reference'];dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():shutil.copyfile(src,dst)
            for index in range(3):
                definition=condition(ref['category'],index)
                for draw in draws:
                    row={k:ref[k]for k in SAFE};row.update({k:definition[k]for k in ['prompt','native_caption','reward_caption','prompt_profile','condition_id']})
                    row.update(sample_id=rid+'-paper-scene'+str(index)+'-draw'+str(draw),seed=ref['seed']if split=='dev'else request_seed(ref['reference_sha256'],draw))
                    rows.append(row)
        requests[split]=rows;write(output/(split+'_inputs.json'),rows)
    assert len(requests['train'])==264 and len(requests['dev'])==48
    write(output/'inference_dev_inputs.json',requests['dev']);write(output/'inference_inputs.json',requests['dev'])
    write(output/'prompt_definitions.json',[condition(c,i)for c in ['lamp','pillow']for i in range(3)])
    write(output/'report.json',dict(schema='scene_paper_request_plan_v1',complete=True,prompt_profile=PROFILE,source_pairs_sha256=PARENT_PAIRS,
        source_DEV_request_sha256=sha(data/'inference_inputs.json'),request_sha256=sha(output/'inference_inputs.json'),TRAIN_request_sha256=sha(output/'train_inputs.json'),
        TRAIN_requests=264,TRAIN_references=44,DEV_requests=48,DEV_references=16,role_disjoint=True,TEST_opened=False,
        targets_created=False,eligible_for_training=False,old_target_pixels_not_recaptioned=True,old_generated_images_not_relabelled=True))


def checked_revision(folder,data,state,original):
    r=json.loads((folder/'report.json').read_text());rows=json.loads((folder/'inference_inputs.json').read_text())
    if(r.get('schema')!='scene_paper_request_plan_v1' or not r['complete'] or r['prompt_profile']!=PROFILE or r['TEST_opened']
       or state['pair_sha256']!=r['source_pairs_sha256']or r['source_pairs_sha256']!=PARENT_PAIRS
       or sha(folder/'inference_inputs.json')!=r['request_sha256']or len(rows)!=48):raise ValueError('Explicit full new-prompt preflight only')
    refs={q['reference_id']:q for q in original};expected=[]
    if len(refs)!=16:raise ValueError('Keep all sixteen original DEV views')
    for rid,ref in sorted(refs.items()):
        for i in range(3):
            q={k:ref[k]for k in SAFE};d=condition(ref['category'],i);q.update({k:d[k]for k in ['prompt','native_caption','reward_caption','prompt_profile','condition_id']});q['sample_id']=rid+'-paper-scene'+str(i)+'-draw2';expected.append(q)
    if rows!=expected or not all(set(q)==set(SAFE)and sha(data/q['reference'])==q['reference_sha256']for q in rows):
        raise ValueError('No hidden target metadata, prompt substitutions or easy-view subset')
    return rows,dict(schema='scene_paper_scene_DEV_preflight_v1',prompt_profile=PROFILE,contexts=48,reference_views=16,request_sha256=r['request_sha256'],
        TEST_opened=False,prior_model_on_new_prompts_not_retrained_model=True,model_trained_on_revised_dataset=False,
        balanced_counterfactual_training_data=False,old_results_not_relabelled=True,source_pairs_sha256=PARENT_PAIRS)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.data,a.output)


if __name__=='__main__':main()
