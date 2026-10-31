"""Raw full48 v7 rendering for a model trained on reviewed new v7 targets."""
import json
from .scene_instruction_data import SAFE,sha
from .scene_paper_prompt_protocol import PROFILE,condition


def checked_revision(folder,data,state,original):
    r=json.loads((folder/'report.json').read_text(encoding='utf8'));actual=json.loads((folder/'inference_inputs.json').read_text(encoding='utf8'))
    if (folder.resolve()!=data.resolve()or r.get('schema')!='scene_paper_paired_dataset_v1'or not r['complete']or not r['eligible_for_training']or r['prompt_profile']!=PROFILE or not r['role_disjoint']or r['TEST_opened']or state['pair_sha256']!=r['pairs_sha256']or sha(folder/'pairs.json')!=r['pairs_sha256']or sha(folder/'inference_inputs.json')!=r['request_sha256']):
        raise ValueError('Actual new v7 model and reviewed paired dataset must match')
    refs={r['reference_id']:r for r in original};expected=[]
    if len(refs)!=16 or len(actual)!=48:raise ValueError('Keep all original16 DEV sources and48 requests')
    for rid,ref in sorted(refs.items()):
        for i in range(3):
            q={k:ref[k]for k in SAFE};d=condition(ref['category'],i);q.update({k:d[k]for k in ['prompt','native_caption','reward_caption','prompt_profile','condition_id']});q['sample_id']=rid+'-paper-scene'+str(i)+'-draw2';expected.append(q)
    if actual!=expected or any(set(q)!=set(SAFE)or sha(data/q['reference'])!=q['reference_sha256']for q in actual):raise ValueError('Changed public prompt/source/seed or selected subset')
    return actual,dict(schema='scene_paper_scene_trained_DEV_v1',prompt_profile=PROFILE,contexts=48,reference_views=16,request_sha256=r['request_sha256'],paired_dataset_sha256=r['pairs_sha256'],TEST_opened=False,model_trained_on_revised_dataset=True,old_results_not_relabelled=True,new_targets_reviewed=True,balanced_counterfactual_training_data=True,full_TRAIN_condition_coverage=not r['missing_TRAIN_source_condition_groups'])


def main():
    from . import scene_natural_prompt as prompts
    previous=prompts.checked_revision;prompts.checked_revision=checked_revision
    try:
        from .scene_paired_render import main as render
        render()
    finally:prompts.checked_revision=previous


if __name__=='__main__':main()
