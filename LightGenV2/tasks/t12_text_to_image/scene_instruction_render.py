"""Explicit new-dataset preflight using the unchanged full electronic renderer.

Default historical render guards are not relaxed. This entry only accepts the
new visible-anchor DEV request manifest, with all original views and no targets.
"""
import json
from pathlib import Path
from .scene_instruction_data import PROFILE, SAFE, condition, sha


def checked_balanced_input_identity(cache_report,state,identity,identity_sha):
    if not cache_report.get('complete')or cache_report.get('cache_sha256')!=state['cache_sha256']or cache_report.get('DEV_input_identity_sha256')!=identity_sha:
        raise ValueError('Actual trained cache and input-only identity SHA must match')
    if identity.get('contains_targets_or_masks')is not False or identity.get('Qwen_input_side')!=state['Qwen_input_side']or len(identity.get('features',{}))!=48:
        raise ValueError('Full48 raw input identities, without targets/masks, required')
    return True


def checked_revision(folder,data,state,original):
    from .scene_instruction_balanced_data import assert_admission_current
    assert_admission_current(data)
    report=json.loads((folder/'report.json').read_text(encoding='utf8'))
    balanced=report['schema']=='scene_balanced_visible_anchor_dataset_v1'
    if (not report['complete'] or report['schema']not in ['scene_visible_anchor_dataset_v1','scene_balanced_visible_anchor_dataset_v1']
        or report['prompt_profile']!=PROFILE or report['TEST_opened'] or not report['role_disjoint']):
        raise ValueError('Require the explicit completed revised dataset')
    if balanced and(not report.get('eligible_for_training')or not report.get('all_reference_conditions_covered')or report.get('TRAIN_products')!=44 or report.get('TRAIN_reference_views')!=44 or report.get('TRAIN_condition_groups')!=132 or state['pair_sha256']!=report['pairs_sha256']or report.get('native_caption_style')!='public_fact_equivalent_description_v1'):
        raise ValueError('Balanced rendering requires complete reviewed coverage and its actual trained model')
    if state['pair_sha256'] not in [report['source_pairs_sha256'],report['pairs_sha256']]:raise ValueError('Unrelated model/data parent')
    actual=json.loads((folder/'inference_inputs.json').read_text(encoding='utf8'))
    if report['request_sha256']!=sha(folder/'inference_inputs.json'):raise ValueError('Changed requests')
    if len(actual)!=48 or not all(set(r)==set(SAFE)for r in actual):raise ValueError('Full target-free DEV48 only')
    refs={r['reference_id']:r for r in original};expected=[]
    if len(refs)!=16:raise ValueError('All sixteen original views must remain')
    for rid,ref in sorted(refs.items()):
        for code in (['B','D','W']if ref['category']=='lamp'else['S','A','D']):
            definition=condition(ref['category'],code);row={k:ref[k]for k in SAFE}
            row.update({k:definition[k]for k in ['condition_id','prompt','native_caption','reward_caption','prompt_profile','category']})
            if balanced:
                from .scene_instruction_native_prior import visible_caption
                row['native_caption']=visible_caption(ref['category'],code)
            row['sample_id']=rid+'-visible-'+code+'-draw2';expected.append(row)
    if actual!=expected:raise ValueError('Do not silently select easy references/prompts/seeds')
    for row in actual:
        if sha(data/row['reference'])!=row['reference_sha256']:raise ValueError('Input pixels changed')
    return actual,dict(schema='scene_visible_anchor_DEV_preflight_v1',prompt_profile=PROFILE,
        contexts=48,reference_views=16,source_pairs_sha256=report['source_pairs_sha256'],
        request_sha256=report['request_sha256'],TEST_opened=False,
        prior_model_on_new_prompts_not_retrained_model=state['pair_sha256']==report['source_pairs_sha256'],
        model_trained_on_revised_dataset=state['pair_sha256']==report['pairs_sha256'],
        balanced_counterfactual_training_data=balanced,
        new_task_results_not_old_score_relabeling=True)


def main():
    from . import scene_natural_prompt as prompts
    from . import scene_paired_render as renderer
    old=prompts.checked_revision;prompts.checked_revision=checked_revision
    try:renderer.main()
    finally:prompts.checked_revision=old


if __name__=='__main__':main()
