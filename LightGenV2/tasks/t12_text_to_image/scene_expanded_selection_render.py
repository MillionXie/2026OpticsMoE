"""Actual raw Qwen RGB/text TEST500; explicitly used for PT selection, not frozen TEST.

The common renderer's revised-request mechanism is reused, with a separate
public scope/provenance. Neither target pixels nor training feature PTs are
opened for conditioning. All500 outputs remain in the scored denominator.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from .scene_instruction_data import SAFE,sha
from .scene_paper_expanded_data import PROFILE,conditions
from .scene_expanded_paired_data import PAIRED_SCHEMA
from .scene_paper_prompt_protocol import write


def checked_revision(folder,data,state,original):
    report=json.loads((folder/'report.json').read_text());rows=json.loads((folder/'inference_inputs.json').read_text())
    protocol=json.loads((folder/'selection_protocol.json').read_text())
    if(folder.resolve()!=data.resolve() or report.get('schema')!=PAIRED_SCHEMA or not report['complete']
       or not report['eligible_for_training'] or report['validation_set'] or not report['TEST_used_for_model_selection']
       or report['prompt_profile']!=PROFILE or state['pair_sha256']!=report['pairs_sha256']
       or not state.get('no_validation_test_PT_selection') or sha(folder/'inference_inputs.json')!=report['request_sha256']
       or protocol['test_request_sha256']!=report['request_sha256'] or not protocol['TEST_used_for_model_selection']):
        raise ValueError('Actual expanded TRAIN model and explicit TEST-selection protocol required')
    if len(rows)!=500 or len({r['sample_id'] for r in rows})!=500 or Counter(r['category'] for r in rows)!={'lamp':250,'pillow':250}:
        raise ValueError('Complete fixed balanced TEST500 required')
    for row in rows:
        definition=conditions(row['category'])[int(row['condition_id'])]
        if set(row)!=set(SAFE) or sha(data/row['reference'])!=row['reference_sha256']:
            raise ValueError('Public target-free input identity changed')
        if any(row[k]!=definition[k] for k in ['prompt','native_caption','reward_caption','prompt_profile']):
            raise ValueError('Canonical facts changed after plan freeze')
    return rows,dict(schema='scene_expanded_TEST_selection_v1',prompt_profile=PROFILE,contexts=500,reference_views=26,
        request_sha256=report['request_sha256'],paired_dataset_sha256=state['pair_sha256'],validation_set=False,
        TEST_used_for_model_selection=True,independent_generalization_claim=False,balanced_counterfactual_training_data=False,
        model_trained_on_expanded_TRAIN=True)


def public_report(output):
    report=json.loads((output/'report.json').read_text());protocol=report['expanded_DEV_protocol']
    if protocol.get('schema')!='scene_expanded_TEST_selection_v1':raise ValueError('Wrong requested cohort')
    timing=report.get('scope')=='DEV_software_latency'
    # Common revised-request machinery is internal only. Expose the actual
    # selection scope, including in every per-image record and final report.
    images=json.loads((output/'images.json').read_text())
    for row in images:
        row.update(split='test',checkpoint_sha256=report['checkpoint_sha256'],TEST_used_for_model_selection=True)
    write(output/'images.json',images)
    report.update(scope='TEST_selection_timing' if timing else 'TEST_selection',validation_set=False,
        TEST_used_for_model_selection=True,independent_TEST_opened=False,independent_generalization_claim=False,
        TEST_selection_protocol=protocol,full500_scored=not timing)
    write(output/'report.json',report)


def main():
    from . import scene_natural_prompt as prompts
    from . import scene_paired_render as renderer
    old_check=prompts.checked_revision;old_render=renderer.render
    def render(a):
        if a.scope!='dev' or a.prompt_revision is None:raise ValueError('Use explicit new selection revision entry')
        old_render(a);public_report(a.output)
    prompts.checked_revision=checked_revision;renderer.render=render
    try:renderer.main()
    finally:prompts.checked_revision=old_check;renderer.render=old_render


if __name__=='__main__':main()
