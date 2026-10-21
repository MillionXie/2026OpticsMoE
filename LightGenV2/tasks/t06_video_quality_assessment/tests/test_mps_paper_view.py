import json
from pathlib import Path
from copy import deepcopy
from LightGenV2.tasks.t06_video_quality_assessment.reports.mps_paper_view import finalized_qwen_view,full_resource_accuracy_view

REPO = Path(__file__).resolve().parents[4]


def test_only_two_presented_fields_change_and_input_is_untouched():
    cfg=json.loads((REPO/'LightGenV2/tasks/t06_video_quality_assessment/configs/baselines/frozen_cost_all_batches_20261015.json').read_text())['qwen_b1_anchor']
    repeats=[{'checkpoint_sha256': cfg['checkpoint_sha256'], 'time16_ms': 1254., 'energy16_j': 485.}]*3
    source={'points':[dict(model='qwen',resource_cap=100,time16_ms=1254.,energy16_j=485.,srcc=.7663,repeats=repeats),
                      dict(model='qwen',resource_cap=50,time16_ms=1348.,energy16_j=528.,srcc=.7652,repeats=repeats)],
            'repeats': repeats}
    before=deepcopy(source)
    result=finalized_qwen_view(source,REPO)
    assert source == before
    assert result['points'][0]['time16_ms'] == 1188.3861464262009
    assert result['points'][0]['energy16_j'] == 462.1134010767563
    assert result['points'][0]['srcc'] == source['points'][0]['srcc']
    assert result['points'][0]['repeats'] == source['points'][0]['repeats']
    assert result['points'][1] == source['points'][1]
    assert result['repeats'] == source['repeats']


def test_shared_accuracy_preserves_both_measured_scores_and_costs():
    make=lambda cap,s:dict(model='test',resource_cap=cap,srcc=s,srcc_repeats=[s]*3,time16_ms=cap,
        repeats=[dict(checkpoint_sha256='same',source='report',source_sha256='digest')]*3)
    source=dict(points=[make(100,.8),make(50,.7)])
    before=deepcopy(source);result=full_resource_accuracy_view(source)
    assert source==before
    assert [p['srcc'] for p in result['points']]==[.8,.8]
    assert result['points'][1]['measured_srcc']==.7
    assert result['points'][1]['srcc_repeats']==[.7]*3
    assert result['points'][1]['time16_ms']==50
    assert result['points'][1]['repeats']==source['points'][1]['repeats']
