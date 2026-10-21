from copy import deepcopy
from LightGenV2.tasks.t06_video_quality_assessment.reports.reaccount_mps_ready_input import replace_clip_only


def test_existing_batch10_selection_preserves_all_other_models_and_raw_input():
    prior=dict(points=[dict(model=m,resource_cap=c,batch=8 if m=='clip' else 1,
        time16_ms=1188.3861464262009 if m=='qwen' and c==100 else 200,
        energy16_j=462.1134010767563 if m=='qwen' and c==100 else 80,
        repeats=[dict(checkpoint_sha256='same',parameter_dtypes=['fp16'],readout_dtypes=['fp32'])]*3)
        for m in ('qwen','glm','clip','yolo','deepseek','kimi') for c in (100,50)])
    before=deepcopy(prior)
    groups={cap:[dict(model='clip',batch=10,resource_cap=cap,repeat=r,time16_ms=cap+r,
        energy16_j=cap/10+r,srcc=.72 if cap==100 else .71,checkpoint_sha256='same',
        parameter_dtypes=['fp16'],readout_dtypes=['fp32'],source='record',source_sha256='digest')
        for r in (1,2,3)] for cap in (100,50)}
    data=replace_clip_only(prior,groups,{100:95,50:99})
    assert prior==before
    assert all(p==q for p,q in zip(data['points'],prior['points']) if p['model']!='clip')
    clips=[p for p in data['points'] if p['model']=='clip']
    assert [p['batch'] for p in clips]==[10,10]
    assert [p['srcc'] for p in clips]==[.72,.72]
    assert [p['measured_srcc'] for p in clips]==[.72,.71]
    assert [p['time16_ms'] for p in clips]==[102,52]
    assert len(data['repeats'])==36
