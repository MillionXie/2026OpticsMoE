from copy import deepcopy
import pytest
from LightGenV2.tasks.t06_video_quality_assessment.reports.temporal_fp32_reference import select,MODELS,audit


def fixture():
    points=[]
    for model in MODELS:
        for condition in ('unrestricted','busy50'):
            points.append(dict(model=model,condition=condition,precision='fp32',method='temporal',
                batch=1,checkpoint_sha256=model,accuracy_source_sha256=model+'accuracy',
                srcc_reference=.75,gpu_count=2 if model=='deepseek' else 1,
                repeats=[{}, {}, {}],target50_reached=True))
    return dict(status='all48_complete',accounting_protocol=audit.ACCOUNTING,values=points)


def test_select_exact_twelve_fp32_temporal_conditions_without_mutation():
    data=fixture();before=deepcopy(data);result=select(data)
    assert data==before and len(result)==12
    assert all(p['method']=='temporal' and p['precision']=='fp32' for p in result)


def test_mismatched_accuracy_or_batch_rejected():
    data=fixture();data['values'][1]['srcc_reference']=.7
    with pytest.raises(AssertionError):select(data)
    data=fixture();data['values'][1]['batch']=2
    with pytest.raises(AssertionError):select(data)


def test_no_silent_capacity_or_control_failure():
    data=fixture();data['values'][-1]['target50_reached']=False
    with pytest.raises(AssertionError):select(data)
    data=fixture();next(p for p in data['values'] if p['model']=='deepseek')['gpu_count']=1
    with pytest.raises(AssertionError):select(data)
