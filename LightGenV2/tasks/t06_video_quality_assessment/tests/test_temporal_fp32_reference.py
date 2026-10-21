from copy import deepcopy
import pytest
from LightGenV2.tasks.t06_video_quality_assessment.reports.temporal_fp32_reference import select,shared_native_accuracy,MODELS,audit


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


def test_shared_native_reference_never_overwrites_fp32_observations():
    data=dict(values=select(fixture()))
    native=dict(points=[dict(model=m,resource_cap=100,batch=1,srcc=.8,
        repeats=[dict(checkpoint_sha256=m,srcc=.8,source=m,source_sha256=m)]*3) for m in MODELS])
    before=deepcopy(data);native_before=deepcopy(native)
    result=shared_native_accuracy(data,native)
    assert data==before and native==native_before
    for old,new in zip(data['values'],result['values']):
        assert new['display_reference_srcc']==.8 and new['srcc_reference']==.75
        assert all(new[k]==v for k,v in old.items())
    native['points'][0]['repeats'][0]['checkpoint_sha256']='wrong'
    with pytest.raises(AssertionError):shared_native_accuracy(data,native)
