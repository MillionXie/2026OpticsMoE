import copy
import pytest
import torch
from LightGenV2.tasks.t19_resnet20_baselines.openmoji_validation_repair import grouped_indices,regularize_features,select_candidate,validate_gate


def test_fold_groups_all_equal_source_scenes_together_and_is_deterministic():
    rows=[dict(source_grid=[[i]],instruction=str(j)) for i in range(40) for j in range(3)]
    a,b,groups=grouped_indices(rows)
    assert len(a)>0 and len(b)>0 and set(a).isdisjoint(b) and set(a)|set(b)==set(range(120))
    assert grouped_indices(copy.deepcopy(rows))==(a,b,groups)
    for i in range(40):assert set(range(3*i,3*i+3)).issubset(a) or set(range(3*i,3*i+3)).issubset(b)


def test_dropout_only_training_preserves_inference_features_exactly():
    x=torch.ones(256,32,3,3)
    assert torch.equal(regularize_features((x,),.2,False)[0],x)
    torch.manual_seed(42);y=regularize_features((x,),.2,True)[0]
    assert 0.15<float(y.eq(0).float().mean())<.25
    assert torch.equal(y[:,:,0,0],y[:,:,2,2])
    assert abs(float(y.mean())-1)<.04


def test_selection_uses_validation_only_with_predeclared_ties():
    def r(name,score,loss,TEST):return dict(candidate={'id':name},validation_metrics={'overall':{'changed_cell_accuracy':score}},validation_loss=loss,TEST=TEST)
    a=r('a',.6,1.,.99);b=r('b',.7,2.,.01)
    assert select_candidate([a,b])==b
    b['validation_metrics']['overall']['changed_cell_accuracy']=.6
    assert select_candidate([a,b])==a
    b['validation_loss']=1.
    assert select_candidate([b,a])==a


def test_probe_gate_binds_fold_and_rejects_TEST_observation_or_source_change():
    m={k:k for k in ('profile_SHA','source_sha256','native_source_sha256','initial_trainable_SHA','fold_SHA')}
    gate=dict(m,status='complete',gate_passed=True,test_evaluations=0);validate_gate(gate,m)
    for key in m:
        other=dict(gate);other[key]='bad'
        with pytest.raises(ValueError):validate_gate(other,m)
    other=dict(gate,test_evaluations=1)
    with pytest.raises(ValueError):validate_gate(other,m)
