import copy
import pytest
import torch
from LightGenV2.tasks.t19_resnet20_baselines.openmoji_validation_repair import grouped_indices,regularize_features,select_candidate,validate_gate,candidate_contract,candidate_loss


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


def test_preservation_ablation_does_not_relax_old_candidate_contract():
    from pathlib import Path
    import json
    root=Path(__file__).parents[1]
    old=json.loads((root/'configs/openmoji_validation_repair_20261024.json').read_text())
    new=json.loads((root/'configs/openmoji_equal_preservation_20261024.json').read_text())
    assert len(candidate_contract(old))==3
    assert candidate_contract(new)[0]['preserve_weight']==1.
    other=copy.deepcopy(new);other.pop('profile')
    with pytest.raises(ValueError):candidate_contract(other)
    other=copy.deepcopy(new);other['candidates'][0]['preserve_weight']=.6
    with pytest.raises(ValueError):candidate_contract(other)


def test_native_loss_control_exactly_matches_original_task_loss():
    from types import SimpleNamespace
    from experiments.qwen3_vl_2b_openmoji_instruction_four_stage_optical_editing.objectives import editing_objective
    settings=SimpleNamespace(changed_cell_weight=12.,foreground_cell_weight=3.,category_loss_weight=1.,
        edit_loss_weight=1.25,preservation_loss_weight=.15,task_loss_weight=.1,ccd_loss_weight=0.,router_balance_weight=0.)
    torch.manual_seed(42)
    output=dict(category_logits=torch.randn(2,17,6,6,requires_grad=True),edit_logits=torch.randn(2,6,6,requires_grad=True),task_logits=torch.randn(2,4,requires_grad=True))
    batch=dict(target_grid=torch.randint(0,17,(2,6,6)),edit_grid=torch.randint(0,2,(2,6,6)).float(),task_index=torch.tensor([0,1]))
    batch['preserve_grid']=1-batch['edit_grid'];candidate=dict(objective='native_original')
    zero=torch.zeros(())
    expected=editing_objective(dict(output,ccd_operating_loss=zero,router_balance_loss=zero),batch,settings)['total']
    actual=candidate_loss(output,batch,candidate,settings)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0);actual.backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in output.values())


def test_latest_GPU_authorization_rejects_GPU6_and_allows_GPU1():
    from LightGenV2.tasks.t19_resnet20_baselines.optimization_shared_worker import require_current_gpu_authorization
    result=require_current_gpu_authorization('GPU-e8837b85-d55b-8e81-aaa5-ec1ac326932d')
    assert result['GPU6_prohibited'] and set(result['authorized_indices'])=={1,2,4,5}
    with pytest.raises(RuntimeError):require_current_gpu_authorization('GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55')
