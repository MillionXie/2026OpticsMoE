import ast
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]


def test_gradient_objective_has_no_development_inputs():
    tree=ast.parse((HERE/'lab_caltech20_TRAIN_hardreadout_20261024.py').read_text())
    fun=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='objective')
    assert [v.arg for v in fun.args.args]==['logits','labels','importance','parent_prob','delta','bias_delta']
    assert not any(isinstance(v,ast.Name)and v.id in ('development','development_labels','zy','z')for v in ast.walk(fun))


def test_optimizer_only_folds_original_head():
    src=(HERE/'lab_caltech20_TRAIN_hardreadout_20261024.py').read_text()
    assert 'delta@precondition' in src and "'new_inference_layers':" not in src
    assert "new_inference_layers=0" in src and 'canonical_affine_logits' in src
    assert 'TRAIN_consumed=1600' in src and 'time.monotonic()-started>=c' in src
    assert 'model.' not in src and 'cuda' not in src.lower()


def test_parent_and_lineage_admission_preserved():
    src=(HERE/'lab_caltech20_fullG2_OAS_20261024.py').read_text()
    assert '105a0f517cd8dda1a4e5bb7f991505476ead763784e828a663921995f7cbf1a6' in src
    assert 'ca[\'matched_samples\']==2011' in src and "loop.validate_state" in src
    ast.parse(src)


def test_joint_followup_uses_completed_new_best_not_last():
    src=(HERE/'lab_caltech20_fullG2_postOAS_20261024.py').read_text()
    assert "if a.hard_best:source=prefix/'fullG2_sameclass_augment_20261024'" in src
    assert "sr['selected_correct']>=248" in src and "pt=source/'best.pt'" in src
    assert 'epochs=60,seconds=480' in src and 'TRAIN_teacher_margin_MSE=True' in src
    assert 'm.allowed_keys(model,5)' in src and '==5150' in src
    ast.parse(src)


def test_repair_waits_for_complete_joint_native_best():
    src=(HERE/'lab_caltech20_fullG2_OAS_20261024.py').read_text()
    assert "fullG2_jointbest_" in src and "'fullG2_best_teacher_joint_20261024'" in src
    assert "full['state']=='complete'" in src and "full['strict_original_CPU']" in src
    assert "fit=source/'fit' if a.continued_joint_head=='local' else source" in src
    assert "maximum_classes=20,steps_per_class=64" in src and "outer_steps=80" in src


def test_centered_regularization_only_training_objective():
    src=(HERE/'lab_caltech20_tucam_TRAIN_linear_20261023.py').read_text()
    tree=ast.parse(src);fun=next(v for v in tree.body if isinstance(v,ast.FunctionDef)and v.name=='TRAIN_affine_objective')
    body=ast.get_source_segment(src,fun)
    assert 'weight-weight.mean(0,keepdim=True)' in body and 'class_centered=False' in body
    assert not any(isinstance(v,ast.Name)and v.id in ('test','test_labels','development') for v in ast.walk(fun))
    assert "if cfg.get('class_centered_penalty',False):penalty_weight=penalty_weight-penalty_weight.mean(0,keepdim=True)" in src
    assert 'common_class_row_shift_invariant=True' in src


def test_one_bounded_trust_extension(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    from lab_caltech20_TRAIN_local_boundary_20261023 import validate
    c=dict(maximum_classes=20,steps_per_class=64,temperature=2.,label_smoothing=.02,neighbor_weight=2.,KL=2.,anchor=.2,trust_step=.25,row_radius=.16,seed=73)
    validate(c)
    import pytest
    with pytest.raises(AssertionError):validate({**c,'row_radius':.32})
    src=(HERE/'lab_caltech20_fullG2_OAS_20261024.py').read_text()
    assert "'row_radius':.16" in src and "('trust','cycles')" in src


def test_fixed_rounds_stop_on_plateau_and_keep_global_update(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    import lab_caltech20_TRAIN_local_boundary_20261023 as b
    import numpy as np
    calls=[]
    def fake(*args):
        before=267+len(calls);improvement=1 if not calls else 0;calls.append(1)
        row=dict(update=4,stage=1,selected_update=1 if improvement else 0,recovered=improvement,lost=0)
        return dict(initial_correct=before,initial_NLL=1.,best=before+improvement,nll=.99 if improvement else 1.,selected=1 if improvement else 0,updates=4,history=[row],plans=[dict(stage=1)],classes=[3],best_weight=np.zeros((20,256)),best_bias=np.zeros(20),last_weight=np.zeros((20,256)),last_bias=np.zeros(20))
    monkeypatch.setattr(b,'fit',fake)
    c=dict(maximum_classes=20,steps_per_class=64,temperature=2.,label_smoothing=.02,neighbor_weight=2.,KL=2.,anchor=.2,trust_step=.25,row_radius=.16,seed=73,rounds=3)
    r=b.fit_cycles(None,None,None,None,None,None,c,340,180,lambda row:None)
    assert len(calls)==2 and r['updates']==8 and r['selected']==1 and r['best']==268
    assert r['history'][-1]['selected_update']==1 and r['history'][-1]['update']==8
