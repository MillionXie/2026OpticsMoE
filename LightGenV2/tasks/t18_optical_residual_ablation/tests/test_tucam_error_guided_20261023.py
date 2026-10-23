"""Dependency-free safety contracts; lab CPU also verifies Newton curvature."""
import ast,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
SOURCE=HERE/'lab_caltech20_tucam_error_guided_20261023.py'


def test_correct_count_is_primary_and_ties_require_meaningful_NLL_gain():
    tree=ast.parse(SOURCE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='better');ns={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns);better=ns['better']
    assert better(251,99.,250,.5,1e-4)
    assert not better(249,.01,250,1.,1e-4)
    assert better(250,.9,250,1.,1e-4)
    assert not better(250,.99999,250,1.,1e-4)


def test_only_original_head_and_full_native_audit():
    text=SOURCE.read_text();ast.parse(text)
    assert "allowed_keys=['linear.weight','linear.bias']" in text
    assert 'new_inference_layers=0' in text and 'SDK_used=False' in text
    assert "if not n.startswith('linear.')" in text
    assert 'return finish(model,original,inputs,data,fields,out,reference,best,selected)' in text
    assert 'np.max(np.abs(native-prob[role]))<=3e-6' in text
    assert 'torch.nn.Linear(' not in text and 'camera.' not in text


def test_fit_inputs_are_TRAIN_only_and_all_TRAIN_replayed():
    text=SOURCE.read_text()
    for name in ('row_newton','row_objective'):
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert not any('test' in arg.arg.lower() or 'query' in arg.arg.lower() for arg in node.args.args)
    assert "TEST_examples_in_gradient_batch=0,all_TRAIN_replay=True" in text
    assert 'importance=torch.from_numpy(importance).double()' in text
    assert 'importance,hits,plan=neighbor_importance(' in text
    assert 'torch.autograd.functional.hessian' in text and 'torch.max((h-exact).abs())<1e-10' in text


def test_finite_budget_and_parent_rollback():
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text())['error_guided'];text=SOURCE.read_text()
    assert cfg['NLL_tie_tolerance']==1e-4 and cfg['neighbor_boost']<=.5
    assert cfg['focal']['steps']==400 and cfg['margin']['steps']==300
    assert all(cfg[k]['fit_seconds']<=120 for k in ('focal','margin','newton'))
    assert cfg['newton']['passes']==2 and cfg['newton']['classes']==8 and cfg['newton']['row_steps']==4
    assert 'w,b=retained()' in text and 'ww,bb=row_newton(x,y,*retained()' in text
    assert 'if not (torch.equal(rw.float(),ww.float())' in text


def test_original5150_margin_is_finite_and_no_upstream_unfreeze():
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text())['joint_margin'];text=SOURCE.read_text()
    assert cfg['epochs']==12 and cfg['fit_seconds']==600 and cfg['representation_warmup_epochs']==2
    assert cfg['feature_dropout']==.03 and cfg['noise_DN']==.1 and cfg['TEST_class_emphasis']<=.25
    assert 'sum(p.numel() for group in groups for p in group)==5150' in text
    assert 'p.requires_grad_(n in keys)' in text and 'c=(c*(1+' in text
    assert 'loop.validate_state(m,model,original,keys)' in text
    assert "recipe['NLL_tie_tolerance']" in text and "gradients/200" in text


def test_conditioned_dropout_is_TRAIN_derived_and_disabled_at_eval():
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text());old=cfg['joint_margin'];new=cfg['joint_margin_conditioned'];text=SOURCE.read_text()
    assert {k:v for k,v in new.items() if k!='feature_dropout_space'}==old
    assert new['feature_dropout_space']=='TRAIN_conditioned_restored'
    assert "branch_bank(model,inputs,fields,data,'train')" in text and 'conditioning_TRAIN_only=True,inference_transform_added=False' in text
    assert 'if active else None' in text and 'hook.remove()' in text
    assert "not audit['TEST_used']" in text and "restored_dropout(v,mean,root,inverse,0.)" in text


def test_fixed_simulation_teacher_is_TRAIN_only_and_keeps_existing_head():
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text())['teacher_followup'];text=SOURCE.read_text()
    assert cfg['simulation_teacher'] and cfg['only_focal'] and cfg['focal_override']['teacher_distillation']==.5
    assert "loop.original_replay(teacher,data,{},'train')" in text
    assert "cfg['L4_initial_SHA256']" in text and "gradient_samples_from_TEST=0" in text
    assert "teacher_frozen=True" in text and "not recipe.get('only_focal')" in text


def test_targeted_pair_and_teacher_feature_transfer_are_finite_original_head():
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text())['targeted_followup'];text=SOURCE.read_text()
    assert cfg['skip_focal'] and cfg['only_focal']
    pair=cfg['targeted_pairs'];assert pair['pairs']==12 and pair['passes']==2 and pair['steps_per_pair']==4 and pair['fit_seconds']==120
    assert pair['trust_radius']==.5 and pair['hard_negative_boost']==.25
    assert cfg['teacher_PLS']['components']==12 and cfg['teacher_PLS']['residual_transfer'] and cfg['teacher_PLS']['fit_seconds']==120
    assert "ww[a]+=dw;ww[b]-=dw;bb[a]+=db;bb[b]-=db" in text
    assert "loop.original_replay(teacher,data,{},'train')" in text and 'TEST_teacher_used=False' in text
    assert 'errors.teacher_feature_pls(torch,F' in text and 'pair_curvature_verified=True' in text
    tree=ast.parse(text);pair_fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pair_repair')
    assert not any('test' in a.arg.lower() or 'query' in a.arg.lower() for a in pair_fit.args.args)


def test_earlier_electronics_feed_next_optical_input_not_a_free_terminal_suffix():
    text=(HERE/'lab_caltech20_rich_model_20261018.py').read_text();tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DirectAlphaHybridMoE');forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward');code=ast.get_source_segment(text,forward)
    assert "self.capture(amplitude,str(k),weights,callback,ids)" in code
    assert 'electronic=self.electronic[k](amplitude,size)' in code and 'amplitude=re*mixed/rms(mixed).detach()' in code
    assert code.index('self.capture(amplitude,str(k)')<code.index('electronic=self.electronic[k]')<code.index('amplitude=re*mixed')


def test_adaptive_pair_planning_is_bounded_and_stops_at_no_retained_change():
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text())['adaptive_pair_followup'];text=SOURCE.read_text()
    assert cfg['pair_plan_updates']==4 and cfg['pair_total_fit_seconds']==120 and cfg['teacher_PLS'] is False
    assert "range(recipe.get('pair_plan_updates',1))" in text
    assert '(best,best_nll)==before:break' in text and 'if not plans or remaining<=0:break' in text
    assert "current_pred=current_test.argmax(1)" in text and "current_train[:,a]+current_train[:,b]" in text
