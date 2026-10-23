import ast
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'lab_caltech20_TRAIN_frozen_readout_20261024.py'


def test_real_suffix_training_with_original_head_frozen():
    s=P.read_text();ast.parse(s)
    assert "if not joint:keys=keys-{'linear.weight','linear.bias'}" in s
    assert "('phase_sets.',c['phase_lr'])" in s and "('electronic.',c['electronic_lr'])" in s and "('fusion_logits.',c['alpha_lr'])" in s
    body=s.split('for begin in range(0,1600,8):',1)[1].split('assert used==1600',1)[0]
    assert "data['train'][1][ix]" in body and "data['test']" not in body
    assert "contrast_logits(held['features'],model.linear.weight,model.linear.bias)" in body
    assert 'original_5140_readout_bitwise_frozen=not joint' in s and 'loop.validate_state(m,native,original,keys)' in s


def test_equivalent_gradient_selftest_and_native_CPU_reports():
    s=P.read_text();assert 'gradient_maxabs' in s and 'error<1e-10 and grad<1e-9' in s
    assert 'transforms.remove();hook.remove()' in s and "for label in ('best','last')" in s and "for role in ('train','test')" in s
    assert 'TEST_gradient=False' in s and 'new_inference_layers=0' in s


def test_teacher_normalized_correct_TRAIN_only_and_loss_capped():
    s=P.read_text();body=s.split('def safe_distillation_loss(',1)[1].split('def selftest',1)[0]
    assert 'teacher.detach()' in body and "t.argmax(1)==labels" in body and '>=.5' in body
    assert '.25*ce.detach()/kd.detach().clamp_min(1e-8)' in body and 'ce.new_tensor(.1)' in body
    assert 'normalized(logits)/2' in body and 'normalized(t)/2' in body
    assert "manifest['role']=='train'" in s and "np.array_equal(z['ids'],data['train'][2])" in s
    assert 'teacher_gradient=False' in s and 'KD_value_fraction_of_CE_at_most=.25' in s


def test_original_conv_penalty_no_activation_or_structure_change():
    s=P.read_text()
    assert "model.electronic[k].local.register_forward_hook" in s and "held[f'pre_tanh_{k}'].abs()-2" in s
    assert '.1*ce.detach()/penalty.detach()' in s and 'ce.new_tensor(.01)' in s and 'TRAIN_original_tanh_statistics' in s
    assert 'parameter_reset=False' in s and 'for handle in saturation_hooks:handle.remove()' in s
    assert 'saturation_penalty_gradient_toward_sensitive_region=True' in s


def test_readonly_TRAIN_backend_audit_no_optimization():
    tree=ast.parse(P.read_text());n=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='TRAIN_backend_audit');body=ast.get_source_segment(P.read_text(),n)
    assert "data['test']" not in body and '.backward(' not in body and 'torch.optim' not in body
    assert 'G2_canonical_changed_pixel_fraction' in body and "collect('cpu')" in body and "collect('cuda')" in body
    assert 'original_state_bitwise_unchanged=True' in body
