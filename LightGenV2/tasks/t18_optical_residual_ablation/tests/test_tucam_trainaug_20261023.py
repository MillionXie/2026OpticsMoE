"""Dependency-free contracts; numerical gradient/folding test runs on lab CPU."""
import ast,json
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
SOURCE=HERE/'lab_caltech20_tucam_trainaug_20261023.py'


def test_only_train_arrays_enter_gradient_loop():
    text=SOURCE.read_text();ast.parse(text)
    assert "inputs['train'][ix]" in text and "fields[str(k)]['train'][ix]" in text
    assert "TEST_examples_in_training=0" in text and 'TEST_gradient=False' in text
    assert 'torch.nn.Linear(' not in text and 'SDK_used=False' in text
    assert 'return finish(model,original,inputs,data,fields,out,reference,best,selected)' in text
    assert 'loop.validate_state(m,model,original,keys)' in text and '==5150' in text


def test_paired_identity_and_native_folding():
    text=SOURCE.read_text()
    assert 'np.array_equal(np.sort(order),np.arange(1600))' in text
    assert 'torch.equal(y[::2],y[1::2])' in text
    assert 'original_input_and_CCD_pairing_preserved=True' in text
    assert 'root.T' in text and 'model.load_state_dict(beststate)' in text
    assert 'hook.remove()' in text and 'if active:return conditioned_logits' in text


def test_two_finite_train_trajectories_not_test_augmentation():
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text())['augmented_joint']
    assert len(cfg['phases'])==2 and cfg['neighbor_boost']<=.5 and cfg['weight_max']==2.
    for r in cfg['phases']:
        assert r['epochs']<=6 and r['fit_seconds']<=240 and r['dropout']<=.01 and r['mix']<=.15
    text=SOURCE.read_text()
    assert 'synthetic_feature_mixup_not_physical_examples=True' in text
    assert "current_bank=branch_bank(model,inputs,fields,data,'train')" in text
    assert "max(.4001001" in text and 'effective_alpha(k)' in text


def test_SAM_is_temporary_TRAIN_weight_perturbation_not_an_inference_layer():
    cfg=json.loads((HERE/'lab_caltech20_tucam_covariance_profile_20261023.json').read_text())['augmented_SAM'];r=cfg['phases'][0]
    assert len(cfg['phases'])==1 and r['epochs']==4 and r['fit_seconds']==300 and len(r['SAM_radii'])==3
    text=SOURCE.read_text();assert 'finally:sam_restore(saved)' in text and 'torch.set_rng_state(view_rng)' in text
    assert 'SAM_group_radius_and_exact_restoration=True' in text
