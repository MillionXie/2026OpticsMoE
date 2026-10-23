import importlib.util,json
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
ENTRY=HERE/'lab_caltech20_TRAIN_flat_response_20261024.py'
spec=importlib.util.spec_from_file_location('flat_response',ENTRY);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)


def test_policy_fixed_TRAIN_only_crossphase_no_hardware_or_gate_change():
    p=json.loads((HERE/'lab_caltech20_TRAIN_flat_response_profile_20261024.json').read_text());e.validate_policy(p)
    for k,v in [('fit_condition','nominal_before'),('TEST_used',True),('grid',16),('intermediate_target',273),('new_CCD',208),('hardware_changed',True)]:
        with pytest.raises(AssertionError):e.validate_policy({**p,k:v})


def test_exact_box_QP_known_map_and_identity():e.selftest()


def test_box_QP_not_worse_than_feasible_dense_grid():
    rng=np.random.default_rng(73);p=rng.uniform(0,3,(20,1,1));o=np.maximum(0,-p+.2)
    g,b=e.fit_gain_bias(p,o,.05)
    def loss(g,b):return ((g*p+b-o)**2).mean()+.05*((g-1)**2+b*b)
    brute=min(float(loss(x,y)) for x in np.linspace(.05,4,100) for y in np.linspace(0,4,100))
    assert float(loss(g,b))<=brute+1e-12


def test_flat_only_fit_unseen_phase_admission_and_no_classification_or_camera():
    s=ENTRY.read_text();run=s.split('def run(',1)[1].split('def selftest(',1)[0]
    assert "v['condition']=='flat'" in run and "v['split']=='fit'" in run
    assert "fit_gain_bias(pp,oo,.05)" in run and "residual_PCC>=.8" in run
    assert "('held_flat','held_unseen_phase','held_nominal_before')" in run
    assert "('nominal_before','half_code','global_offset128')" in run
    for bad in ('Bench(','capture(',"['test']",'cross_entropy','torch.save','saved_model','set_exposure'):
        assert bad not in run
    assert 'unadmitted_prior_nuisance_fits_used=False' in run and 'model_checkpoint_written=False' in run


def test_nominal_original_physics_and_no_image_warp_in_response_map():
    s=ENTRY.read_text();body=s.split('def apply_map(',1)[1].split('def run(',1)[0]
    assert 'F.interpolate' in body and "mode='bilinear',align_corners=False" in body
    assert 'grid_sample' not in body and 'warp' not in body
    assert 'nominal=torch.tensor([1.,1.,0.,0.])' in s
