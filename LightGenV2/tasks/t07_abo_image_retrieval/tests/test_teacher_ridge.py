import pytest
import torch
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.metric_readout import ridge_teacher_projection


def fixture():
    generator=torch.Generator().manual_seed(123)
    x=torch.randn(100,384,generator=generator)
    t=torch.randn(100,64,generator=generator)
    w=torch.randn(64,384,generator=generator)*.05
    b=torch.randn(64,generator=generator)*.1
    eligible=torch.arange(100)<80
    return x,t,eligible,w,b


def test_ridge_zero_strength_preserves_parameters_exactly():
    x,t,e,w,b=fixture()
    new_w,new_b,a=ridge_teacher_projection(x,t,e,w,b,.1,0.)
    assert torch.equal(new_w,w) and torch.equal(new_b,b)
    assert a['eligible_rows']==80 and not a['fitted_to_query']
    assert a['normal_equation_relative_residual']<1e-10
    assert a['rotation_orthogonality_error']<1e-12


def test_ridge_is_finite_anchored_and_ignores_ineligible_values():
    x,t,e,w,b=fixture()
    nw,nb,a=ridge_teacher_projection(x,t,e,w,b,.1,.1)
    assert a['updated_target_rmse']<a['source_target_rmse']
    assert a['parameter_delta_frobenius']>0
    assert nw.shape==w.shape and nb.shape==b.shape and not nw.requires_grad
    x[~e]*=100;t[~e]*=-10
    ow,ob,_=ridge_teacher_projection(x,t,e,w,b,.1,.1)
    assert torch.equal(nw,ow) and torch.equal(nb,ob)


def test_ridge_stronger_regularization_reduces_update_norm():
    x,t,e,w,b=fixture()
    a=ridge_teacher_projection(x,t,e,w,b,.1,.1)[2]
    strong=ridge_teacher_projection(x,t,e,w,b,100.,.1)[2]
    assert strong['parameter_delta_frobenius']<a['parameter_delta_frobenius']


def test_ridge_rejects_degenerate_or_trainable_inputs():
    x,t,e,w,b=fixture()
    for ridge,strength in [(0.,.1),(-1.,.1),(.1,2.),(float('nan'),.1)]:
        with pytest.raises(ValueError):
            ridge_teacher_projection(x,t,e,w,b,ridge,strength)
    with pytest.raises(ValueError):
        ridge_teacher_projection(x,t,torch.zeros(100,dtype=torch.bool),w,b,.1,.1)
    with pytest.raises(ValueError):
        ridge_teacher_projection(x,torch.ones_like(t),e,w,b,.1,.1)
    with pytest.raises(ValueError):
        ridge_teacher_projection(x.requires_grad_(),t,e,w,b,.1,.1)
