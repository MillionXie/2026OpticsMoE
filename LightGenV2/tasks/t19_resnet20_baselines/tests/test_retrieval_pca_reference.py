import torch
import pytest
from LightGenV2.tasks.t19_resnet20_baselines.retrieval_pca_reference import fit_projection,project


def test_train_only_projection_has_fixed_dimension_and_orthonormal_axes():
    torch.manual_seed(42);train=torch.randn(80,64);transform=fit_projection(train,16)
    assert transform['components'].shape==(16,64)
    torch.testing.assert_close(transform['components']@transform['components'].T,torch.eye(16),atol=1e-6,rtol=0)
    output=project(torch.randn(7,64),transform)
    assert output.shape==(7,16)
    torch.testing.assert_close(output.norm(dim=1),torch.ones(7),atol=1e-6,rtol=0)


def test_projection_does_not_refit_on_query_data():
    train=torch.randn(90,64);transform=fit_projection(train,16);saved={k:v.clone() for k,v in transform.items()}
    a=torch.randn(1,64);first=project(a,transform)
    second=project(torch.cat((a,torch.randn(99,64)*100)),transform)[:1]
    torch.testing.assert_close(first,second)
    assert all(torch.equal(transform[k],v) for k,v in saved.items())


def test_invalid_projection_dimensions_rejected():
    with pytest.raises(ValueError):fit_projection(torch.ones(3,4),64)
