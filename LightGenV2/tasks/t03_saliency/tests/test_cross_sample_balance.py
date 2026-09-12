from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t03_saliency.cross_sample_balance import terms


def sample():
    torch.manual_seed(71)
    p = torch.randn(8, 4, dtype=torch.float64).softmax(-1).requires_grad_()
    h = torch.zeros_like(p).scatter(1, p.topk(2, dim=1).indices, 1)
    return p, h


def test_values_and_gradients_match_explicit_distinct_pairs():
    p, h = sample()
    out = terms(dict(probabilities=p, selected_mask=h))
    hs = h.detach()/2 + p - p.detach()
    def explicit(x, y):
        return 4 * sum((x[i]*y[j]).sum() for i in range(8) for j in range(8) if i != j)/56
    expected = dict(soft_delta=explicit(p,h/2)-4*(p.mean(0)*(h/2).mean(0)).sum(),
        importance_delta=explicit(p,p)-4*p.mean(0).square().sum(), hard=explicit(hs,hs)-1)
    for key in out:
        torch.testing.assert_close(out[key], expected[key])
        torch.testing.assert_close(torch.autograd.grad(out[key],p,retain_graph=True)[0],
                                   torch.autograd.grad(expected[key],p,retain_graph=True)[0])
    perm = torch.randperm(8)
    other = terms(dict(probabilities=p[perm], selected_mask=h[perm]))
    for key in out:
        torch.testing.assert_close(out[key], other[key])


def test_uniform_and_collapse_and_negative_estimates():
    p = torch.full((4,4), .25, requires_grad=True)
    h = torch.tensor([[1.,1,0,0],[0,0,1,1],[1,0,1,0],[0,1,0,1]])
    out = terms(dict(probabilities=p, selected_mask=h))
    assert abs(out['importance_delta'].item()) < 1e-6
    assert out['hard'].item() == pytest.approx(-1/3)
    collapsed = torch.tensor([[1.,1,0,0]]).repeat(4,1)
    assert terms(dict(probabilities=p,selected_mask=collapsed))['hard'].item() == pytest.approx(1.)
    with pytest.raises(ValueError):
        terms(dict(probabilities=p[:1],selected_mask=h[:1]))


def test_model_hook_preserves_capture_and_eval_loss():
    from LightGenV2.tasks.t03_saliency.modeling import LightGenVision2SaliencyStudent as Student
    p,h = sample()
    importance = 4*p.mean(0).square().sum()-1
    soft = 4*(p.mean(0)*(h/2).mean(0)).sum()+.123
    hard = 4*(h/2).mean(0).square().sum()-1
    model = SimpleNamespace(core=SimpleNamespace(last_routing=dict(probabilities=p,selected_mask=h),
        router_losses=lambda: (soft, importance)), router_backend='optical',
        _router_soft_weight=.08,_router_hard_weight=.1,_router_balance_estimator='cross_sample',
        training=True,router_hard_load_balance_loss=lambda:hard)
    a,b = Student.router_losses(model)
    t = terms(model.core.last_routing)
    torch.testing.assert_close(a, soft+t['soft_delta']+1.25*t['hard'])
    torch.testing.assert_close(b, importance+t['importance_delta'])
    model.training=False
    a,b=Student.router_losses(model)
    torch.testing.assert_close(a,soft+1.25*hard)
    torch.testing.assert_close(b,importance)


def test_paired_config_and_serialization(tmp_path):
    import yaml
    from LightGenV2.tasks.t03_saliency.settings import load_settings, save_resolved_config
    from LightGenV2.tasks.t03_saliency.modeling import architecture_label
    root=Path(__file__).resolve().parents[1]/'configs'
    a=load_settings(root/'moe_alpha40_sam_batch8_20260913.yaml')
    b=load_settings(root/'moe_alpha40_sam_batch8_crosssample_20260913.yaml')
    assert architecture_label(a)==architecture_label(b)
    assert a.router_balance_estimator=='batch' and b.router_balance_estimator=='cross_sample'
    for key in ['initialization_checkpoint_sha256','student_epochs','student_batch_size',
                'student_learning_rate','phase_learning_rate','router_learning_rate','ema_decay',
                'router_hard_load_balance_weight','router_balance_weight','router_importance_weight',
                'fusion_alpha_min','top_k','language_optical_phase_zero_order_intensity_min',
                'language_optical_phase_zero_order_intensity_max']:
        assert getattr(a,key)==getattr(b,key)
    b.output_dir=tmp_path
    save_resolved_config(b)
    assert yaml.safe_load((tmp_path/'resolved_config.yaml').read_text())['loss']['router_balance_estimator']=='cross_sample'
