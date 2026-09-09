from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t03_saliency.lightweight_residual import TokenGlobalResponseNorm, configure_grn, initialize_identity_grn
from LightGenV2.tasks.t03_saliency.modeling import configure_spatial_kernel, initialize_student, architecture_label
from LightGenV2.tasks.t03_saliency.settings import load_settings
from experiments.qwen3_vl_embedding_2b_caltech101_electronic_retrieval.electronic_blocks import ElectronicResidualMLPBlock
from experiments.qwen3_vl_embedding_2b_caltech101_balanced_optical_fusion_ablation.modeling import _ScaleMatchedFusionMixin

TASK = Path(__file__).resolve().parents[1]


def test_grn_identity_gradients_and_no_batch_interaction():
    module = TokenGlobalResponseNorm(384)
    x = torch.randn(2,196,384,requires_grad=True)
    torch.testing.assert_close(module(x), x, rtol=0, atol=0)
    module(x).square().mean().backward()
    assert module.gamma.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    module.gamma.data.fill_(.3)
    module.beta.data.fill_(.1)
    torch.testing.assert_close(module(x[:1]), module(x)[:1])
    perm = torch.randperm(196)
    torch.testing.assert_close(module(x[:,perm]), module(x)[:,perm])
    with pytest.raises(ValueError):
        module(torch.randn(1,195,384))


def test_grn_zero_backward_finite_and_equation():
    module = TokenGlobalResponseNorm(4)
    module.gamma.data.fill_(.2)
    x = torch.zeros(1,196,4,requires_grad=True)
    module(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    x = torch.randn(1,196,4)
    grid = x.reshape(1,14,14,4)
    response = torch.norm(grid, p=2, dim=(1,2), keepdim=True)
    expected = grid + .2 * grid * response / (response.mean(-1,keepdim=True) + 1e-6)
    torch.testing.assert_close(module(x), expected.reshape_as(x))


class TinyFusion(_ScaleMatchedFusionMixin, torch.nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.block1_optical_fusion_logit = torch.nn.Parameter(torch.zeros(()))
        self.block2_optical_fusion_logit = torch.nn.Parameter(torch.zeros(()))
        self._configure_balanced_fusion(settings)
        self.blocks = torch.nn.ModuleList([ElectronicResidualMLPBlock(192,2,0,.1,
            token_mixer_enabled=True, token_mixer_kernel_size=3, token_mixer_type="depthwise_conv2d") for _ in range(2)])
        configure_spatial_kernel(self, settings.electronic_spatial_kernel_size)
        if settings.electronic_grn:
            configure_grn(self)


@pytest.mark.parametrize('variant,delta', [('control',0),('flip',0),('kernel5',6144),('grn',1536),('kernel5_grn',7680)])
def test_early_transfer_all_variants_preserve_residual_function(tmp_path, variant, delta):
    old = load_settings(TASK/'configs/moe_dc20_mean_only_continue.yaml')
    core = torch.nn.Module(); core.hybrid = TinyFusion(old)
    head = torch.nn.Linear(192,1)
    path = tmp_path/'early.pt'
    torch.save({'architecture': architecture_label(old), 'epoch': 75,
                'core':core.state_dict(),'saliency_head':head.state_dict()}, path)
    s = load_settings(TASK/f'configs/moe_alpha40_early_{variant}.yaml')
    s.initialization_checkpoint = path
    s.initialization_checkpoint_sha256 = None
    target = torch.nn.Module(); target.hybrid = TinyFusion(s)
    model = SimpleNamespace(core=target,head=torch.nn.Linear(192,1),checkpoint_architecture=architecture_label(s))
    report = initialize_student(model,s)
    assert report['identity_grn_added'] == ('grn' in variant)
    assert sum(p.numel() for p in target.parameters()) - sum(p.numel() for p in core.parameters()) == delta
    assert float(target.hybrid.block1_optical_fusion) == pytest.approx(.45)
    x = torch.randn(2,196,192)
    kwargs = dict(padding_mask=torch.zeros(2,196,dtype=torch.bool),causal=False,spatial_shapes=[(1,14,14)]*2)
    for i in range(2):
        torch.testing.assert_close(target.hybrid.blocks[i](x,**kwargs),core.hybrid.blocks[i](x,**kwargs),atol=1e-6,rtol=1e-5)


def test_grn_transfer_rejects_unrelated_missing_keys():
    source = {'old':torch.ones(1)}
    with pytest.raises(RuntimeError):
        initialize_identity_grn(source, {'old':torch.ones(1),'unrelated':torch.zeros(1)})


def test_early_profiles_share_training_and_hardware_contract():
    configs = [load_settings(TASK/f'configs/moe_alpha40_early_{v}.yaml') for v in ('control','flip','kernel5','grn','kernel5_grn')]
    for s in configs:
        assert s.initialization_checkpoint == configs[0].initialization_checkpoint
        assert s.student_epochs == 100 and s.staged_warmup_epochs == 5
        assert s.fusion_alpha_min == .4 and s.fusion_alpha_initial == .45
        assert s.router_backend == 'optical' and s.top_k == 2
        assert s.active_size == 478 and s.expert_size == 224
        assert s.language_optical_phase_zero_order_intensity_min == .2
        assert s.language_optical_phase_zero_order_intensity_max == .3
        assert s.distillation_initial_weight == .6 and s.distillation_final_weight == .2
        assert s.augmentation_mode == 'aligned_flip' and s.crop_scale_min == 1
    assert len({architecture_label(s) for s in configs}) == 4
