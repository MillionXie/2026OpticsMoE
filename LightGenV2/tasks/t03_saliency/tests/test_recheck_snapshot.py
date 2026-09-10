import hashlib

import torch
import pytest
from types import SimpleNamespace

from LightGenV2.tasks.t03_saliency.recheck_aligned import load_hashed_checkpoint
from LightGenV2.tasks.t03_saliency.recheck_aligned import apply_inference_ablation


def test_digest_binds_loaded_bytes_not_later_best(tmp_path, monkeypatch):
    path = tmp_path / 'best_checkpoint.pt'
    torch.save({'epoch': 5, 'value': torch.tensor([.25])}, path)
    original = path.read_bytes()
    original_load = torch.load

    def load_then_replace(stream, **kwargs):
        payload = original_load(stream, **kwargs)
        torch.save({'epoch': 10, 'value': torch.tensor([.75])}, path)
        return payload

    monkeypatch.setattr(torch, 'load', load_then_replace)
    payload, digest = load_hashed_checkpoint(path)
    assert payload['epoch'] == 5
    torch.testing.assert_close(payload['value'], torch.tensor([.25]))
    assert digest == hashlib.sha256(original).hexdigest()
    assert digest != hashlib.sha256(path.read_bytes()).hexdigest()


def test_ablation_only_sets_runtime_mode_and_can_restore_normal():
    class Hybrid(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor([.4,.6]))
            self.mode='none'
        def set_fusion_ablation(self,mode):
            self.mode=mode
    hybrid=Hybrid()
    model=SimpleNamespace(core=SimpleNamespace(hybrid=hybrid))
    before={k:v.clone() for k,v in hybrid.state_dict().items()}
    for mode in ['remove_optical','none']:
        apply_inference_ablation(model,'optical',mode)
        assert hybrid.mode==mode
        assert hybrid.state_dict().keys()==before.keys()
        for k,v in hybrid.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)


def test_qwen_ablation_rejected_and_normal_needs_no_optical_core():
    apply_inference_ablation(object(),'qwen','none')
    with pytest.raises(ValueError,match='no optical branch'):
        apply_inference_ablation(object(),'qwen','remove_optical')


@pytest.mark.parametrize('system,mode',[('unknown','none'),('optical','remove_electronic')])
def test_ablation_rejects_unsupported_modes(system,mode):
    with pytest.raises(ValueError,match='Unsupported'):
        apply_inference_ablation(object(),system,mode)


def _stub_saliency_forward(mode):
    from LightGenV2.tasks.t03_saliency.modeling import LightGenVision2SaliencyStudent
    model=LightGenVision2SaliencyStudent.__new__(LightGenVision2SaliencyStudent)
    torch.nn.Module.__init__(model)
    model.core=torch.nn.Module()
    model.core.hybrid=SimpleNamespace(fusion_ablation_mode=mode)
    model.core.optical_branch=SimpleNamespace(core=SimpleNamespace(current_detector_readout=None))
    class Visual(torch.nn.Module):
        def __init__(self,core):
            super().__init__();self.patch_embed=torch.nn.Linear(1,1);self.core=core
        def forward(self,pixels,grid_thw):
            self.core.last_latent_groups=[torch.full((196,3),float(i+1)) for i in range(len(grid_thw))]
    model.visual=Visual(model.core)
    model.capture_block=SimpleNamespace(set_grid=lambda grid:None)
    model.head=torch.nn.Conv2d(3,1,1,bias=False)
    model._active=True
    return model


def test_remove_optical_fresh_forward_needs_no_detector_and_never_reuses_stale_one():
    model=_stub_saliency_forward('remove_optical')
    grid=torch.tensor([[1,14,14],[1,14,14]])
    first=model(torch.zeros(1),grid)
    assert first[2] is None and first[0].shape==(2,1,14,14)
    model.core.optical_branch.core.current_detector_readout=torch.full((19,1,1),float('nan'))
    again=model(torch.zeros(1),grid)
    assert again[2] is None
    assert model.core.optical_branch.core.current_detector_readout is None
    torch.testing.assert_close(first[0],again[0],atol=0,rtol=0)
    assert model(torch.zeros(1),grid[:1])[0].shape==(1,1,14,14)


def test_normal_saliency_forward_delegates_unchanged(monkeypatch):
    from experiments.qwen3_vl_embedding_2b_lsp_pose_optical_router.modeling import RobustVision2PoseStudent
    model=_stub_saliency_forward('none')
    sentinel=object()
    monkeypatch.setattr(RobustVision2PoseStudent,'forward',lambda self,pixels,grid:sentinel)
    assert model(torch.zeros(1),torch.tensor([[1,14,14]])) is sentinel
