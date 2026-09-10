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
