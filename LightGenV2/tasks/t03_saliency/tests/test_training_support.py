from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from LightGenV2.tasks.t03_saliency.training_support import ModelEMA, TrainTeacherMaps, distillation_weight
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label


def test_resolved_config_replaces_inherited_pose_selection_text(tmp_path, monkeypatch):
    import yaml
    from LightGenV2.tasks.t03_saliency import settings as module
    s=load_settings(Path(__file__).resolve().parents[1]/'configs/moe_alpha40_viewreg_sam_spatialcc.yaml')
    s.output_dir=tmp_path
    def inherited_writer(settings):
        (settings.output_dir/'resolved_config.yaml').write_text(yaml.safe_dump({
            'lightgen':{},'protocol':{'checkpoint_selection':'max PCK@0.2; then torso NME',
                                     'initialization_seed':42}}),encoding='utf-8')
    monkeypatch.setattr(module,'save_t02_resolved_config',inherited_writer)
    before=(s.student_epochs,s.test_interval_epochs,s.initialization_checkpoint_sha256)
    module.save_resolved_config(s)
    data=yaml.safe_load((tmp_path/'resolved_config.yaml').read_text(encoding='utf-8'))
    assert data['protocol']['primary_metric']=='CC'
    assert data['protocol']['checkpoint_selection']=='maximum public-test CC; ties retain the earlier selected epoch'
    assert data['protocol']['initialization_seed']==42
    assert data['protocol']['test_used_for_checkpoint_selection'] is True
    assert before==(s.student_epochs,s.test_interval_epochs,s.initialization_checkpoint_sha256)


def test_ema_step_hook_and_restoration():
    model = SimpleNamespace(core=torch.nn.Linear(1, 1, bias=False), head=torch.nn.Linear(1, 1, bias=False))
    model.core.weight.data.zero_()
    ema = ModelEMA(model, .5)
    opt = torch.optim.SGD(model.core.parameters(), lr=1)
    hook = opt.register_step_post_hook(ema.update)
    model.core.weight.grad = torch.full_like(model.core.weight, -2.)
    opt.step()
    with pytest.raises(RuntimeError):
        with ema.applied():
            assert model.core.weight.item() == 1
            raise RuntimeError('test restoration')
    assert model.core.weight.item() == 2
    hook.remove()


def test_kd_tapers_to_zero():
    assert distillation_weight(.6, 50, 1) == .6
    assert distillation_weight(.6, 50, 50) == 0
    assert distillation_weight(.6, 50, 100) == 0


def test_cache_alignment_and_identity(tmp_path):
    path = tmp_path / 'cache.pt'
    torch.save({'sample_ids': ['train/1'], 'logits': torch.ones(1,1,2,2),
                'manifest': {'checkpoint_sha256': 'abc', 'image_size': 2, 'augmentation': False}}, path)
    s = SimpleNamespace(augmentation_enabled=False, distillation_cache=path,
                        distillation_teacher_sha256='abc', image_size=2)
    cache = TrainTeacherMaps(s, [SimpleNamespace(sample_id='train/1')])
    assert cache.get(['train/1'], 'cpu').shape == (1,1,2,2)
    with pytest.raises(ValueError):
        TrainTeacherMaps(s, [SimpleNamespace(sample_id='validation/1')])
    s.augmentation_enabled = True
    with pytest.raises(ValueError):
        TrainTeacherMaps(s, [SimpleNamespace(sample_id='train/1')])


def test_generalization_keeps_optical_contract():
    root = Path(__file__).resolve().parents[1] / 'configs'
    original = load_settings(root / 'moe_alpha40_refine_weakaug.yaml')
    for name in ('regularized', 'aligned', 'kd020', 'kd060'):
        s = load_settings(root / f'moe_alpha40_generalize_{name}.yaml')
        assert architecture_label(s) == architecture_label(original)
        assert s.ema_decay == .995 and s.phase_weight_decay == 0
        assert s.weight_decay == .01
        assert s.fusion_alpha_min == .4 and s.top_k == 2
        assert s.language_optical_zero_order_enabled
        assert s.language_optical_phase_zero_order_intensity_min == .2
        assert s.language_optical_phase_zero_order_intensity_max == .3
