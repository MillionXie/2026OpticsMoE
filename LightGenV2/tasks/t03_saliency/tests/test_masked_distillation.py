from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from LightGenV2.tasks.t03_saliency.masked_distillation import MaskedTeacherRecovery, spatial_patterns, configure
from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label
from LightGenV2.tasks.t03_saliency.sam_training import train_sam_epoch
from LightGenV2.tasks.t03_saliency.training import _checkpoint
from LightGenV2.tasks.t03_saliency.training_support import ModelEMA
from LightGenV2.tasks.t03_saliency.tests.test_relational_distillation import cache_setup, Toy
from experiments.qwen3_vl_embedding_2b_salicon_vision_optical_saliency import training as legacy


def setup(tmp_path):
    s, records, values = cache_setup(tmp_path)
    s.masked_distillation = dict(s.relational_distillation, learning_rate=.0003,
                                 mask_probability=.5, generator_warmup_epochs=3)
    return s, records, values


def test_cache_contract_and_rng_isolation(tmp_path):
    s, records, _ = setup(tmp_path)
    rng = torch.get_rng_state().clone()
    aux = MaskedTeacherRecovery(s, records)
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in aux.parameters()) == 331776
    assert all(k.startswith('generator.') for k in aux.state_dict())
    assert aux.provenance['inference_parameters_added'] == 0
    with pytest.raises(ValueError): MaskedTeacherRecovery(s, list(reversed(records)))
    s.masked_distillation['cache_sha256'] = '0'*64
    with pytest.raises(ValueError, match='SHA256'): MaskedTeacherRecovery(s, records)


def test_generator_warmup_then_student_gradient_and_fp32(tmp_path):
    s, records, values = setup(tmp_path)
    aux = MaskedTeacherRecovery(s, records)
    ids = [r.sample_id for r in records]
    x = values.clone().requires_grad_()
    aux.loss(x, ids, detach_student=True).backward()
    assert x.grad is None and all(p.grad is not None for p in aux.parameters())
    aux.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = aux.loss(x, ids)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.equal(spatial_patterns(torch.ones_like(x)), torch.zeros_like(x))
    with pytest.raises(KeyError): aux.loss(x, ['test/a', 'train/b'])
    with pytest.raises(ValueError): aux.loss(x[..., :-1], ids)


def test_config_preserves_original_inference_and_rejects_combination():
    root = Path(__file__).resolve().parents[1]/'configs'
    old = load_settings(root/'moe_alpha40_extra_control.yaml')
    new = load_settings(root/'moe_alpha40_masked_kd.yaml')
    assert architecture_label(old) == architecture_label(new)
    for k in ('initialization_checkpoint_sha256', 'sam_rho', 'student_epochs', 'phase_learning_rate',
              'router_backend', 'top_k', 'fusion_alpha_min', 'ema_decay', 'ccd_normalization',
              'student_learning_rate', 'dense_head_learning_rate', 'distillation_initial_weight'):
        assert getattr(old, k) == getattr(new, k)
    assert new.initialization_checkpoint_sha256.startswith('87ad4db5')
    for k, v in [('augmentation_enabled', True), ('sam_rho', 0), ('relational_distillation', {'x': 1}),
                 ('feature_pretraining', {'enabled': True}), ('reset_fusion_on_warmstart', True)]:
        bad = deepcopy(new); setattr(bad, k, v)
        with pytest.raises(ValueError): configure(bad, new.masked_distillation, root)
    for k, v in [('mask_probability', 1.), ('learning_rate', float('nan')), ('generator_warmup_epochs', True)]:
        opts = dict(new.masked_distillation); opts[k] = v
        with pytest.raises(ValueError): configure(deepcopy(new), opts, root)


def test_sam_mask_replay_aux_update_and_checkpoint_separation(tmp_path, monkeypatch):
    s, records, _ = setup(tmp_path)
    for k, v in dict(sam_rho=.05, gradient_clip_norm=1., masked_current_weight=1., masked_generator_warmup=False,
        kl_weight=1., cc_weight=.5, sim_weight=.25, nss_weight=.1, map_kd_weight=0., map_kd_temperature=1.,
        distillation_loss='spatial_cc', router_balance_weight=0., router_importance_weight=0.,
        phase_dc_weight=0., log_interval_batches=100).items(): setattr(s,k,v)
    aux = MaskedTeacherRecovery(s, records); model = Toy(); before = deepcopy(aux.state_dict())
    keys = set(model.state_dict()); masks = []
    hook = aux.generator.register_forward_pre_hook(lambda m,a: masks.append((a[0].abs().sum(1)==0).clone()))
    opt = torch.optim.AdamW([{'params':model.core.parameters(),'name':'electronic'},
        {'params':model.head.parameters(),'name':'saliency_head'},
        {'params':aux.parameters(),'name':'training_mgd'}],lr=.001)
    ema=ModelEMA(model,.9); steps=[]
    eh=opt.register_step_post_hook(lambda *a:(steps.append(1),ema.update(*a)))
    monkeypatch.setattr(legacy,'_autocast',lambda *a:nullcontext())
    monkeypatch.setattr(legacy,'preprocess_vision',lambda p,images,d:dict(pixel_values=images,image_grid_thw=None))
    batch=dict(images=torch.randn(2,3,14,14), sample_ids=[r.sample_id for r in records],
               density=torch.rand(2,1,14,14), fixation=torch.rand(2,1,14,14)>.9)
    result=train_sam_epoch(model,[batch],SimpleNamespace(device=torch.device('cpu'),processor=None),
                           s,opt,masked_targets=aux)
    hook.remove(); eh.remove()
    assert len(steps)==1 and len(masks)==2 and torch.equal(*masks)
    assert result['masked_loss']>0 and result['samples']==2
    assert set(model.state_dict())==keys
    assert any(not torch.equal(v,aux.state_dict()[k]) for k,v in before.items())
    model.core.hybrid=SimpleNamespace(fusion_alpha_min=.4,fusion_alpha_max=.95)
    model.checkpoint_architecture='toy'
    _checkpoint(tmp_path/'last.pt',model,1,{},None,training_only_mgd=aux.state_dict())
    _checkpoint(tmp_path/'best.pt',model,1,{}, {'cc':.1})
    last=torch.load(tmp_path/'last.pt',weights_only=False); best=torch.load(tmp_path/'best.pt',weights_only=False)
    assert 'training_only_mgd' in last and 'training_only_mgd' not in best
    assert not any('generator' in k for k in best['core'])


def test_p25_is_config_only_matched_ablation():
    from experiments.qwen3_vl_embedding_2b_caltech101_robust_hybrid_retrieval.settings import _read_config
    root = Path(__file__).resolve().parents[1]/'configs'
    original = _read_config(root/'moe_alpha40_masked_kd.yaml')
    candidate = _read_config(root/'moe_alpha40_masked_kd_p25.yaml')
    assert candidate['masked_distillation']['mask_probability'] == .25
    assert candidate['output_dir'] != original['output_dir']
    candidate['masked_distillation']['mask_probability'] = .5
    candidate['output_dir'] = original['output_dir']
    assert candidate == original
    a = load_settings(root/'moe_alpha40_masked_kd.yaml')
    b = load_settings(root/'moe_alpha40_masked_kd_p25.yaml')
    assert b.masked_distillation['mask_probability'] == .25
    assert architecture_label(a) == architecture_label(b)
    for key in a.masked_distillation:
        if key != 'mask_probability':
            assert a.masked_distillation[key] == b.masked_distillation[key]
