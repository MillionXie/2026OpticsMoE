from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from LightGenV2.tasks.t03_saliency import semisupervised as semi
from LightGenV2.tasks.t03_saliency.settings import load_settings

TASK = Path(__file__).resolve().parents[1]


def test_mixed_objective_preserves_gt_gradient_and_averages_regularizers():
    values = [torch.tensor(float(i+1),requires_grad=True) for i in range(5)]
    actual = semi.mixed_objective(*values,weight=.6)
    gradients = torch.autograd.grad(actual,values)
    assert [float(x) for x in gradients] == pytest.approx([1.,.6,.5,.5,1.])
    for weight in (0.,-1.,3.,float('nan')):
        with pytest.raises(ValueError): semi.mixed_objective(*values,weight=weight)


def test_stream_retains_cursor_across_epochs_and_closes():
    stream = semi.CyclingUnlabeledBatches([{'id':1},{'id':2}],None)
    assert stream.next_batch()['id'] == 1
    assert stream.next_batch()['id'] == 2
    assert stream.next_batch()['id'] == 1 and stream.restarts == 1
    stream.close()
    assert stream.iterator is None
    with pytest.raises(RuntimeError): stream.next_batch()
    with pytest.raises(ValueError): semi.CyclingUnlabeledBatches([],None)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.e = torch.nn.Parameter(torch.tensor([.2,.4,.1,.3]).reshape(1,1,2,2))
        self.phase = torch.nn.Parameter(torch.tensor([.3,.1,.4,.2]).reshape(1,1,2,2))
        self.calls = []
        self.regularizer_states = []

    def forward(self,pixels,grid):
        self.last = pixels*self.e + self.phase
        self.calls.append(pixels.detach().clone())
        return self.last,None,None

    def router_losses(self):
        self.regularizer_states.append(self.last.detach().clone())
        return self.last.square().mean(),self.last.mean().square()

    def operating_loss(self):
        return self.last.abs().mean()


class Teacher:
    def __init__(self): self.calls = []
    def get(self,ids,device):
        self.calls.append(list(ids))
        return torch.tensor([.1,.3,.7,.2],device=device).reshape(1,1,2,2).repeat(len(ids),1,1,1)


def test_real_sam_epoch_uses_four_forwards_one_data_draw_and_one_update(monkeypatch):
    monkeypatch.setattr(semi.legacy,'_autocast',lambda *_:nullcontext())
    monkeypatch.setattr(semi.legacy,'preprocess_vision',
                        lambda processor,images,device:{'pixel_values':torch.stack(images).to(device),'image_grid_thw':None})
    monkeypatch.setattr(semi.legacy,'phase_dc_loss',lambda model:model.phase.square().mean())
    s = SimpleNamespace(sam_rho=.05,teacher_only_epochs=0,augmentation_enabled=False,
        kl_weight=1.,cc_weight=1.5,sim_weight=.25,nss_weight=.1,map_kd_weight=2.,map_kd_temperature=1.,
        distillation_loss='spatial_cc',unlabeled_weight=.6,router_balance_weight=.1,
        router_importance_weight=.1,ccd_operating_point_weight=.1,phase_dc_weight=.01,
        gradient_clip_norm=1.,log_interval_batches=100)
    model = Toy()
    optim = torch.optim.AdamW([{'params':[model.e],'name':'electronic'},
                              {'params':[model.phase],'name':'optical'}],lr=.01,weight_decay=0.)
    updates=[]
    hook=optim.register_step_post_hook(lambda *_:updates.append(1))
    before = model.phase.detach().clone()
    labeled = {'images':[torch.tensor([1.,2.,3.,4.]).reshape(1,2,2)],'sample_ids':['train/1'],
               'density':torch.tensor([.1,.4,.3,.2]).reshape(1,1,2,2),
               'fixation':torch.tensor([0.,1.,0.,0.]).reshape(1,1,2,2)}
    extra = {'images':[torch.tensor([4.,3.,2.,1.]).reshape(1,2,2)],
             'sample_ids':['unlabeled/coco2017/1']}
    ut,lt=Teacher(),Teacher()
    stream=semi.CyclingUnlabeledBatches([extra],ut)
    metrics=semi.train_semisupervised_epoch(model,[labeled],SimpleNamespace(device=torch.device('cpu'),processor=None),s,optim,lt,stream)
    hook.remove()
    assert len(updates)==1 and len(model.calls)==4
    assert len(ut.calls)==1 and len(lt.calls)==1 and stream.restarts==0
    assert torch.equal(model.calls[0],model.calls[2]) and torch.equal(model.calls[1],model.calls[3])
    assert len(model.regularizer_states)==4
    assert not torch.equal(model.regularizer_states[0],model.regularizer_states[1])
    assert not torch.equal(before,model.phase)
    assert torch.isfinite(model.phase).all() and torch.isfinite(model.e).all()
    assert metrics['samples']==1 and metrics['unlabeled_samples']==1
    assert 'unlabeled_map_kd' in metrics and 'unlabeled_nss' not in metrics


def config(tmp_path,override=''):
    path=tmp_path/'LightGenV2/tasks/t03_saliency/configs/semi.yaml'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(f'base_config: {(TASK/"configs/moe_alpha40_sam_spatialcc_kd2.yaml").as_posix()}\n'
                    'unlabeled_distillation:\n  weight: 0.6\n  image_manifest: images.json\n'
                    f'  image_manifest_sha256: "{"a"*64}"\n  cache_file: maps.pt\n  cache_sha256: "{"b"*64}"\n'+override)
    return path


def test_settings_are_opt_in_and_do_not_change_inference(tmp_path):
    from LightGenV2.tasks.t03_saliency.modeling import architecture_label
    base=load_settings(TASK/'configs/moe_alpha40_sam_spatialcc_kd2.yaml')
    s=load_settings(config(tmp_path))
    assert base.unlabeled_weight==0 and s.unlabeled_weight==.6
    assert architecture_label(base)==architecture_label(s)
    assert s.kl_weight==base.kl_weight and s.cc_weight==base.cc_weight
    assert s.fusion_alpha_min==.4 and s.router_backend=='optical' and s.top_k==2


def test_published_extra_profile_matches_control_and_pins_real_cache():
    from LightGenV2.tasks.t03_saliency.modeling import architecture_label
    control=load_settings(TASK/'configs/moe_alpha40_extra_control.yaml')
    extra=load_settings(TASK/'configs/moe_alpha40_extra_coco20k.yaml')
    assert architecture_label(control)==architecture_label(extra)
    assert control.student_epochs==extra.student_epochs==40
    assert control.initialization_checkpoint_sha256==extra.initialization_checkpoint_sha256
    assert extra.unlabeled_weight==.6 and control.unlabeled_weight==0
    assert extra.unlabeled_image_manifest == (TASK.parents[2]/'cache/qwen3_vl_embedding_2b_salicon_lightgen/coco20k_pretrain_20260910/image_manifest.json').resolve()
    assert extra.unlabeled_cache_sha256=='232e02d243a3d58b8d5cc48557da8f566f77a81e7020968a8a84c0c85d7c305c'
    assert extra.fusion_alpha_min==.4 and extra.top_k==2 and extra.router_backend=='optical'


def test_earlier_pair_keeps_inference_and_matches_training_budget():
    from LightGenV2.tasks.t03_saliency.modeling import architecture_label
    late=load_settings(TASK/'configs/moe_alpha40_extra_coco20k.yaml')
    control=load_settings(TASK/'configs/moe_alpha40_extra_early_control.yaml')
    extra=load_settings(TASK/'configs/moe_alpha40_extra_early_coco20k.yaml')
    assert architecture_label(control)==architecture_label(extra)==architecture_label(late)
    assert control.student_epochs==extra.student_epochs==80
    assert control.unlabeled_weight==0 and extra.unlabeled_weight==.6
    assert control.initialization_checkpoint_sha256==extra.initialization_checkpoint_sha256=='de477b8c13c46c50cb9f17eb0b62bc577aaefef5512887e1e17a86d0affb5eea'
    assert extra.initialize_ffn_on_warmstart and control.initialize_ffn_on_warmstart
    for name in ('student_learning_rate','phase_learning_rate','router_learning_rate',
                 'dense_readout_learning_rate','dense_head_learning_rate','ffn_spatial_learning_rate',
                 'staged_warmup_epochs','staged_polish_start','weight_decay','sam_rho','ema_decay',
                 'kl_weight','cc_weight','sim_weight','nss_weight','distillation_initial_weight',
                 'distillation_final_weight','distillation_end_epoch','student_batch_size'):
        assert getattr(control,name)==getattr(extra,name)
    for s in (control,extra):
        assert s.fusion_alpha_min==.4 and not s.reset_fusion_on_warmstart
        assert s.router_backend=='optical' and s.top_k==2
        assert s.active_size==478 and s.expert_size==224
        assert s.language_optical_phase_zero_order_intensity_min==.2
        assert s.language_optical_phase_zero_order_intensity_max==.3
        assert not s.augmentation_enabled and s.teacher_only_epochs==0
    assert extra.unlabeled_cache_sha256==late.unlabeled_cache_sha256


@pytest.mark.parametrize('override',[
    'training:\n  sam_rho: 0\n',
    'distillation:\n  teacher_only_epochs: 10\n',
    'augmentation:\n  enabled: true\n',
    'loss:\n  cc_weight: 0\n',
])
def test_incompatible_supervision_is_rejected(tmp_path,override):
    with pytest.raises(ValueError): load_settings(config(tmp_path,override))
