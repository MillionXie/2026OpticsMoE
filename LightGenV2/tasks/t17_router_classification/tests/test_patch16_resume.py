from argparse import Namespace
import pytest
from LightGenV2.tasks.t17_router_classification.gpu_policy import validate_gpu_uuids, require_non_a100_gpu
from LightGenV2.tasks.t17_router_classification.refine_patch16 import resume_progress, validate_resume_settings


def test_a100_rejected_even_when_explicit_uuid_and_other_cards_allowed():
    inventory = {'GPU-a': 'NVIDIA A100-PCIE-40GB', 'GPU-b': 'NVIDIA GeForce RTX 3090'}
    with pytest.raises(ValueError, match='prohibits'):
        validate_gpu_uuids(['GPU-b', 'GPU-a'], inventory)
    assert validate_gpu_uuids(['GPU-b'], inventory) == ['GPU-b']
    with pytest.raises(ValueError, match='UUID'):
        validate_gpu_uuids(['6'], inventory)


def test_trainer_cannot_silently_select_any_gpu(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    with pytest.raises(ValueError, match='explicit GPU UUID'):
        require_non_a100_gpu()


def test_resume_preserves_parent_based_early_stop_clock_and_completed_prefix():
    history = [dict(epoch=i+1, validation=dict(accuracy=v, balanced_accuracy=v))
               for i, v in enumerate((.58, .601, .600, .602))]
    best, significant, last = resume_progress(history, 4, .6)
    assert best == .602 and significant == .6 and last == 0
    with pytest.raises(ValueError, match='exactly match'):
        resume_progress(history, 3, .6)
    history.append(dict(epoch=5, validation=dict(accuracy=.607, balanced_accuracy=.607)))
    assert resume_progress(history, 5, .6)[2] == 5


def test_resume_rejects_silent_budget_or_loss_change():
    cfg = dict(init_run='parent',epochs=20,min_epochs=10,patience=10,train_eval_every=5,
               phase_lr=.001,head_lr=.0002,balance_weight=.1,balance_mode='dense_kl',
               router_warmup_epochs=3,router_warmup_lr=.005,batch=32,seed=17)
    validate_resume_settings(Namespace(**cfg), cfg)
    changed = dict(cfg, balance_weight=1.)
    with pytest.raises(ValueError, match='balance_weight'):
        validate_resume_settings(Namespace(**changed), cfg)
