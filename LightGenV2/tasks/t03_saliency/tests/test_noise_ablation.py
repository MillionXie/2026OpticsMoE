"""The DC-only trial changes stochastic training corruption, not inference."""
from pathlib import Path

from LightGenV2.tasks.t03_saliency.settings import load_settings
from LightGenV2.tasks.t03_saliency.modeling import architecture_label

TASK = Path(__file__).resolve().parents[1]


def test_dc_only_profile_preserves_leakage_geometry_budget_and_inference():
    base = load_settings(TASK/'configs/moe_alpha40_extra_control.yaml')
    trial = load_settings(TASK/'configs/moe_alpha40_dc_only.yaml')
    assert architecture_label(base) == architecture_label(trial)
    assert trial.student_epochs == base.student_epochs == 40
    assert trial.initialization_checkpoint_sha256 == base.initialization_checkpoint_sha256 == '87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a'
    assert trial.unlabeled_weight == 0 and trial.teacher_only_epochs == 0
    assert trial.fusion_alpha_min == .4 and trial.router_backend == 'optical'
    assert trial.top_k == 2 and trial.active_size == 478 and trial.expert_size == 224
    assert trial.pixel_pitch_um == 17 and trial.global_to_detector_distance_m == .1
    assert trial.language_optical_zero_order_enabled
    assert trial.language_optical_zero_order_random_relative_phase
    for kind in ('amplitude','phase'):
        assert getattr(trial,f'language_optical_{kind}_zero_order_intensity_min') == .2
        assert getattr(trial,f'language_optical_{kind}_zero_order_intensity_max') == .3
    for key in ('router_balance_weight','router_importance_weight','phase_dc_weight',
                'kl_weight','cc_weight','sim_weight','nss_weight','sam_rho','ema_decay',
                'student_learning_rate','phase_learning_rate','router_learning_rate',
                'dense_readout_learning_rate','dense_head_learning_rate',
                'ffn_spatial_learning_rate','staged_polish_start','student_batch_size',
                'distillation_initial_weight','distillation_final_weight'):
        assert getattr(trial,key) == getattr(base,key), key
    assert trial.language_optical_phase_dropout_p == 0
    assert trial.optical_router_phase_dropout_p == 0 and trial.router_noise_std == 0
    assert trial.language_optical_gain_min == trial.language_optical_gain_max == 1
    for key in ('mean','std','min','max'):
        assert getattr(trial,f'language_optical_ccd_noise_{key}_fraction') == 0
    assert trial.language_optical_max_shift_pixels == 0
    for key in ('phase','ccd'):
        assert getattr(trial,f'language_optical_{key}_shift_pixels') == 0
    for key in ('input','phase','ccd'):
        assert getattr(trial,f'optical_router_{key}_shift_pixels') == 0
