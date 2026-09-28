"""TRAIN-holdout continuation of the single six-stage optical/electronic model.

Keep all twelve optical phase tensors frozen; train the existing electronic
residuals, alpha logits and readout against moderate optical/CCD perturbations.
This does not create an image bypass or a post-hoc descriptor-fusion branch.
"""
from LightGenV2.tasks.t07_abo_image_retrieval.standalone import (
    retrieval_refine as refine,
    robust_training as robust,
)


profile = dict(refine.PROFILES['physical_bounded30_mlp448_paired'])
profile.pop('electronic_expansion')  # The pinned starting checkpoint already has MLP448.
profile.update(
    holdout_selection=True,
    teacher_weight=0.,
    paired_consistency=.5,
    router_consistency=0.,
    noise_probability=.35,
    pixel_shift=1,
    phase_dropout=.015,
    phase_dropout_noisy_only=True,
    sam_rho=0.,
    mild_augmentation=True,
    alpha_lr_multiplier=10.,
    head_lr_multiplier=1.,
    weight_decay=.03,
)
refine.PROFILES['internal_electronic_recovery'] = profile

original_prepare = robust.prepare
original_attach = robust.attach


def prepare(payload, selected_profile):
    result = original_prepare(payload, selected_profile)
    noise = result['metadata']['optical_training_noise']
    noise.update(ccd_mean=.025, ccd_std=.05, ccd_low=-.08, ccd_high=.18)
    result['metadata']['internal_electronic_recovery'] = {
        'optical_phase_scope': 'frozen',
        'ccd_noise': 'uncalibrated training proxy, not electrons',
        'extra_inference_branch': False,
    }
    return result


def attach(model, selected_profile):
    original_attach(model, selected_profile)
    frozen = 0
    for name, parameter in model.named_parameters():
        if refine.optical_parameter(name):
            parameter.requires_grad_(False)
            frozen += 1
    if frozen != 12:
        raise RuntimeError(f'Expected twelve frozen phase tensors, found {frozen}')


robust.prepare = prepare
robust.attach = attach

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import main


if __name__ == '__main__':
    main()
