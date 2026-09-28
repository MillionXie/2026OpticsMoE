"""Clean-preserving continuation of the original six-stage optical/electronic model.

No extra image path or post-hoc descriptor fusion is introduced. Optical phase
tensors remain frozen; this short trial trades less noise for clean retention.
"""
import abo_internal_electronic_recovery as base

from LightGenV2.tasks.t07_abo_image_retrieval.standalone import (
    retrieval_refine as refine,
    robust_training as robust,
)

profile = dict(base.profile)
profile.update(
    paired_consistency=.25,
    noise_probability=.20,
    pixel_shift=1,
    phase_dropout=.005,
    alpha_lr_multiplier=2.,
    weight_decay=.02,
)
refine.PROFILES['internal_electronic_clean_anchor'] = profile


def prepare(payload, selected_profile):
    result = base.prepare(payload, selected_profile)
    noise = result['metadata']['optical_training_noise']
    noise.update(ccd_mean=.015, ccd_std=.035, ccd_low=-.06, ccd_high=.13)
    result['metadata']['internal_electronic_recovery']['variant'] = 'clean_anchor'
    return result


robust.prepare = prepare

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import main


if __name__ == '__main__':
    main()
