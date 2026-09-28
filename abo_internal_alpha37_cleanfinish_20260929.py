"""Short clean recovery of the existing two-path ABO model at low alpha.

No third image branch or late fusion is introduced. Optical phase tensors
remain frozen. This continuation is selected on TRAIN continuation holdout.
"""
import abo_internal_electronic_clean_anchor as base
from LightGenV2.tasks.t07_abo_image_retrieval.standalone import retrieval_refine as refine
from LightGenV2.tasks.t07_abo_image_retrieval.standalone import robust_training as robust


profile = dict(base.profile)
profile.update(noise_probability=.08, phase_dropout=0., paired_consistency=.1,
               alpha_lr_multiplier=1., weight_decay=.01)
refine.PROFILES['internal_alpha37_cleanfinish'] = profile

_original_prepare = base.prepare


def prepare(payload, selected_profile):
    result = _original_prepare(payload, selected_profile)
    result['metadata']['optical_training_noise'].update(
        ccd_mean=.01, ccd_std=.025, ccd_low=-.04, ccd_high=.09)
    result['metadata']['internal_electronic_recovery']['variant'] = 'alpha37_cleanfinish'
    return result


robust.prepare = prepare

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import main


if __name__ == '__main__':
    main()
