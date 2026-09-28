"""Extend only the existing in-block electronic residual width to 512.

The input still splits from the same token tensor into six optical stages and
the original electronic residuals; no third image path or late fusion exists.
All phase tensors stay frozen for this short clean/noisy holdout trial.
"""
import abo_internal_electronic_clean_anchor as base
from LightGenV2.tasks.t07_abo_image_retrieval.standalone import retrieval_refine as refine


profile = dict(base.profile)
profile['electronic_expansion'] = dict(kernels=dict(vision=5, language=5), mlp_width=512)
refine.PROFILES['internal_electronic512_clean_anchor'] = profile

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import main


if __name__ == '__main__':
    main()
