"""Widen the existing electronic context kernels without adding a branch.

Starting from the pinned 5x5/5 electronic residual, zero padding preserves
the initial function while allowing the vision context to learn a wider view.
The six optical phase tensors remain frozen in this TRAIN-only trial.
"""
import abo_internal_electronic_clean_anchor as base
from LightGenV2.tasks.t07_abo_image_retrieval.standalone import retrieval_refine as refine


profile = dict(base.profile)
profile['electronic_expansion'] = dict(kernels=dict(vision=13, language=7), mlp_width=448)
refine.PROFILES['internal_conv13_clean_anchor'] = profile

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import main


if __name__ == '__main__':
    main()
