"""Recover clean ranking after lowering the existing in-block optical alpha.

The two original modalities and six optical stages are retained; no image
bypass or descriptor-level fusion is added. All twelve phase tensors stay fixed.
"""
import math

import abo_internal_electronic_clean_anchor as base
from LightGenV2.tasks.t07_abo_image_retrieval.standalone import robust_training as robust


_original_prepare = base.prepare


def prepare(payload, profile):
    result = _original_prepare(payload, profile)
    low = float(result['metadata']['fusion_alpha_min'])
    high = float(result['metadata']['fusion_alpha_max'])
    target = .37
    if not low < target < high:
        raise ValueError(f'Alpha target {target} outside bounds ({low}, {high})')
    logit = math.log((target - low) / (high - target))
    count = 0
    for name, tensor in result['state_dict'].items():
        if name.endswith(('block1_optical_fusion_logit', 'block2_optical_fusion_logit')):
            tensor.fill_(logit)
            count += 1
    if count != 4:
        raise RuntimeError(f'Expected four existing alpha logits, found {count}')
    result['metadata']['internal_electronic_recovery']['variant'] = 'alpha37_clean_recovery'
    result['metadata']['internal_electronic_recovery']['initial_alpha'] = target
    return result


robust.prepare = prepare

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import main


if __name__ == '__main__':
    main()
