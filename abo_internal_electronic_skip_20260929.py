"""TRAIN-only optical skip to strengthen the existing electronic residual.

At inference the original six-stage optical/electronic architecture is
unchanged. This adds no input bypass, descriptor branch, or post-hoc fusion.
"""
import os
import torch

import abo_internal_electronic_clean_anchor  # installs phase-frozen profile
from LightGenV2.tasks.t07_abo_image_retrieval.standalone.model import Modality


_original_forward = Modality.forward
_skip_probability = float(os.environ.get("ABO_NOISY_OPTICAL_SKIP_PROB", ".5"))
if not 0. <= _skip_probability <= 1.:
    raise ValueError("ABO_NOISY_OPTICAL_SKIP_PROB must be in [0,1]")


def forward_with_train_skip(self, inputs):
    if (self.training and self.optics.noise_enabled and not self.remove_optical
            and torch.rand((), device=inputs.device).item() < _skip_probability):
        self.remove_optical = True
        try:
            return _original_forward(self, inputs)
        finally:
            self.remove_optical = False
    return _original_forward(self, inputs)


Modality.forward = forward_with_train_skip

from LightGenV2.tasks.t07_abo_image_retrieval.standalone.retrieval_adapt import main


if __name__ == "__main__":
    main()
