"""Large-vocabulary supervised pretraining for the frozen P11 optical backbone.

This package is deliberately independent from the running ImageNet-1K
continuation trainer.  It only reads the immutable eight-stage backbone asset
and constructs a new task readout for the declared ImageNet-21K/22K taxonomy.
"""

from .dataset import ClassFolderMMapDataset, GlobalAffineDistributedSampler
from .initialization import initialize_from_frozen_p11_backbone

__all__ = [
    "ClassFolderMMapDataset",
    "GlobalAffineDistributedSampler",
    "initialize_from_frozen_p11_backbone",
]
