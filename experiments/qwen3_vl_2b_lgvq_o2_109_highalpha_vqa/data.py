"""LGVQ manifest/cache contract shared with the audited baseline project.

The O2-109 experiment changes the feature extractor and optical model, not the
dataset split or cache container.  Re-exporting the audited data layer keeps a
single implementation of the path-keyed 2250/558 split contract.
"""

from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.data import (
    LGVQFeatureDataset,
    ManifestRow,
    build_canonical_cache,
    cache_report,
    canonicalize_feature_tensor,
    file_sha256,
    load_canonical_cache,
    read_manifest,
    split_counts,
)

__all__ = [
    "LGVQFeatureDataset",
    "ManifestRow",
    "build_canonical_cache",
    "cache_report",
    "canonicalize_feature_tensor",
    "file_sha256",
    "load_canonical_cache",
    "read_manifest",
    "split_counts",
]


