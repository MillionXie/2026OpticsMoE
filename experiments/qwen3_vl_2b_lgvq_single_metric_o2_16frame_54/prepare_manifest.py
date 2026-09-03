"""Path-keyed LGVQ manifest preparation shared with the audited LGVQ split.

The underlying manifest deliberately retains both MOS columns.  A spatial run
and a temporal run then select exactly one column in :mod:`data`; this lets the
large target-neutral Vision cache be shared without weakening target/prompt
binding checks.
"""

from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.prepare_manifest import (  # noqa: F401
    _flatten_quality_records,
    _normalized_video_key,
    _read_mos_by_path,
    main,
    prepare_manifest,
)

__all__ = [
    "_flatten_quality_records",
    "_normalized_video_key",
    "_read_mos_by_path",
    "main",
    "prepare_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
