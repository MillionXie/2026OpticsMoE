"""CLI-compatible entry point for the unchanged audited LGVQ split."""

from experiments.qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa.prepare_manifest import (  # noqa: F401
    _flatten_quality_records,
    main,
    prepare_manifest,
)

if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["main", "prepare_manifest"]


