"""Export all four native phase BMPs from a CCD-noise checkpoint."""

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
    export_phase_bmps as _export,
)
from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
    hardware_bridge as _bridge,
)

from .modeling import build_hybrid_student, load_backbone
from .settings import load_settings


def main() -> int:
    _bridge.build_hybrid_student = build_hybrid_student
    _bridge.load_backbone = load_backbone
    _bridge.load_settings = load_settings
    _export._load_model = _bridge._load_model
    _export.load_settings = load_settings
    return _export.main()


if __name__ == "__main__":
    raise SystemExit(main())
