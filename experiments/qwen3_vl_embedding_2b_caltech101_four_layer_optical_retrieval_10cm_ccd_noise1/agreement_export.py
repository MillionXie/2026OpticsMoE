"""Export sim-to-real probes with the selected CCD-noise model contract."""

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_warmstart5 import (
    agreement_export as _export,
)

from .modeling import build_hybrid_student, load_backbone
from .settings import load_settings


def main() -> int:
    _export.build_hybrid_student = build_hybrid_student
    _export.load_backbone = load_backbone
    _export.load_settings = load_settings
    return _export.main()


if __name__ == "__main__":
    raise SystemExit(main())
