"""Hardware export/fine-tuning CLI for the selected CCD-noise model.

The tensor topology is identical to warmstart5, but this shim deliberately
loads the 1% optical-fusion contract and the trained CCD-noise replacement.
Using the warmstart5 shim would reinterpret the same gate logits under its old
5% lower bound during hardware fine-tuning.
"""

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
    hardware_bridge as _bridge,
)

from .modeling import build_hybrid_student, load_backbone
from .settings import load_settings


def main() -> int:
    _bridge.build_hybrid_student = build_hybrid_student
    _bridge.load_backbone = load_backbone
    _bridge.load_settings = load_settings
    return _bridge.main()


if __name__ == "__main__":
    raise SystemExit(main())
