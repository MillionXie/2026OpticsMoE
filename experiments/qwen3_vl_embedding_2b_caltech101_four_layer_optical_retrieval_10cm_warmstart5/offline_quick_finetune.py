"""Warmstart5 entry point for the Qwen-free Language-global offline tail.

The generalized implementation is shared with the audited 10 cm robust
project and validates both the checkpoint architecture and its corresponding
0.05/0.10 fusion floor.  Importing this module loads PyTorch only; it does not
import Qwen, Transformers, or the full optical simulator.
"""

from __future__ import annotations

from experiments.qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust import (
    offline_quick_finetune as _implementation,
)


EXPECTED_CHECKPOINT_ARCHITECTURE = (
    "vision2_language2_moe4_10cm_warmstart5_stage_b_v1"
)
if _implementation.SUPPORTED_CHECKPOINT_ARCHITECTURES.get(
    EXPECTED_CHECKPOINT_ARCHITECTURE
) != 0.05:
    raise RuntimeError("Shared offline runner lacks the warmstart5 5% contract")

OfflineQuickData = _implementation.OfflineQuickData
load_offline_quick_data = _implementation.load_offline_quick_data
load_offline_tail = _implementation.load_offline_tail
finetune_offline_quick = _implementation.finetune_offline_quick


def main() -> int:
    return _implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
