# Spatial single-video 4-frame rejected upper bound

> **Rejected after architecture audit:** this checkpoint contains a frozen
> pretrained ResNet18 front.  It is preserved only as a capacity upper bound;
> it is not the formal LightGenV2 Spatial result and must not be deployed.

The selected strict two-branch Spatial candidate reaches SRCC **0.666503** on
all 558 test videos. The same checkpoint with every optical stage bypassed
reaches 0.584650, so the measured optical contribution is +0.081853 SRCC.

There is one electronic residual branch, one optical branch, and one MOS
readout. A frozen attention-free ResNet18 stem through layer3 supplies a
bounded correction only inside E1; it is not an independent scoring branch.
The optical masks and optical Top-2 routers remain active and unchanged.

The authoritative architecture, metrics, SHA256 values, and reproduction
command are in
[`SPATIAL_OPTIMIZATION_RESULT.md`](../../../../../../experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/SPATIAL_OPTIMIZATION_RESULT.md).
