# Caltech10 accepted optical baselines (human confirmed 2026-10-16)

The human accepts current L2/L3/L4 accuracy, asks to avoid model-version confusion,
and selects the earliest original model attaining the accepted L4 score. This
registry fixes the original checkpoints by parameter digest and actual file SHA;
it does not rename, move, overwrite, copy, retest, retrain or delete any run/PT.
Future hardware packages should reference these IDs and verify BOTH hashes before
use, rather than choosing a last checkpoint or a later resume container.

| Accepted ID | Selected epoch | First saved segment | Train / val / test % | Test macro % | Gap vs frozen100 pp |
|---|---:|---:|---:|---:|---:|
| Caltech10_L2_rho03_E1143_accepted_20261016 | 1143 | 1150 | 80.6833 / 75.7962 / 81.7239 | 76.7726 | 4.037098 |
| Caltech10_L3_rho03_E123_accepted_20261016 | 123 | 125 | 90.9330 / 81.5287 / 87.4523 | 84.6657 | 7.201309 |
| Caltech10_L4_rho03_E488_accepted_20261016 | 488 | 500 | 99.6058 / 87.2611 / 93.3988 | 90.1104 | 9.492635 |

The segment epoch is a SAVE/verification endpoint, not the epoch of the chosen
weights or a new training run. In particular L4 E488 first appears in the original
185 run block0016 (actual500). Later575/650/725/745/765 states must not replace the
model's canonical provenance. Later PT containers can have different file SHA due
to metadata/history; identical model parameter digests mean identical weights.
L2 E1143 first appears at its1150 saved segment; the later1170 review does not
change its parameters. L3 E123 first appears at125.

## L4 follow-ups and the metric tradeoff

Original E488: val87.2611465%, balancedNLL0.5708260277, test93.3987998%,
test macro90.1104069%, test balancedNLL0.5503175680. Its gap to frozen no-residual
100/E100 test83.9061648% is9.4926350pp. Refinement193 retained E488 per its frozen
profile and saved final report; current read-only parameter checks directly verify
original185, SAM197, quantSAM203 and color213 reuse the same E488 digest.

Dropout is the genuine exception to 'nothing changed': it selected E695 with
better validation balancedNLL0.5600753682 (about1.88% lower), but val accuracy
85.9872611%, test92.7441353%, test macro89.9087696%, test balancedNLL0.5556490825
are all worse than E488. It is retained as historical evidence, not adopted into
the current baseline. No highest-test epoch was substituted into any old run.

All no-residual models remain the existing actual100-epoch controls. The human's
current acceptance does not alter old target flags or claim equal training budget.
These are fixed validation-informed classes, single-seed test-informed development
results with extra residual training; not independent unseen generalization.

## Electronic size: separate the accepted model and the hardware adapter

All three accepted SIMULATION models have **0 trainable electronic classification
parameters**. All trainable weights are optical phases: L2 479364 / L3 680068 /
L4 908552. Fixed camera/ROI operations, OEO, energy normalization and argmax still
require electronic computation; zero learned electronic head is not zero CPU work.
The frozen L4 teacher used to train L2 is not part of L2 inference.

The separate hardware-side implementation `lab_readout_20261016.py` contains an
8554-parameter residual head: final478x478 CCD -> fixed mean/log normalization ->
fixed16x16 average pooling ->256 inputs ->Linear256x32+ReLU->Linear32x10. The ten
outputs add to log optical probabilities before softmax. Input is actual final CCD,
not a parallel rawRGB classifier. Train-only mean/std add512 fixed FP32 values.
This header/code count is not evidence that hardware fitting/deployment is complete,
and no physical scores are claimed here. The old RESISC490-parameter RGB fallback
was not adopted and is a different task; do not count it in accepted Caltech models.

Head count: 8554; FP32 head weights 34216 bytes = 33.4141KiB; with mean/std about35.4141KiB, excluding serialization overhead. Dense-layer MACs 8512 per sample, excluding CCD preprocessing; this is arithmetic, not measured latency. Inspected head source SHA e8bab0c31475801c2210848c07b44ca7628b10dc9328e12aca5b2741b752c627.

Private read-only collection verified own producer exit/rc0, exclusive science/completed SHA and saved test audit/completed SHA, current source identities and selected parameter digests; 0 new model constructions/forward/optimizer/test. No lab machine was contacted. All existing source/PT/data/failures/claims/plots and other windows remain protected. T18 and09 timers remain paused at the earlier human request. Aggregate CSVs contain identity and summary metrics only; raw PT/IDs/predictions/logs stay private.
