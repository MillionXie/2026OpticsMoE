# Fast interpolation after retained electronic adaptation, depths4/6 only

Latest human: L2 is final because already handed to laboratory; urgently improve
L4/L6 only, continue original workflow after display recovered. No more L2 model
construction/forward/training/TEST. No lab interaction, timers, new data or head.
L2 two curriculum candidates were deliberately stopped by exact owned-PID SIGINT;
their rc-2 and partial artifacts remain. These stops do not invalidate the
separately completed L4/L6 producer and fit/test science receipts.

Hypothesis: gradually reducing alpha while all existing electronics adapt moved
VALID accuracy differently from TEST. For L4 endpoint local8, TRAIN94.94%,
VALID79.90% exceeded canonical VALID79.11%, but TEST81.02% regressed from82.00%.
Do not adopt it. A finite interpolation with the canonical state may preserve
more of its generalization while retaining some adaptation. L6 must first finish
its own eight epochs and both independent audits; no use of early results.

Use three independently verified states per depth: unchanged canonical, current
curriculum first minimum VALID-bNLL best, and its local8 last EMA. Optical phase
bytes must be identical in all three. For t in [0,.25,.5,1], form a single model
(1-t)*canonical+t*(best+lastEMA)/2 in Float64 then cast to each original dtype.
Integer buffers equal exactly. This is parameter interpolation, never prediction
ensembling; original router-once E/G depth, same 5140 linear head and upstream
electronic graph/parameter count remain. New optimizer steps0/joint epochs0;
parents added eight electronic epochs, no single averaged electronic epoch.

Select maximum VALID correct count, then minimum balanced NLL, then smallest t.
Only this selected model gets one new TEST evaluation; t0 reuses saved canonical
TEST exactly. No TEST-based fraction search. Global best remains canonical on
any lower or tied TEST correct count; earlier model identities retained. Whole
project is TEST-informed development, not independent generalization/equal-budget
pure residual causality. Controls50 and L2 e7c/5c8 final remain untouched.

Independent CPU mathematical checks create no model or forward and verify full
bank identities/producers rc0/actualexit, per-depth fit/test science-completed
SHA, Float64 interpolation, exact phase bytes, head5140 and unchanged shapes.
Actual GPU all20 TRAIN unchanged-parent probability/capture≤3e-6 before choices.
Independent saved science verifies arithmetic, VALID selection, ID/label/
features→head probabilities/CM/macro/NLL and unchanged reuse/new once TEST.
Separate exclusive claims/out/logs/exit/audit-completed SHA; preserve any failure,
no retry, source mutation or thresholds relaxed. Best single candidate PT only.

Source450=protected446 plus these four leaves, source-only Git. Two shared free
non-A100 GPUs at most; prefer3/4 released by intentional L2 stops, 2GiB+800MiB
each, five-global one-per-card guard, foreign untouched, release after stages.
Dataset rich20 original TRAIN3141/VALID383/TEST411 unchanged. Raw/PT/predictions/
IDs/logs private. No capture replacement: electronic/alpha changes alter future
CCD inputs even though optical phases are identical. Lab owner separately
coordinates all acquisition and same-head fitting with the human.
