# Caltech20 alternating2/4/6: verified equal100 baseline

Six arms each truly completed100 epochs/8600 Adam updates. All actual fit/test
children exited with original rc0; CPU campaign exited (its original exit code
was not captured, so no CPU rc0 is claimed). Independent saved science and
frozen TEST audits passed; every new selected model tested exactly once.
Final read-only collection matched original receipt hashes, PT/history and
prediction identities; no second forward or scientific audit was run.

|Case|Actual / selected E|Train %|Validation %|Test %|Test macro %|
|---|---:|---:|---:|---:|---:|
|L2_rho03|100 / 100|97.5948|71.3781|85.2229|67.1605|
|L2_rho0|100 / 100|94.4606|69.6113|84.1085|66.5121|
|L4_rho03|100 / 100|100.0000|67.4912|83.2364|62.8181|
|L4_rho0|100 / 96|97.6676|66.4311|81.7345|62.6133|
|L6_rho03|100 / 100|99.7813|63.2509|83.0426|60.3422|
|L6_rho0|100 / 100|98.9067|60.0707|78.0039|58.3628|

Residual-control TEST gaps2/4/6:1.11434109/1.50193798/5.03875969 percentage points.
L4 residual83.2364 exceeds80. L6 residual83.0426 is.1938pp (4 correct samples)
below L4; six-layer superiority is not achieved. None of the preferred10pp
gaps is achieved. Do not mix these original20 results with accepted old10.

## Architecture and electron count

One router, then E-G repeated1/2/3 times. Old actual Kather forward was checked
for this order/router count; old nine-expert geometry is not treated as the
same four-expert model. Current four224 experts/global478/RGB100/joint8bit.
Both rho.3/0 arms have identical learned electronic residuals and readout.

Actual ABO `standalone/model.py` RMS/fuse equations were extracted and exactly
matched in values/gradients. Each stage alpha is learned within(.7001,.95),
initial.8; separate detached-RMS branch normalization, weighted feature sum,
renormalization/restoration of electronic reference RMS. Alpha is not a fixed
probability blend or global optical power fraction. Original fixed.8/.2
probability draft never trained. Router is not repeated and has no fusion.

Tiny bias-free electronic3x3 kernel9params plus scalar alpha per main stage;
terminal pool16/nonaffineLN/Linear256to20=5140parameters. Total electronics
5160/5180/5200 (about20.16/20.23/20.31KiB float32 parameter storage), optical
phases479364/908552/1337740. All are trained from simulation stage. Future
physical deployment uses this same graph and trained state, not another head.

First b33850f7 source/run failed the GPU gate before any optimizer/test because
the last convolution bias had a null surrogate gradient. Failure/claim/source
were retained; corrected649ef57e removes redundant biases, no gate relaxation.
Only the corrected run here is an actually trained accepted baseline.

## Identity and data

Run:`caltech20_alt246_biasfree_rmslearned_linear_s17_equal100_20261016`. Frozen training source649ef57e580f8ee490210f417b7b55468d316b8c,
closure168; actual per-arm metadata commits:15a8c2d75c5465ad5f19c5e8d63e901b6b9aef45, 649ef57e580f8ee490210f417b7b55468d316b8c.
Later report HEAD differs from source; all registered blobs were independently
checked against each actual metadata commit.
Science SHA:`548c5a90f818b5cb6f3f77f34edbe429d28e6e5d20ffdf01a013437689303ba3`.
Saved TEST audit SHA:`93db16f16e4f8ecbae1b07591595a8399dfe308ae3245dd3f7b4927e13fde8a7`.

Original Caltech20 split1372/283/2064 is unchanged; cache SHA
a81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1.
No class re-screening or role changes. Test support is highly uneven; all20
class supports/recalls and macro accompany aggregate accuracy. Class screening
was validation-informed, older Caltech tests had been observed, and subsequent
feedback is development, not independent unseen generalization. No claim of
absence of near-duplicates/subject overlap follows from saved-math checks.

## Learning diagnosis and next stage

Residual selected train-val gaps2/4/6:26.2167/32.5088/36.5305pp. Deeper models
fit near100 but validation weakens; increasing depth has not automatically
improved held-out accuracy. Last10 validation balancedNLL still improved a
little, while training accuracy saturated; overfitting remains substantial.
No claim of full convergence or solved overfitting.

No-residual100 checkpoints/results now freeze. A separate registered residual
generalization refinement may add training-only horizontal flip, feature
dropout and stronger regularization, with unchanged evaluation graph and
separate short-segment science/test. It must retain this equal-budget baseline
and disclose extra budget/depth-specific training. Preferred gaps are goals,
never a reason to weaken controls or pick highest TEST epochs.

All600 real epoch rows are exported without interpolation or smoothing.
`plot_metrics.py` draws a paired TEST chart from60% (no macro series) and full
train/validation curves. Generated pictures require actual visual QA before
being described as delivered. No raw JSON, sample IDs, weights or full logs
are public; these files contain only aggregate MD/CSV and plotting source.

## Selected identities

- `L2_rho0`: selectedPT `2b08a39e843013448b127af028704e1c645486b92d9750ae17fd1f688f5aa01e`, test result `09e9519e8649eb1536f18c0020954c74fd8778d730a63164e50ff37416e17d59`.
- `L2_rho03`: selectedPT `7c7f8c6561e295131a5a0fc87f0d4b07917e694ecb8e2dabe45f158e3f062860`, test result `e90974c5b242b2dffce92f444405c2a906346e64d2cbe81b6f57fba69ba7ff18`.
- `L4_rho0`: selectedPT `a9e5c66c2f718a6bf78473f09eaf3cc46bf5ef792ea611c9946217566a6cfa74`, test result `770883a4f6d526d3135ebc7dd0601753c13d97b8768cccaa193ed5b38f62d807`.
- `L4_rho03`: selectedPT `462d127071b38031690fb047f99af081d59c6cb5b75b511583c321125a92f082`, test result `89e502689f0ce3beeacb024271ca98c7f234e029134d2e5ae13ebcc3f3c6657e`.
- `L6_rho0`: selectedPT `643c9f1a5c9eb8c3af31f8463f3258a3d10a14e9e9be952be2e92ce8879a7065`, test result `06da5181e8bf8fca04e65d78955d0406dfe8383d373496a47f1a9ea43d2609bb`.
- `L6_rho03`: selectedPT `3ca93e61a84a998cc3946e345d3fe0ef05b1b625b864f7418b01230567371b47`, test result `a5e85b88c373abd1af8eeec96aeab9e4f970a6cce36ebed87fcb3cf0554091a0`.
