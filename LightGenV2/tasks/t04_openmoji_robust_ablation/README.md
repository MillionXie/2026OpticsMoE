# OpenMoji four trainings, five displayed conditions

Pinned reference: user's shared-readout `.8890` checkpoint, SHA-256
`579b19e78befe93cd258ce02308235f1a9176316f37bc42fc2e1b6ff06d38e9f`.
The reference package is `handoffs/sister_simulation_final_20260927/openmoji_final_simulation_bounded.zip`.
This study uses the same T04 architecture and the earlier shared-readout checkpoint
`a69ddcee...` as the **common optimization start** for four independently trained
groups. Therefore r0 has no *added* detector/DC/grid perturbations, but the warm
start is not a model trained from scratch and saw the original TRAIN data.

| Display | Trained checkpoint | Added perturbation | Test condition |
| --- | --- | --- | --- |
| G1 ideal theory | r0_base | none | ideal 17 µm simulation |
| G2 basic device | r0_base | none | mapped device / real CCD |
| G3 + detector noise | r1_ccd | gain/offset/read noise | same device |
| G4 + coherent DC30% | r2_ccd_dc30 | phase leakage intensity fraction .30 | same device |
| G5 + training-in-graph grid | r3_ccd_dc30_grid | differentiable 17→8→17 raster proxy | same device |

All four use identical zero-preserving phase-preserving `tanh(abs(field)/.5)` before
propagation. Hardware BMP must be **only** `round(255*amplitude)`: no additional
batch-peak scaling, gamma or clipping. The G5 roundtrip is an explicitly limited
proxy; it does **not** model propagation at 8 µm and must not be labelled exact
hardware equivalence. Phase dropout .08 is common to all groups.

`profiles.bmp_amplitude(field)` is the reference 8-bit exporter for the shared
input-amplitude contract. The optical bench still needs the usual phase/SLM
orientation bridge and a six-stage numerical equivalence test before capture;
this exporter alone is not a deployment script.

Selection is 4000 FIT/1000 VAL from the original 5000 TRAIN; the original 1000
TEST is evaluated once per validation-selected checkpoint. Original warm start
was previously exposed to TRAIN (and historically selected using TEST), so this
is a continuation comparison, not a fresh unbiased external benchmark. Preserve
the same split/seed and evaluate hardware on the same 1000 TEST scenes.

Commands on the pinned server checkout:

```
python -m LightGenV2.tasks.t04_openmoji_robust_ablation.train --group r0_base --quick --epochs 1 --steps 2
python -m LightGenV2.tasks.t04_openmoji_robust_ablation.train --group r0_base --epochs 10 --steps 100
```

Each group has its own `best.pt`, `last.pt`, `history.json`, `split.json`,
`protocol.json`, and `report.json`. Do not use TEST to choose an epoch or to
silently pick a group. Before physical capture, audit actual source/BMP
equivalence and phase geometry; capture full datasets one optical layer at a
time with persistent devices.

After all four groups complete, `python -m
LightGenV2.tasks.t04_openmoji_robust_ablation.evaluate_conditions` emits the
predeclared five-condition TEST comparison. G2–G5 use the **same** grid-raster
inference proxy; G1 alone is ideal17um. This extra evaluation is not used for
checkpoint selection and must not be reported as physical CCD accuracy.

## Electronic-capacity follow-up

The standard shared readout has 381,976 parameters, including 74,112 in an
additional FiLM, 226,178 in two conditioned residual convolutions, and 38,976
in the decoder's pre-convolution. Initial structure-removal probes `slim`,
`slim_one`, `lite`, and `lite_one` showed the pretrained transforms are not
dispensable: abrupt deletion hurt TRAIN-derived validation markedly. They are
diagnostics, not final models. The next calibration compares `lite` (retains
decoder pre-convolution) with `lowrank64` (SVD-factorized conditional/pointwise
transforms) using the user's own `.8890` bounded checkpoint as a common start.
The low-rank mapping is initialized from truncated SVD, not fresh random
matrices. Neither calibration touches TEST. The optical path, router, phase
count, 17 µm / 10 cm geometry, and bounded BMP contract stay fixed. Select one
meaningful smaller architecture on the TRAIN-derived validation split before
retraining all four perturbation profiles at that exact same capacity; do not
tune capacity to TEST or assert a physical result from simulation alone. For
the common 12-epoch/100-step capacity calibration, choose the smallest of
`lowrank32`, `lowrank48`, `lowrank64`, and `lite` whose best validation
Changed-cell Accuracy is at least 0.942 (within two percentage points of the
reference checkpoint's 0.962 on the identical TRAIN holdout). If none meet
that guard, retain the strongest validation candidate and report the miss;
the 0.889 TEST target must not be selected on TEST itself.

The fixed calibration selected `lowrank48`: 234,520 readout parameters versus
381,976 standard (-38.6%), best TRAIN-derived validation 0.958. Four formal
12-epoch runs at that same capacity are named `compact_lowrank48_<group>`.
Their five-condition simulation results are 0.9135 / 0.9030 / 0.9045 /
0.8875 / 0.9005. These are **not CCD measurements**. Full metrics, split
and weight hashes are in `handoffs/openmoji_robust_ablation_20260928/compact`.
