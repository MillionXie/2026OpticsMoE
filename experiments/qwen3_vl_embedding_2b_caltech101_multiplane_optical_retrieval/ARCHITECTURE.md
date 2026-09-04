# Architecture and comparison contract

## Shared Qwen boundary

```text
RGB image
-> frozen Qwen processor and Vision patch/position embedding [Nv,1024]
-> Vision optical stack
-> final Vision CCD/readout -> [Nv,1024]
-> frozen Qwen main merger
-> one visual-token injection into frozen Qwen token embeddings [B,S,2048]
-> Language optical stack
-> final Language CCD token features [B,S,224]
-> valid-token mean/max pooling [B,448]
-> LN -> Linear(448,64) -> L2-normalized retrieval embedding
```

DeepStack is disabled.  No native or copied attention block is enabled.

The entrance `Linear(D,224) -> LN -> Softplus` is a modality interface needed
to encode signed Qwen hidden states as nonnegative amplitude.  It is not an
electronic residual path.  The electronic Top-k router is evaluated at the
entrance only, except in the explicitly named dynamic-router ablation.

There is no trainable or frozen electronic identity/residual branch around an
optical stack: both replacement hooks reject `residual_base` at runtime.  The
task loss therefore always traverses the optical path.  There is no separate
"optical contribution" penalty, because there is no bypass whose gate needs
to be suppressed.  `architecture_report.json` nevertheless reports optical
phase, router and remaining electronic-interface parameter counts separately.

## Continuous MoE

```text
weighted MoE4 amplitude load, router R1
-> Expert plane 1 -> P(d)
-> Expert plane 2 -> P(d)
-> Expert plane 3 -> P(d)
-> Expert plane 4 -> P(d)
-> Global phase -> P(d_detector)
-> one CCD
```

Every expert plane owns four independent 224x224 phase masks.  The four planes
share spatial expert identities but not phase parameters.  Routing weights are
applied only to the initial amplitude.  Static expert windows are physical
finite apertures, not sample-dependent electronic masks.

## Continuous D2NN

```text
224x224 amplitude
-> Phase 1 -> P(d)
-> Phase 2 -> P(d)
-> Phase 3 -> P(d)
-> Phase 4 -> P(d)
-> Phase 5 -> P(d_detector)
-> one 224x224 CCD
```

It has 250,880 phase parameters per modality.  The continuous MoE has
1,031,300 phase parameters per modality.  This difference is reported rather
than hidden: D2NN is a conventional same-aperture baseline, not a
parameter-matched MoE replacement.

For compatibility with the shared fixed-column training CSV, D2NN phase
diagnostics place planes 1--4 in the `*_expert_*` columns and plane 5 in the
`*_global_*` columns.  This is a logging alias only; plane 5 is not a MoE
global mask and the architecture JSON labels all five planes as D2NN.

## D2NN OEO + sigmoid

```text
Phase 1 -> P(d) -> CCD -> full-aperture normalization -> sigmoid -> reload
Phase 2 -> P(d) -> CCD -> full-aperture normalization -> sigmoid -> reload
Phase 3 -> P(d) -> CCD -> full-aperture normalization -> sigmoid -> reload
Phase 4 -> P(d) -> CCD -> full-aperture normalization -> sigmoid -> reload
Phase 5 -> P(d_detector) -> final task CCD/readout
```

The four intermediate conversions are whole-aperture D2NN conversions.  No
expert crop, router mask or routing weight is introduced into this baseline.

## Fixed-router OEO (supplemental)

After every expert propagation, the four expert ROIs are square-law detected,
independently normalized, passed through sigmoid, multiplied by the original
input routing weights, and reloaded with zero phase.  The same selected expert
IDs and weights therefore remain active at all four expert planes.

## Dynamic-router OEO

Each expert plane owns an independent electronic router.  After OEO, selected
expert responses are averaged into a canonical 224x224 field.  The old routing
weights are removed, the next router predicts a new Top-2 set and new weights,
and that result is loaded for the next expert plane.  The last expert response
is reloaded using its current router and then enters the global phase.

## Fairness contract

All primary release configs and the supplemental config inherit the same:

- image split and ten target classes;
- augmentation and PK batch order seed;
- Qwen checkpoint and processor budget;
- 28 partial epochs, cyclic sampler, optimizer, schedule and loss coefficients;
- 532 nm wavelength, 16 um sampling and 10 cm propagation;
- retrieval readout, embedding dimension and evaluation code.

Only architecture-required parameter groups differ.  The phase and total
trainable parameter counts are saved in every `architecture_report.json`.

## Ten-hour sampling budget

The measured full-dataset epoch time is 449--518 seconds, so four 80-epoch
runs would require roughly 43--46 hours.  Release configs instead use 28
partial epochs, 65 optimizer steps and `10 classes x 2 images = 20` samples per
step.  Each epoch therefore contains 1,300 samples.

The first measured partial epoch on the available RTX 3090 took 297.4 seconds
including the 200-image test evaluation.  Twenty-eight epochs leave practical
startup/checkpoint margin under the ten-hour four-run budget.

`cyclic_balanced` maintains one deterministic circular queue per class.  It
does not repeat an image within a class until every image in that class has
been visited.  The largest selected Caltech-101 class has 777 training images;
at 130 images per class per epoch, every training image is covered within at
most six epochs.  All four variants receive exactly the same batch membership
and order for a given epoch and seed.

The 10 cm geometry is a controlled free-space simulation.  It is not presented
as a literal chip thickness.  Short-distance tape-out feasibility requires a
separate sampling/NA study; distances remain explicit in YAML for that reason.
