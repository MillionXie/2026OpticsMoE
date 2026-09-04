# Controlled comparison plan and status

Snapshot date: 2026-08-27.

## What this experiment measures

This is a ten-class Caltech-101 image-retrieval experiment.  It keeps the
Qwen3-VL processor, Vision patch/position embedding, token embedding and main
Vision merger frozen.  Both the Vision processing stack and Language
processing stack are replaced by the selected optical variant.  The training
objective is the same supervised contrastive retrieval plus gallery/prototype
objective for every group; Teacher KD is disabled in the release recipe.

DeepStack, copied/native attention, electronic Transformer/Mixer blocks and
electronic identity residuals are disabled.  The output is a 64-dimensional
L2-normalized retrieval embedding and evaluation reports Top-1, Top-3 and MRR.

## Four primary comparisons

1. `d2nn_continuous`: five continuously propagated 224x224 D2NN phase planes,
   then one final CCD.
2. `d2nn_oeo_sigmoid`: the same five D2NN phase planes, with full-aperture
   square-law detection, non-affine normalization, sigmoid and zero-phase
   reload after planes 1--4; plane 5 ends at the common final CCD.
3. `moe_continuous_fixed_router`: four MoE4 expert planes plus one global
   phase, continuously propagated; the input Top-2 route is loaded once.
4. `moe_oeo_dynamic_router`: four MoE4 expert planes plus one global phase;
   each expert plane is followed by per-expert OEO and the next plane receives
   a newly predicted independent Top-2 route.

`moe_oeo_fixed_router` remains available only as a supplemental ablation: it
has the same per-expert OEO boundaries but reuses the input route at all four
expert planes.

## Optical contribution audit

There is no explicit optical-contribution regularizer.  There is also no
electronic bypass around either optical stack, so every task-loss path must
pass through the optical propagation.  Both replacement hooks reject an
electronic `residual_base` at runtime.

The experiment is nevertheless hybrid rather than all-optical.  Trainable
electronics remain at physical interfaces:

- hidden-to-224 dimensional amplitude interface and its normalization;
- electronic Top-k routers in MoE variants;
- final CCD conditioning and 448-to-64 retrieval readout.

The frozen Qwen patch/token embeddings and Vision merger also remain.  The
release D2NN has 501,760 trainable phase parameters out of 1,451,264 trainable
parameters (34.6%).  Continuous/fixed-router MoE has 2,062,600 phase parameters
out of 3,013,680 (68.4%).  New architecture reports separate phase, router and
remaining electronic-interface counts so these contributions cannot be
conflated.

An additional optical-contribution loss is not appropriate here because there
is no residual gate to suppress.  If causal contribution must be quantified,
the clean follow-up is an inference-only phase ablation (trained phases versus
zeroed/shuffled phases) or a frozen-electronics training ablation, reported
separately rather than changing the four controlled models.

## Completed preliminary results

The numbers below use the deliberately labelled, selection-biased
`best_observed_test` checkpoint only for trend inspection.  Formal reporting
should also retain final-epoch and predeclared checkpoint metrics.

| Group | Best observed Top-1 | Epoch | Final Top-1 | Final Top-3 | Final MRR |
|---|---:|---:|---:|---:|---:|
| Continuous D2NN | 80.5% | 26 | 79.5% | 92.5% | 0.8706 |
| Continuous MoE | 71.5% | 24 | 70.0% | 86.5% | 0.8007 |
| Repeated-router OEO MoE | 32.5% | 12 | 25.5% | 56.5% | 0.4704 |
| Fixed-router OEO MoE (supplement) | 29.0% | 16 | 27.0% | 54.5% | 0.4718 |
| D2NN OEO + sigmoid | 81.5% | 25 | 80.0% | 93.5% | 0.8736 |

The completed D2NN OEO baseline reaches 82.2% final train Top-1 and slightly
exceeds continuous D2NN on final test Top-1 (80.0% versus 79.5%); the 0.5-point
difference is too small to claim an advantage, but it clearly rules out a
generic sigmoid-OEO failure.  Both OEO MoE variants instead fail to fit the
training set (final train Top-1 around 20--23%).  The dominant failure is
therefore specific to the current MoE OEO interaction: per-expert
normalization, routing-weight reload and/or repeated sparse allocation.  The
fixed-router supplement being just as weak as repeated routing shows that
recomputing the router is not the sole cause.
