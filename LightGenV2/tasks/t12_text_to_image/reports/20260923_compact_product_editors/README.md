# Compact Qwen–optical product editors (2026-09-23)

This report freezes the first complete comparison between the full-Qwen electronic baseline and the three-layer-Qwen compact optical editor. Both models are single-pass supervised image generators: one input image plus one text instruction produces one new RGB image. Neither training nor inference uses a GAN or an iterative diffusion loop.

## Main result

| model | counted parameters | Qwen blocks | test latent MSE ↓ | exact text control ↑ | first-block-to-image latency ↓ |
|---|---:|---:|---:|---:|---:|
| Full Qwen + electronic decoder baseline | 1,824,200,930 | 28/28 | 0.18630 | 50% exact background combination | 63.49 ms |
| Compact Qwen + optical decoder | 494,852,487 | 3/28 | **0.16354** | **100% exact background combination** | **35.06 ms** |

The compact optical model has 72.87% fewer counted parameters, 12.22% lower test latent MSE, and 1.81× lower latency (44.77% reduction) under the agreed hardware timing convention.

Parameter accounting starts at the first Qwen Transformer block and includes the final norm, frozen VAE encoder, conditional adapter, generator, router, and VAE decoder. The shared 311,164,928-parameter token embedding is reported in every raw result but excluded by project convention. The Qwen vision tower and language-model head are not used.

## Architecture

```text
input RGB ── frozen VAE encoder ── reference latent ─────────────┐
                                                                 ├─ one compact UNet call ─ VAE decoder ─ edited RGB
prompt ── token embedding (excluded) ─ 3 Qwen blocks ─ adapter ──┘
                                      │
                                      └─ attribute/catalogue router

decoder entrance: shared feature ─┬─ parameter-free identity residual ─┐
                                  └─ phase-only optical MoE block ─────┴─ RMS fusion (α≈0.499)
```

The optical block replaces a deleted decoder residual transform; it is not appended beside a complete electronic copy. Seven shallow or redundant cross-attention modules were removed (72,095,040 parameters). The remaining UNet has 255,888,681 parameters, including a 1,146,597-parameter differentiable optical block. The frozen VAE encoder has 34,163,664 parameters and its decoder has 49,490,179.

## Tasks and data

### Background replacement

ABO CleanRender CC BY 4.0 lamp renders are composited into clean procedural rooms. The instruction controls four room layouts, three colour temperatures, two brightness levels, and two window-light directions (48 combinations). Training uses 576 source identities and 4,608 pairs; validation and test each use 64 disjoint source identities, 512 pairs, and eight target combinations excluded from training.

Compact-model test results:

- latent MSE: 0.16354;
- background L1: 0.26547;
- foreground change L1: 0.04091;
- room/tone/brightness/direction and exact-combination accuracy: 100%;
- improvement over copying the input: 73.58%.

The full electronic baseline reaches 0.18630 latent MSE and 50% exact-combination accuracy. Its best checkpoint is epoch 1; later epochs overfit the training combinations.

### Catalogue object replacement

The first random-target version averaged hundreds of weakly described ABO instances and produced blurry generic chairs/tables. The corrected task uses a four-item CC BY 4.0 catalogue with unambiguous prompts: black wood chair, blue wood chair, oak hardwood table, and cinder-gray metal table. Every source lamp is paired with every catalogue target, yielding 2,304 training pairs and 256 validation/test pairs. Test input identities remain disjoint; catalogue targets are deliberately fixed, so this evaluates controllable scene editing rather than unseen-object generalization.

The final separate object-replacement weight has 494,838,144 counted parameters. Test latent MSE is **0.02766**, four-way catalogue accuracy is **100%**, and error improves **94.30%** over copying the input. Hard cases can still retain a faint trace of the source lamp, which is the main remaining visual limitation.

## Timing protocol

- Hardware: one server GPU, batch size 1, 256×256 output, 30 warm-ups and 100 measured runs.
- Boundary: input to the first Qwen Transformer block through the final composited RGB image.
- Tokenization, model loading, and token-embedding lookup are outside the interval.
- The compact measurement bypasses both the software FFT simulator and its parameter-free electronic residual branch: 28.7924 ms mean electronic time.
- Physical optical time is added as `1.0447 × 6 = 6.2682 ms`, giving 35.0606 ms total.
- Baseline mean is 63.4864 ms. Its complete electronic decoder is executed normally.

## Artifacts

- `compact_background_test_grid.jpg`: identity-disjoint, held-out-combination compact optical results.
- `electronic_baseline_test_grid.jpg`: matched full-Qwen electronic baseline results.
- `catalogue_object_validation_grid.jpg`: four controlled catalogue edits after epoch 4 (`input | target | generated`).
- `latency_comparison.json`: raw timing distribution and speedup.
- `*_summary.json` and `*_test_results.json`: raw metrics, architecture, and parameter accounting.

Server checkpoints (not committed because each is about 0.5–0.65 GB):

- `/DATA/DATA1/guest3/t12_assets/runs/abo_scene_replace_3layer_compact_optical_v1/best_model.pt`
- `/DATA/DATA1/guest3/t12_assets/runs/abo_object_catalogue_3layer_compact_optical_v2/best_model.pt`
- `/DATA/DATA1/guest3/t12_assets/runs/abo_scene_replace_28layer_electronic_baseline_v1/best_model.pt`
