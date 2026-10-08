# Decoder-optical instruction generation (2026-09-22)

This release implements two single-pass, text-conditioned product-image tasks with the optical branch in the first generator decoder block (16×16), parallel to the electronic residual branch. The fused decoder state is

`z = RMSMatch((1 - alpha) * electronic + alpha * optical)`

and both trained hybrid checkpoints satisfy `alpha >= 0.4`. The encoder and later upsampling layers remain electronic. Qwen3-VL-2B is frozen and used once to cache the six/eighteen instruction embeddings; it is not included in the trainable parameter count or required for repeated inference after caching.

## Tasks and data

- Backpack restyling: 20/3/3 ABO backpack identities for train/validation/test, 6 target styles and 3 textual paraphrases per style (360 train pairs and 54 held-out pairs). Images and manifests retain the source CC BY 4.0 attribution.
- Canonical-to-adjacent view: 27/5/4 ABO chair identities, each with ten clean turntable renders. Training uses the canonical product view as input and the immediately adjacent ±36° view as target, with 3 paraphrases per direction (162 train pairs and 24 held-out pairs). This uses chairs because the available ABO clean-render archive contains no eligible backpack turntable sequence.

Identity splits are disjoint. The models are 128×128, single-forward generators; there is no diffusion loop. A small conditional discriminator is used only while training (`adversarial_weight=0.01`) and is absent at inference.

## Held-out results

| Model | Trainable generator | Optical params | Final alpha | Test L1 | Test edge | Test background |
|---|---:|---:|---:|---:|---:|---:|
| Backpack electronic | 929,860 | 0 | — | 0.10621 | 0.13564 | 0.00855 |
| Backpack decoder-optical | 966,728 | 36,865 | 0.49974 | 0.05628 | 0.06450 | 0.00530 |
| View electronic | 1,926,710 | 0 | — | 0.08727 | 0.25606 | 0.04762 |
| View decoder-optical | 2,017,338 | 90,625 | 0.49992 | 0.08754 | 0.25486 | 0.04803 |

The backpack optical model is the strongest result: it preserves the product silhouette and produces clear text-selected color/material changes while halving held-out L1 relative to the matched electronic run. The view model is a conservative proof of concept: a bounded flow candidate is blended with at least 55% of the original image, so the object remains recognizable, but thin structures can show double edges. It should not yet be presented as photorealistic novel-view synthesis.

## Held-out image grids

- `backpack_style_electronic_test.png`
- `backpack_style_optical_test.png`
- `chair_view_electronic_test.png`
- `chair_view_optical_test.png`

Each row is input / deterministic target / prediction. The optical computation is a simulated Fourier feature transformation inside the decoder. It is not expected to show a visible diffraction pattern in the RGB output: later learned decoder layers convert the optical feature into the requested image.

Exact checkpoint hashes, metrics, and constraints are recorded in `results.json`. Evaluation is reproducible with `decoder_optical_evaluate.py`.
