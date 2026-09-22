# ABO text-selected product restoration — large optical/electronic model

## Result

This run implements a non-recurrent image-and-text generation task on official
ABO No-BG renders.  A source image contains two missing product regions and the
instruction selects exactly one region to restore.  Each source is paired with
two counterfactual targets: the image pixels and Gaussian seed are identical,
and only the instruction changes.

The final checkpoint is
`/DATA/DATA1/guest3/t12_assets/runs/abo_product_repair_big_optical_rgb_v6/best_model.pt`.
It was selected at RGB-refinement epoch 4.

## Dataset

- Source: Amazon Berkeley Objects No-BG Blender renders.
- License: CC BY 4.0; the official license file and source-member provenance are
  retained with the prepared dataset.
- Categories: pillow, cabinet, dresser, ottoman.  Chair-like pillow aliases such
  as chair/lounger/seat cushions are removed by title filtering.
- Identity-disjoint split: 48 train, 12 validation, and 12 test product
  identities.
- Base images: 960 train, 72 validation, 72 test.
- Counterfactual pairs: 1,920 train, 144 validation, 144 test samples.
- Resolution: 256 × 256; one centered product per image.

`dataset_pairs.jpg` shows `clean | two-hole input | one-region target`.

## Architecture

```text
input RGB ─ frozen VAE encoder ─ reference latent ─┐
                                                   ├─ 8-channel one-pass UNet editor ─ VAE decoder ─ RGB
real seed ─ small Gaussian texture latent ─────────┘                 │
                                                                     │
text ─ frozen Qwen3-VL-2B ─ condition adapter ─ cross-attention ─────┤
                          └─ 10K upper/center/lower router ─ gate ────┘

deepest decoder input ─┬─ original electronic up block ─┐
                       └─ Router→Top-2 optical experts  ─┴─ detached-RMS fusion
```

- Electronic and optical branches consume the same decoder activation and are
  fused only after both branches finish; there is no electronic-to-optical
  cascade.
- The optical branch is at the first/deepest UNet decoder up block.
- Optical alpha is constrained to `[0.40, 0.75]`; final alpha is `0.50394`.
- Inference uses one UNet call, one VAE decode, and no denoising loop.
- No GAN is used.  Training uses latent reconstruction, selected-region,
  preservation, counterfactual pair-difference, region-routing, detail, and
  decoded-RGB consistency losses.

### Parameter accounting

| Component | Parameters |
|---|---:|
| BK-SDM-v2-Tiny UNet including decoder optical branch | 329,623,401 |
| Frozen VAE decoder | 49,490,179 |
| Qwen-to-decoder condition adapter | 4,273,280 |
| Text region router | 10,243 |
| **Generation tail total** | **383,397,103** |
| Optical branch inside the UNet total | 2,786,277 |

The frozen Qwen3-VL-2B text backbone is upstream of the requested
first-generation-block parameter boundary and is reported separately rather
than hidden in the 383.40M tail number.

## Identity-disjoint test

| Metric | Result |
|---|---:|
| Test counterfactual samples | 144 |
| Underlying unseen test images | 72 |
| Text region routing accuracy | 97.92% |
| Latent MSE | 0.03524 |
| Copy-input latent MSE | 0.05926 |
| Improvement over copying | 40.52% |
| Selected-region latent L1 | 0.51359 |
| Non-selected-region change L1 | 0.25893 |

`test_grid.jpg` uses the layout `input | exact target | generated`. Adjacent
rows share the same input and seed but use opposite repair instructions.

The model reliably preserves global product geometry and routes upper/lower
edits to different regions.  Remaining limitations are visible rather than
hidden: some filled regions are darker or softer than the exact target, and
nearby upper/center holes can partially overlap through the smooth spatial
gate.  These are suitable targets for a later mask-localization or perceptual
fine-tuning stage; they do not require adding another large electronic block.

## Baseline contract

The comparison baseline remains the previously agreed **Qwen + decoder**
baseline.  No matched or differently defined baseline is substituted here.
This report covers the large optical/electronic model only; latency measurement
and the <=50M small model are separate follow-up milestones.

