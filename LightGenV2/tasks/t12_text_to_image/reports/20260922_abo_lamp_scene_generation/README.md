# ABO lamp scene generation — large optical/electronic model

This run replaces a white product background according to a Qwen instruction while keeping the input lamp unchanged. It is the replacement for the rejected masked-restoration task.

## Task and data

- Source: official ABO No-BG Blender renders, CC BY 4.0.
- Category: lamp only; one object per image.
- Identity-disjoint source split: 48 products / 576 views train, 8 / 64 validation, 8 / 64 test.
- Each source view has six counterfactual targets with the same lamp and different text-selected scenes: warm minimalist, cool modern, dark luxury, terracotta gallery, sage reading nook, and concrete loft.
- Pair counts: 3,456 train, 384 validation, 384 test.
- Backgrounds are deterministic procedural assets and contain no additional copyrighted imagery.
- Official ABO alpha is retained. The previously white-matted antialiased edge is inverted before compositing on a new background.

For a counterfactual comparison, the input image and Gaussian seed are held fixed and only the prompt changes. The task therefore cannot be solved by copying the input or ignoring the text.

## Model

```text
white-background RGB ── frozen VAE encoder ── reference latent ─┐
real Gaussian seed ──────────────────────────────────────────────┼─ 8-channel input
prompt ── frozen Qwen3-VL-2B ── 4.27M condition adapter ────────┘
                                      │
                         one BK-SDM-v2-Tiny UNet call
                                      │
             deepest decoder: electronic(shared input)
                              || optical FFT MoE(shared input)
                              └─ RMS-matched alpha fusion
                                      │
                         frozen VAE decoder (one call)
                                      │
             exact source lamp alpha-composited on generated scene
```

- No diffusion loop: one UNet call and one VAE decode.
- No GAN.
- Optical and electronic branches are parallel, not serial.
- Optical location: the first/deepest UNet decoder up block.
- Learned optical fusion alpha: **0.50156**, constrained to `[0.40, 0.75]`.
- Optical branch: **2,786,277 parameters**.
- Generation tail from the first generation block to RGB: **383,403,250 parameters**, below the 500M large-model limit.
- Qwen is frozen and excluded from the requested generation-tail count.
- The baseline definition remains **Qwen + decoder only**; no alternative baseline is substituted here.

## Identity-disjoint test result

The test contains 64 unseen input views and six prompts per view, for 384 generations.

| Metric | Result |
|---|---:|
| Scene instruction accuracy | **100.0%** |
| Latent MSE | **0.10163** |
| Background latent L1 | **0.17446** |
| MSE improvement over copying the white input | **94.00%** |
| Inference iterations / UNet calls | **1 / 1** |

The qualitative grid is ordered `input | paired target | generated`. The same test input and seed are used in all six rows; only the text instruction changes.

## Reproducibility

- Final checkpoint: `/DATA/DATA1/guest3/t12_assets/runs/abo_lamp_scene_big_optical_v4/best_model.pt`
- Source dataset: `/DATA/DATA1/guest3/t12_assets/datasets/abo_lamp_scene_source_v1`
- Final latent cache: `/DATA/DATA1/guest3/t12_assets/datasets/abo_lamp_scene_latents_v4`
- Test report: `/DATA/DATA1/guest3/t12_assets/reports/abo_lamp_scene_big_optical_v4_test`
- Model/data configuration: `configs/abo_lamp_scene_big_optical_matte.yaml`
- All task GPUs were released after evaluation.

## Known limitation

The six backgrounds deliberately form a controlled, clean benchmark rather than unrestricted natural-image generation. Fine high-contrast edges can still show mild frozen-VAE smoothing, especially in the dark scene. The current version is suitable for validating text-controlled, one-pass optical/electronic scene generation; broader real-room backgrounds should be treated as a later dataset expansion, not mixed into this initial benchmark.
