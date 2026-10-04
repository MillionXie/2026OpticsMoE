# T12 three-task large/small checkpoint bundle

The current primary bundle contains three 282M latent editors and three
18.99M direct-RGB students.  All six primary checkpoints retain a Qwen text
Transformer front-end.  The earlier 3.27M byte-GRU models are retained only as
no-Qwen ablations and must not be reported as the proposed small architecture.
No checkpoint was overwritten. All models use one generator call, have no
diffusion loop or GAN, and do not paste input pixels back after inference.

## Checkpoints

| Task | Large checkpoint | Counted parameters | Small checkpoint | Parameters |
|---|---|---:|---|---:|
| text-guided background replacement | `abo_scene_replace_2layer_progressive_282m_v1/best_model.pt` | 282,431,515 | `abo_background_qwenmini2_optical_under50m_v2/best_model.pt` | 18,975,128 |
| text-guided product-form redesign | `abo_global_redesign_2layer_282m_v1/best_model.pt` | 282,417,172 | `abo_redesign_qwenmini2_optical_under50m_v2/best_model.pt` | 18,975,128 |
| premium material/studio restyling | `abo_premium_material_2layer_282m_v1/best_model.pt` | 282,433,564 | `abo_premium8_qwenmini2_optical_under50m_v2/best_model.pt` | 18,975,128 |

Every path is relative to `/DATA/DATA1/guest3/t12_assets/runs/`.
The large-model convention excludes the shared 311.16M token embedding, as
agreed, and does not use the Qwen vision tower or language-model head.

The backpack morphology experiment is not part of this final bundle. The
product-form model shown below is the earlier lamp/table version selected by
the user.

## Final parameter and quality summary

The matched electronic baseline is
`abo_scene_replace_28layer_electronic_baseline_v1/best_model.pt`: 28 Qwen
language blocks plus the full electronic BK-SDM decoder. Its counted size is
1,824,200,950 parameters from the first language block through RGB output.
The shared 311,164,928-parameter token embedding is excluded from baseline,
large, and small counts alike.

| Weight | Resolution | Counted parameters | Test quality |
|---|---:|---:|---|
| baseline Qwen-28 + electronic decoder | 256 | 1,824,200,950 | background latent MSE 0.18630; background L1 0.28932; copy improvement 69.90%; exact combination accuracy 50% |
| large background | 256 | 282,431,515 | latent MSE 0.09948; latent L1 0.20683; exact combination accuracy 100% |
| large product-form redesign | 256 | 282,417,172 | latent MSE 0.02496; latent L1 0.10890; copy improvement 96.94%; target accuracy 100% |
| large premium material/studio | 256 | 282,433,564 | latent MSE 0.06713; latent L1 0.15208; copy improvement 94.01%; target accuracy 100% |
| small background + Qwen-mini-2 | 128 | 18,975,128 | pixel MSE 0.001748; L1 0.02671; edge L1 0.06712; copy improvement 99.03% |
| small product-form redesign + Qwen-mini-2 | 128 | 18,975,128 | pixel MSE 0.001961; L1 0.02572; edge L1 0.07203; copy improvement 99.07% |
| small premium material/studio + Qwen-mini-2 | 128 | 18,975,128 | pixel MSE 0.002871; L1 0.02987; edge L1 0.04303; copy improvement 99.48% |

The large models reduce the counted parameter total by about 84.52% relative
to the matched baseline. The small models reduce it by 98.96%. Large metrics
are measured in frozen-VAE latent space; small metrics are measured in RGB
pixel space, so MSE values must not be compared directly across the two model
sizes.

## Matched A100 inference latency

All seven models were measured on the same NVIDIA A100-PCIE-40GB, batch size
1, with 20 warmups and 100 CUDA-event repetitions. The boundary is the input
of the first retained language Transformer block through the generated RGB
tensor. Model loading, tokenization, token-embedding lookup, and host-to-device
copy are excluded. For optical models, software FFT simulation is bypassed and
the requested physical latency `1.0447 * 6 = 6.2682 ms` is added. The parallel
parameter-free/electronic residual is not added a second time.

| Weight | Measured non-optical mean | + physical optics | Final mean | Speedup vs baseline | Latency reduction |
|---|---:|---:|---:|---:|---:|
| baseline Qwen-28 + electronic decoder | 63.738 ms | 0 | 63.738 ms | 1.00x | 0% |
| large background | 31.917 ms | 6.268 ms | 38.185 ms | 1.669x | 40.09% |
| large product-form redesign | 32.887 ms | 6.268 ms | 39.155 ms | 1.628x | 38.57% |
| large premium material/studio | 32.063 ms | 6.268 ms | 38.331 ms | 1.663x | 39.86% |
| small background + Qwen-mini-2 | 6.774 ms | 6.268 ms | 13.042 ms | 4.887x | 79.54% |
| small product-form redesign + Qwen-mini-2 | 6.744 ms | 6.268 ms | 13.013 ms | 4.898x | 79.58% |
| small premium material/studio + Qwen-mini-2 | 6.705 ms | 6.268 ms | 12.973 ms | 4.913x | 79.65% |

Raw p50/p95/min/max timing values are stored in
`latency_a100_matched.json`.

## Large architecture

The large models use the following shared flow:

`RGB -> frozen VAE encoder -> narrow one-pass conditional UNet -> frozen VAE decoder -> RGB`

Text uses the first two Qwen language blocks and the existing condition
adapter. The UNet widths are `[192, 384, 768]`. At the decoder entrance, the
parameter-free electronic residual and optical expert/global branch consume
the same tensor in parallel and are RMS-fused. Learned optical alpha remains
about `0.50`, above the required `0.40` floor.

## New premium task

This task is separate from the four-fixed-product redesign checkpoint. It
preserves the source product identity and alters material plus the complete
editorial studio treatment. It uses only CC BY 4.0 ABO-derived images and no
chairs:

- 576 lamp and 576 table training views from ABO CleanRender;
- 20 backpack training identities from the cleaned ABO backpack subset;
- 4 category-appropriate controlled material families per product;
- 4,688 train, 524 validation, and 524 test pairs.

The materials are category-specific versions of heritage/brass-walnut-leather,
matte obsidian, ivory ceramic/travertine/canvas, and smoked teal
glass/lacquer/textile. Masks are used only when constructing paired targets;
the inference model receives RGB and text only and generates the complete RGB
frame.

Large premium held-out results: latent MSE `0.06713`, latent L1 `0.15208`,
material-target accuracy `1.0`, and `94.01%` improvement over copying the input.

![Large premium material model](large_premium_material.jpg)

Columns are `input | paired target | fully generated output`.

## Small architecture

The corrected 128x128 student keeps the requested Qwen architecture:

`Qwen tokenizer + shared frozen token embedding -> 2048-to-768 projection ->`
`two width-distilled Qwen-style causal Transformer blocks -> conditional`
`direct-RGB encoder/decoder -> RGB`

The two retained blocks use Qwen's pre-RMSNorm, rotary causal self-attention,
and gated SwiGLU topology.  They are distilled against the pooled output of
the corresponding two full-width Qwen blocks.  The generator consumes the
input RGB, the distilled language condition, and a seeded Gaussian texture
input.  At its 16x16 decoder bottleneck, the electronic conditioned residual
and compact Fourier optical MoE operate in parallel and are scale-matched
before fusion.  Every output pixel is decoded by the network; there is no hard
foreground/background compositing.

Counted task parameters are 15,855,520 for Qwen-mini, 125,956 for the optical
path/fusion, and 2,993,652 for the remaining decoder, totalling 18,975,128.
No catalogue, category, scene, or material ID enters the generator at
inference.  The shared 311,164,928-parameter Qwen token embedding is excluded
under the same convention used for the large model; the Qwen vision tower and
LM head are absent.

| Small task | Best epoch | Test MSE | Test L1 | Improvement over copying | Optical alpha |
|---|---:|---:|---:|---:|---:|
| background | 20 | 0.001748 | 0.02671 | 99.03% | 0.4954 |
| redesign | 20 | 0.001961 | 0.02572 | 99.07% | 0.5014 |
| premium material, eight styles | 14 | 0.002871 | 0.02987 | 99.48% | 0.4994 |

Large latent-space and small pixel-space MSE values are not numerically
comparable. The images below are the appropriate qualitative comparison.

![Qwen-mini background student](qwenmini_background_v2.jpg)

![Qwen-mini form-redesign student](qwenmini_redesign_v2.jpg)

![Qwen-mini premium-material student](qwenmini_premium8_v2.jpg)

An arbitrary-prompt smoke test used the user's previously supplied lamp image
and the unseen wording "Keep this exact lamp, but place it in a dim cool-toned
modern study with side window light."  It used no mask or control ID:

![User lamp open-prompt result](user_lamp_cool_study.png)

Changing only the seed currently has negligible visual effect.  These models
should therefore be described as deterministic controllable image editors,
not as diverse samplers.

The no-Qwen checkpoints `abo_background_small_rgb_3m_v1`,
`abo_redesign_small_rgb_3m_v1`, and `abo_premium_small_rgb_3m_v3` remain useful
as architectural ablations only.

## Runtime and resource hygiene

The three corrected small students were trained concurrently and all training
processes exited after 268--490 seconds. Their GPUs were returned to idle.
