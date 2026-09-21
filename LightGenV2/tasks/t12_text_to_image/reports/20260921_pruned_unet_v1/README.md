# Structurally pruned one-step UNet V1

This experiment reduces the electronic UNet before any further optical
replacement. It keeps one UNet call and one VAE decode; there is no diffusion
sampling loop.

## What was removed

Parameter auditing showed that the first (deepest) up block alone contains
175.72M parameters. Its two `Transformer2DModel` modules contain 35.42M
parameters each. V1 replaces the second of those two modules with a true
parameter-free identity path:

```text
16x16 deepest feature
  -> first deep-up ResNet + Transformer
  -> second deep-up ResNet + [removed Transformer -> identity]
  -> remaining up blocks -> latent -> VAE decoder
```

The removed weights are not masks or zeros stored inside the checkpoint. The
runtime module is physically absent, while all tensor shapes, skip connections,
Qwen cross-attention in the other blocks, and the one-step generation contract
remain unchanged.

## Screening

Whole-module bypasses and lower-rank replacements were tested before training.
For the same 35.42M reduction, removing the second deep-up Transformer caused
less initial damage than removing the first deep-up Transformer or the deepest
down Transformer:

| location | untrained recovery nMSE |
|---|---:|
| deep-up second (selected) | **0.45538** |
| deep-up first | 0.48616 |
| deep-down | 0.51123 |

More aggressive removal was rejected: removing all deep-up attention reduced
the UNet to 255.99M but raised nMSE to 0.61340 before recovery. Removing
higher-resolution up attention caused numerical collapse. SVD rank-512 retained
313.46M parameters yet still produced 0.45425 nMSE, so it offered worse
parameter/quality tradeoff than the selected full bypass.

## Recovery training

- Frozen condition path: the existing Qwen adapter.
- Student supervision: 3,872 cached Qwen-conditioned one-step latent samples.
- Validation: all 400 cached generations.
- Epoch 0 after pruning: nMSE 0.45537, cosine 0.73951.
- Epoch 1: nMSE 0.39181.
- Epoch 2: nMSE 0.38772.
- Epoch 3 (selected): **nMSE 0.38706, cosine 0.78583**.
- Epoch 4: nMSE 0.39206; rejected as validation overfit.

The 326.83M quality baseline remains better at nMSE 0.36432 and cosine 0.79956.
Decoded examples show that chair and lamp are largely retained, while shoe can
gain a secondary artifact and table geometry becomes less stable. Therefore V1
is an aggressive size candidate, not a replacement for the quality baseline.

## Size

- UNet: 326,825,604 -> **291,409,284** parameters (-10.84%).
- Relative to the original SD-Turbo UNet: -66.35%.
- Generation tail (adapter + UNet + VAE decoder):
  380,589,063 -> **345,172,743** parameters (-9.31%).
- The frozen Qwen text encoder is reported separately.

Checkpoint:

```text
/DATA/DATA1/guest3/t12_assets/runs/22b5b64c/
  qwen_bksdm_v2_tiny_pruned_up1_seed42/best_unet
```

The checkpoint contains `manifest.json` and `model.fp16.safetensors` and is
loaded by `pruned_turbo.load_pruned_unet`. The end-to-end CLI is
`pruned_turbo_infer.py`.

No inference latency was measured. All GPU jobs exited after use; the selected
RTX 4090 returned to a 12 MiB idle allocation.

