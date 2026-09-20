# T12 server training record — 2026-09-20

## Frozen protocol

- Dataset: ABO single-object subset, exact categories `SHOES`, `CHAIR`, `LAMP`, `TABLE`.
- Split: 800 train / 100 validation / 100 sealed test, one main image per product.
- Text: cached frozen Qwen3-VL-2B-Instruct hidden states.
- Image codec: frozen `stabilityai/sd-vae-ft-mse`, latent shape `4 x 28 x 28`.
- LightGen graph: the electronic residual and optical branch read the same input at each stage;
  their outputs are fused. There is no electronic-to-optical cascade.
- Inference: one generator forward and one VAE decode. There is no diffusion or autoregressive loop.
- Formal device: A100 40 GB selected by full UUID. Every pilot, training run and diagnostic
  returned to 14 MiB after its process exited.

## Runs

| Row | Commit | Epochs | Selected epoch | Validation loss | Decision |
|---|---:|---:|---:|---:|---|
| LightGen latent CVAE | `a4fe54e9` | 80 | 77 | 0.383153 | Keep as stage-1 reference |
| Qwen + VAE electronic baseline | `a4fe54e9` | 80 | 79 | 0.366753 | Keep as paired baseline |
| Unconstrained LightGen GAN | `35212645` | 25 | 23 | 0.393702 | Reject |
| Decoder-only constrained GAN | `5608cd10` | 15 | 4 | **0.379784** | Recommended sharpening checkpoint |

The unconstrained GAN made posterior reconstructions somewhat sharper but its random-prior samples
collapsed into a class-independent high-contrast pattern. The cause was that the random prior only
saw an unconditional discriminator and could drift away from the stage-1 conditional generator.
Training the matching baseline GAN was therefore intentionally skipped.

The corrected profile freezes the warm-started posterior, LightGen parallel backbone and old latent
head. Only four newly inserted decoder residual blocks are optimized (2,658,820 parameters), and a
latent-delta loss anchors random-prior output to the stage-1 generator. It preserved the four class
structures, avoided collapse, reduced validation loss, and modestly sharpened posterior edges. The
prior images remain blurry, so this is a viable research baseline rather than a claim of high-quality
or SOTA generation.

## Server artifacts

```text
/DATA/DATA1/guest3/t12_assets/runs/a4fe54e9/lightgen_parallel_seed42/
/DATA/DATA1/guest3/t12_assets/runs/a4fe54e9/qwen_vae_baseline_seed42/
/DATA/DATA1/guest3/t12_assets/runs/35212645/lightgen_parallel_gan_seed42/
/DATA/DATA1/guest3/t12_assets/runs/5608cd10/lightgen_parallel_decoder_gan_seed42/
/DATA/DATA1/guest3/t12_assets/runs/5608cd10/diagnostic/
```

The fixed-test FID/KID/CLIP/LPIPS suite remains a separate final evaluation step. Its evaluator
weights were not present in the server cache during this run; validation losses and diagnostic grids
must not be presented as substitutes for those test metrics.
