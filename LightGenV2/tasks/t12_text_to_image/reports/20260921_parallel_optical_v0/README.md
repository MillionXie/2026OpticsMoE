# Parallel optical Turbo V0 training result

This run trains the first optical branch without shrinking the electronic
baseline. It is a software optics experiment, not a claim of measured optical
hardware performance.

## Architecture actually trained

The BK-SDM-v2-Tiny checkpoint has no electronic `mid_block`. Its deepest
activation is `[B, 1280, 16, 16]`, so the insertion is:

```text
deepest 16x16 activation
  |-- electronic branch: identity residual (the checkpoint's original path)
  `-- optical branch: pool 16x16 -> 8x8 -> condition projection
        -> top-2 of 4 compact Fourier experts -> global optical block
        -> project to 1280 channels -> upsample 8x8 -> 16x16
                     |
        scale-matched gated fusion (learned alpha)
                     |
              original up path -> VAE decoder
```

The electronic and optical branches receive the same input and are fused only
at their outputs. No existing electronic layer was removed for V0. The optical
branch has 1,146,597 trainable parameters and uses the
`compact_fft_simulation` backend with 64 valid spatial tokens.

## Accepted pilot

- Frozen base: native-condition BK-SDM-v2-Tiny electronic checkpoint.
- Data: ABO CC BY 4.0, 800/100/100 real train/validation/test images across
  chair, lamp, shoe, and table. Cached teacher supervision expands the training
  set to 10,016 one-step latent samples; it does not represent extra real images.
- Training: three epochs on one RTX 4090; 342.3 seconds.
- Validation nMSE: 0.3165676 -> **0.3149963** (0.50% relative reduction).
- Validation cosine: 0.8276184 -> **0.8286197**.
- Fusion alpha: 0.0500 -> **0.05824**, so training did not suppress the optical
  branch to its lower bound.
- Real Qwen-condition evaluation on all 400 cached validation generations:
  nMSE 0.3772729 -> **0.3756327**, cosine 0.7909916 -> **0.7920912**.

Checkpoint:

```text
/DATA/DATA1/guest3/t12_assets/runs/99142c6b/
  qwen_bksdm_v2_tiny_parallel_optical_v0_seed42/best_optical_mid.pt
```

## Strict comparison with the best electronic checkpoint

A second run used the current Qwen-detail electronic checkpoint as the frozen
base and trained only the same 1.15M optical parameters on 3,872 Qwen-aligned
latent samples. It did **not** pass the non-regression gate:

- electronic start: nMSE 0.3643549, cosine 0.7995269;
- best optical epoch by nMSE: 0.3643576, cosine 0.7995422;
- the nMSE regression is 0.0000027 (about 0.00075%).

Transplanting the accepted native optical state onto the Qwen-detail base was
also negative (0.3643157 -> 0.3650345 nMSE). Therefore neither of these two
Qwen-detail combinations is promoted as the model candidate. They are retained
as evidence that the optical branch needs joint-condition calibration before
electronic blocks are removed.

## Decision

V0 proves that the parallel optical block is trainable and gives a small,
repeatable gain on the base it was trained with. It does not yet beat the best
Qwen-detail electronic baseline. The next experiment should keep the topology
fixed and improve optical/Qwen alignment; electronic reduction should begin
only after that strict comparison passes.

No latency number is reported here, by request. The training and evaluation GPU
was released after every run (12 MiB idle allocation reported by `nvidia-smi`).

