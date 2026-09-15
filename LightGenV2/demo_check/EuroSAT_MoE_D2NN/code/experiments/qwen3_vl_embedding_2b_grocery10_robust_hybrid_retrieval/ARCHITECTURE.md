# Architecture and baseline audit

## What the current baseline actually does

The short answer is: **two optical phase planes in vision and two in language,
but DeepStack is used.** “Two optical layers” here means one MoE expert phase
plane followed by one global phase plane, not two Qwen transformer blocks.

```text
image + fixed retrieval instruction
  -> frozen Qwen processor / patch embedding
  -> vision adapter (hidden -> 224)
  -> electronic Top-2 router over a 2x2 / four-expert bank
  -> expert phase plane -> 5 cm propagation -> CCD
  -> per-expert LN/ReLU/OEO reload
  -> global phase plane -> 5 cm propagation -> CCD
  -> 224x224 detector pooling/LN/ReLU
  -> vision adapter (224 -> hidden) and frozen Qwen visual merger
  -> one retained Qwen DeepStack injection at native language index 5
  -> frozen token embedding
  -> language optical replacement at language layer index 1
  -> the same expert/global two-plane optical stack
  -> last valid token's 224 detector values
  -> LayerNorm -> Linear(224,64) -> L2 normalize
  -> cosine-similarity retrieval against Student gallery embeddings
```

The original Qwen model has three native DeepStack visual injection indexes.
The Student retains one (`[5]`) because the optical vision stack exposes one
intermediate tap (`vision_tap_stages=[1]`). Therefore the hypothesis “DeepStack
is not used” is false; it is reduced from three native injections to one.

The released run has about 1.795M trainable parameters: 858,376 phase values,
1,576 router values, about 920k effective adapter values and a 14,848-parameter
retrieval head. The Qwen backbone, final RMSNorm and unused language output
adapter are frozen.

## Baseline objective

For normalized Student embedding `s`, frozen Teacher embedding `t`, labels
`y`, Student gallery prototypes `G_s` and Teacher gallery prototypes `G_t`, the
training objective is:

```text
L = lambda_kd       * mean(1 - cosine(s,t))
  + lambda_rel      * MSE(offdiag(s s^T), offdiag(t t^T))
  + lambda_ret      * supervised_contrastive(s,y)
  + lambda_gallery  * CE(s G_s^T / tau_gallery, y)
  + lambda_tgallery * CE(s G_t^T / tau_gallery, y)
  + lambda_balance  * router_balance
  + lambda_import   * router_importance
  + optional lambda_response * router_response_consistency
  + optional lambda_dc       * coherent_phase_DC
```

The release model config uses weights `5.0, 0.25, 1.0, 0.5, 0.25, 0.05,
0.01`; the actual 31-class pretrain -> Grocery10 continuation used on the
server uses `8.0, 0.1, 1.0, 0.25, 0.1, 0.02, 0.005`. Phase-DC is off. There is
no ten-class classifier: evaluation is Student-query versus Student-gallery
cosine retrieval in 64 dimensions.

## Server evidence (checked 2026-08-15)

| system | Top-1 | Top-3 | MRR | note |
|---|---:|---:|---:|---|
| released simulation, epoch-40 EMA | 67.69% | 87.31% | 0.7916 | canonical result |
| response-preserving OEO, epoch-40 EMA | 69.62% | 90.38% | 0.8092 | fair fixed-epoch comparison |
| response-preserving best observed | 70.00% | 91.92% | 0.8153 | test-selected, diagnostic only |
| physical CCD + causal downstream fine-tune | 33.08% | 62.31% | 0.5248 | test images excluded from adaptation |

The 34.61-point absolute simulation-to-hardware Top-1 drop is large enough that
registration and response mismatch deserve first-class modeling. Adding more
phase capacity alone would make that gap harder to control.

## New robust hybrid path

For encoded electronic input `X`, expert optical response `O1`, global optical
response `O2`, and learned scalar gates `g1,g2`:

```text
Z1 = Refiner1(g1 * RMSNormPositive(X_routed)
              + (1-g1) * RMSNormPositive(O1))

Z2 = Refiner2(g2 * RMSNormPositive(X)
              + (1-g2) * RMSNormPositive(O2))
```

There are separate `g1,g2` values in vision and language. Sigmoid
parameterization constrains them to `[0,1]`; all start at `0.8`, so training
begins with a stable electronic bypass but can increase optical contribution
when it is useful. `Z1` is reloaded as a nonnegative zero-phase field and sent
to the global phase plane. `Z2` supplies both the frozen Qwen boundary and the
retrieval detector feature.

Each refiner is `1x1 expansion -> depthwise 3x3 -> depthwise dilated 3x3 ->
1x1 projection`, with a zero-initialized final projection. It has under 1,000
parameters per fusion and can learn local denoising/re-registration without a
`50176 x hidden` dense matrix.

The embedding head applies:

```text
LN(224) -> [identity + sigmoid(g) * MLP(224->96->224)]
        -> LN(224) -> Linear(224,64) -> L2 normalize
```

## Robustness perturbations

Training independently samples zero-filled integer shifts for:

1. routed amplitude input versus the expert phase plane;
2. each phase mask versus the incident field;
3. CCD coordinates versus the expected electronic crop.

All samples in a mini-batch share a shift, matching a systematic physical
registration state rather than pretending every image moves the hardware.
Zero-fill is important: circular `roll` would introduce unphysical energy from
the opposite edge. Clean evaluation is the default; perturbations remain off
in `eval()` unless explicitly enabled for a robustness sweep.

Block phase bypass dropout (`p=0.05`, block 8) makes local phase-mask regions
occasionally behave as zero phase. The k-space cutoff remains 0.65 degrees.
Together with KD on every stochastic forward, the Student is optimized to land
near the same Teacher embedding under these optical changes.

## Recommended experiment order

1. Train the checked-in configuration and report clean simulation retrieval.
2. Evaluate fixed shift sweeps at 0/4/8/12 logical pixels separately for input,
   phase and CCD, then jointly; report mean, worst case and clean-to-perturbed
   drop rather than selecting the best test epoch.
3. Export and measure calibration/gallery/train hardware captures before the
   held-out test set. Fit only downstream electronics on calibration/train.
4. Ablate: residual only; residual + low phase LR; + phase dropout; + alignment;
   + enhanced head. This identifies which mechanism closes the hardware gap.
