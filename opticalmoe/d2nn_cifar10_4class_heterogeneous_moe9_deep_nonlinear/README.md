# Deep heterogeneous MoE with staged OEO nonlinearities

This independent CIFAR-10 four-class experiment does not modify the running
`deep_linear` experiment, the homogeneous MoE, or shared modules.

## Data flow

```text
grayscale CIFAR image
-> input-dependent top-k router and optical prompt
-> 3x3 heterogeneous expert bank
-> five synchronized expert stages, each followed selectively by shared OEO
-> reassemble 480x480 canvas
-> existing 450x450 global phase at the Stage5 output plane
-> propagation to the existing final square-law detector
```

There is deliberately no OEO, normalization, activation, or re-encoding after
the global phase. It is followed only by propagation and final detection.
Stage5 already includes the final 5 cm expert propagation, so no duplicate
`expert_to_global_fc` propagation is inserted after reassembly.

## Five expert stages

- D2NN: five independent spatial phase+propagation stages; all use OEO.
- Fourier: three finite-aperture Fourier-convolution+propagation stages, then two spatial phase+propagation stages; all use OEO.
- Fiber: two D2NN encoder stages, coherent Gaussian-mode projection/modulation/reconstruction, and two decoder stages. Its schedule is `[true, false, true, true, true]`.

Fiber Stage2 keeps its full complex output, including phase, for the Gaussian
mode projection in Stage3. No detection, normalization, activation, or
zero-phase re-encoding occurs at that boundary. There is no extra fiber
coupling phase mask: the second encoder mask and propagation are the mode
matcher.

## OEO definition

```text
I = abs(E)^2
mu = mean(I over all OEO-enabled expert pixels)
sigma = sqrt(mean((I - mu)^2) + eps)
I_hat = (I - mu) / sigma
A_next = ReLU(I_hat)
E_next = A_next + 0j
```

`mu` and `sigma` are runtime statistics, not trainable parameters. They are
computed separately for every sample and stage, but shared by all enabled
expert regions inside that sample/stage. They are not shared between stages or
between batch items. Fiber Stage2 bypasses OEO and is excluded from its stage's
statistics.

This OEO has no affine LayerNorm parameters, learned gain, learned threshold,
or amplitude exponent. ReLU output is loaded directly as the zero-phase
amplitude of the next optical stage. The separate per-expert output amplitude
gain is disabled; routing remains the input-dependent amplitude controller.

## Loss

The default classification objective remains the full detector-plane MSE. It
is controlled by:

```yaml
normalize_detector_plane_mse: true
detector_ce_weight: 0.0
```

When normalization is enabled, each predicted detector plane is rescaled so
its total energy equals the target mask's total energy before MSE is evaluated.
This removes the trivial global-power degree of freedom but preserves the
spatial error. Setting it to `false` restores the original raw detector-plane
MSE. The optional detector-region CE path is present for later ablation, but
its default weight is zero. Router importance loss is also zero by default;
the existing balance weight remains unchanged.

## Outputs

`metrics/stage_nonlinearity_epoch_XXXX.json` records every stage's pre-OEO,
normalized, and output powers; active-pixel ratio; normalization input mean and
standard deviation; enable state; and per-type aggregates. Fiber metrics additionally include coupling
efficiency, mode power, effective mode count, and reconstruction power.

Figures save pre-OEO intensity, signed stage-global LayerNorm intensity,
ReLU/re-encoded amplitude, and bypass complex phase. The Fiber Stage2
phase is saved explicitly. Fourier and Fiber exports remain type-specific and
are never presented as ordinary D2NN masks.

The staged OEO adds zero trainable parameters. Architecture reports keep this
explicitly separate from physical masks, router, and global phase.


See [COMMANDS.md](COMMANDS.md).
