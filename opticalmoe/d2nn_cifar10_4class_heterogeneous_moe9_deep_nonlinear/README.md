# Deep heterogeneous MoE with staged OEO nonlinearities

This independent CIFAR-10 4/10-class experiment does not modify the running
`deep_linear` experiment, the homogeneous MoE, or shared modules.

The ten-class config uses ten `50 x 50` detector squares, the same size as the
four-class experiment. They are arranged as centered `3 / 4 / 3` rows with a
50-pixel clear gap horizontally and vertically. This replaces the older,
smaller crowded ten-class layout.

## Data flow

```text
grayscale CIFAR image
-> input-dependent top-k router and optical prompt
-> 3x3 heterogeneous expert bank
-> five synchronized expert stages, each followed selectively by OEO
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
mu_i = mean(I_i over one expert's H x W pixels)
sigma_i = sqrt(mean((I_i - mu_i)^2) + eps)
I_hat_i = (I_i - mu_i) / sigma_i
A_next_i = ReLU(gamma_{stage,i} * I_hat_i + beta_{stage,i})
E_next = A_next + 0j
```

By default, `mu_i` and `sigma_i` are runtime statistics computed independently
for each sample and each enabled expert. This avoids suppressing D2NN, Fourier,
or Fiber experts merely because their physical output-power scales differ.
They are not trainable. Set `normalization.per_expert_enabled: false` to run the
legacy ablation that uses one shared mean/std over all enabled experts in a
stage.

The affine maps `gamma_{stage,expert}` and `beta_{stage,expert}` are trainable
and default to one and zero. Every expert at every stage owns an independent
120 x 120 pair. This adds `5 x 9 x 2 x 120 x 120 = 1,296,000` parameters and
allows D2NN, Fourier, and Fiber experts to learn different post-normalization
operating ranges. Set `affine_sharing: per_stage` for the lower-parameter
144,000-parameter shared-affine ablation, or `elementwise_affine: false` to
disable affine parameters completely.

ReLU output is loaded directly as the zero-phase amplitude of the next optical
stage. It is deliberately **not multiplied by the sample's routing amplitude
again** after normalization. Routing still controls which experts and fields
enter the expert bank, while the per-expert LayerNorm gives each selected
expert a comparable nonlinear operating range. Fiber Stage2 bypasses OEO and
therefore preserves its full complex field for coherent mode projection.

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
its default weight is zero. Router importance loss is also zero by default.
The router balance weight is `0.2`: stronger than the earlier `0.1`, but still
small relative to the classification objective, so it discourages expert
collapse without forcing uniform routing at the expense of the task.

The ten-class config retains `train_samples_per_class_per_epoch: 1000`. CIFAR-10
has 5,000 training images per class, and the rotating per-class sampler uses a
different non-overlapping slice each epoch, so all images are covered in five
epochs. Mini-batches are shuffled across classes. This limits epoch time
without permanently reducing the training set.

Both default configs use new `perexpert_affine...balance02` run names, so these
runs do not overwrite checkpoints or metrics from the earlier stage-global
normalization experiments.

## Outputs

`metrics/stage_nonlinearity_epoch_XXXX.json` records every stage's pre-OEO,
normalized, and output powers; active-pixel ratio; normalization input mean and
standard deviation; enable state; and per-type aggregates. Fiber metrics additionally include coupling
efficiency, mode power, effective mode count, and reconstruction power.

Figures save pre-OEO intensity, signed per-expert LayerNorm intensity,
ReLU/re-encoded amplitude, and bypass complex phase. The Fiber Stage2
phase is saved explicitly. Fourier and Fiber exports remain type-specific and
are never presented as ordinary D2NN masks.

The default staged OEO affine maps add 1,296,000 trainable parameters. Architecture
reports keep them explicitly separate from physical masks, router, and global
phase, and record whether statistics are per-expert or stage-global.


See [COMMANDS.md](COMMANDS.md).
