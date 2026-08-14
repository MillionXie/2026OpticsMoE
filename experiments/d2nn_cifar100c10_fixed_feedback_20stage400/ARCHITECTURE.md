# Architecture and feedback rule

## Forward path

The formal model contains twenty independent 400 x 400 phase-only planes. Every
stage is a complete OEO stage:

```text
real nonnegative amplitude
-> current phase-only mask
-> unpadded 5 cm angular-spectrum propagation
-> 400 x 400 square-law CCD
-> non-affine LayerNorm over the complete CCD
-> ReLU
-> learned positive residual mixture with the preceding amplitude
-> zero-phase amplitude reload
```

The two residual coefficients are independent trainable logits followed by a
two-element softmax. They start at optical=0.9 and skip=0.1, remain positive and
sum to one. They always use ordinary backpropagation. After stage 20, fixed
20 x 20 average pooling feeds a small `Linear(400,128) -> GELU -> Linear(128,100)`
electronic readout.

There is no padding, k-space constraint, phase dropout, DC penalty, phase
smoothness loss, MoE router, or sample-wise power renormalization beyond the
specified post-CCD LayerNorm.

## Feedback alignment

For optical stage `l`, let

```text
A_l(phi) = P_5cm diag(exp(i phi))
```

All methods use `A_l(current_phi)` in the forward pass. They differ only in the
connector that returns the current batch's error signal to the preceding stage:

```text
BP:             A_l(current_phi)^H
FA-pretrained:  A_l(pretrained_phi)^H
FA-random:      diag(exp(-i random_phi_l)) P_5cm^H
```

The local phase gradient continues to use the current phase and current input.
CCD, LayerNorm, ReLU, residual mixing, and the electronic readout retain their
ordinary local derivatives. No sample-specific backward field is cached.

FA-random uses a fixed unit-modulus phase screen per stage. This gives the same
shape, propagation, and norm scale as FA-pretrained without allocating an
impossible dense 160000 x 160000 feedback matrix.

## Dataset protocol

BP pretraining uses clean CIFAR-100 and the full 100-way readout. The downstream
task uses ten fixed CIFAR-100 classes from CIFAR-100-C. The 100-way readout is
retained, and the loss selects the same ten output rows; therefore the task head
does not change and the no-fine-tuning baseline is well-defined.

The ten classes are apple, bicycle, bus, butterfly, clock, dolphin, forest,
lion, rocket, and telephone. Base test-image indices are split 60/20/20 per
class before corruption views are expanded, so one base image cannot cross the
train/validation/test boundary.

## Controlled comparison

All fine-tuning methods load the same pretraining checkpoint, initialize a new
AdamW optimizer with identical parameter groups, and use deterministic epoch
orders and sample-level augmentation seeds. Weight decay, gradient clipping,
AMP, phase dropout and schedulers are disabled. The comparison command refuses
to aggregate runs whose saved batch-order hashes differ from BP.
