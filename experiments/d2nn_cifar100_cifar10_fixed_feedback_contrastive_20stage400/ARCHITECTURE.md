# Architecture and loss definitions

## Optical backbone

All 20 stages use the same data flow but independent phase masks and residual
weights:

```text
A_s
→ exp(i phase_s)
→ angular-spectrum propagation (5 cm)
→ |E|²
→ LayerNorm over 400 × 400, no affine parameters
→ ReLU
→ w_optical,s × activated + w_skip,s × A_s
→ A_{s+1}
```

`[w_optical,s, w_skip,s] = softmax(residual_logits_s)`. Both logits use ordinary
BP for every method. They start at `[0.35,0.65]` and are logged every epoch.

## Embedding losses

For normalized embeddings, SupCon pulls all same-class views together and uses
every other embedding in the batch as a denominator negative. Fine-tuning adds
a leave-one-out prototype loss. An anchor is removed from its own class mean so
it cannot classify itself trivially.

```text
pretrain_loss = 1.0 × SupCon
finetune_loss = 1.0 × SupCon + 0.5 × prototype CE
```

Both temperatures default to 0.1. Batch samplers guarantee multiple images and
two views per class.

## Fixed feedback

Forward propagation always uses current phase masks. BP sends the exact current
input gradient through each optical stage. FA-pretrained replaces only that
input-feedback connector with the phase operator saved at the end of CIFAR-100
pretraining. FA-random uses a fixed random phase operator. Local current-phase
gradients, residual weights, embedding LayerNorm and projection retain their
ordinary local BP rules.

The current batch loss and current output error are recomputed on every step;
no sample-specific backward field is cached.
