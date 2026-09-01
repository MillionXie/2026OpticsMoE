# P09 architecture and comparison lock

## Data path

1. A frozen, extracted Qwen3-VL Conv2D patch embedding and position embedding
   produces 196 block-major tokens of width 1024. No Qwen Transformer is loaded.
2. One trainable LayerNorm/Linear/Softplus adapter maps 1024 to 224 and packs the
   tokens into three identical 224x224 latent optical banks.
3. Eight independent OEO stages run phase modulation, angular-spectrum
   propagation, square-law detection, normalization and optical/electronic
   fusion. Every fusion gate is constrained to keep the optical branch at or
   above 0.5.
4. A temporary ImageNet classification head fuses the three banks and pools the
   first 196 token rows. It is excluded from the reusable backbone export.

## Per-stage electronic bypass

For each bank, the first 196 rows are interpreted as 196 tokens of width 224.
The three banks are folded into the batch dimension, so one set of per-stage
weights processes all banks:

```text
224 -> LayerNorm -> Linear(224, 96)
    -> h + sigmoid(g_spatial) * Pointwise(GELU(DWConv3x3(Grid(LN(h)))))
    -> h + sigmoid(g_channel) * MLP_96_192_96(LN(h))
    -> LayerNorm -> Linear(96, 224) -> bounded output residual -> ReLU/RMS
```

The grid conversion exactly inverts Qwen's 2x2 block-major token order before
the depthwise convolution and restores it afterwards. The spatial and channel
gates both initialize to 0.10. The final projection is zero-initialized, and a
third bounded output scale initializes to 0.10 with a maximum of 0.25; this
preserves P08's identity-bypass startup while retaining the two internal mixer
gates requested for the architecture.

## Parameter accounting

The deterministic design counts are:

| Component | Trainable parameters |
|---|---:|
| Eight optical phase planes | 1,204,224 |
| 1024-to-224 input adapter | 231,648 |
| Eight width-96 mixer bypasses plus fusion gates | 733,472 |
| Temporary ImageNet head | 650,603 |

The reusable backbone therefore has 2,169,344 trainable parameters and an
optical parameter fraction of 55.51%. If the disposable ImageNet task head is
also counted, the fraction is 42.70%; both numbers are emitted in every run
manifest. The 0.733M residual electronics remain within the laboratory's
1--2M total electronic-residual budget.

The frozen 0.988M Qwen stem is reported separately because it is a fixed input
front end, not a trainable optical/electronic allocation. No claim should hide
that fixed inference compute.

## Controlled P08 comparison

The formal P09 config copies P08's seed, dataset, augmentation, optimizer,
per-parameter-group learning rates, warmup, cosine decay, AMP settings, global
batch and 90-epoch horizon. Both use two ranks with batch 96 per rank. Thus the
only planned causal change is low-resolution CNN bypass versus width-96 token
mixer bypass.
