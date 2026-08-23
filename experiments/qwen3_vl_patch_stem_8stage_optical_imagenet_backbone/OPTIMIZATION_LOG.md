# P08 optimization and run log

## 2026-08-23: architecture decision

User requirement: retain only the frozen Qwen3-VL Patch/Position Stem, process
static images, never execute an electronic Transformer after tokenization, do
not cache ImageNet hidden states, keep eight optical stages and use an
aggressive enough phase learning rate to create measurable phase motion.

Implemented decisions:

1. Created an independent P08 experiment rather than modifying the completed
   P07 code or checkpoints.
2. Read only `model.visual.patch_embed.proj.weight`, its bias and
   `model.visual.pos_embed.weight` from the Qwen safetensors file.
3. Collapsed the repeated-frame temporal kernel into a static Conv2D by summing
   its two temporal slices. This removes the need to load the 2B model.
4. Fixed the deployed input at 224x224, producing 196 tokens. This permits a
   one-time 14x14 position-table extraction and removes variable-length token
   packing from the hot path.
5. Used one 1024->224 adapter and parameter-free replication into three latent
   optical banks. The banks are not RGB channels.
6. Retained the P07 low-resolution electronic residual, disabled long skips,
   and locked every optical fusion gate to a lower bound of 0.5.
7. Reduced the classifier hidden width to 448 so optical phases remain at least
   half of all newly trainable parameters.
8. Set the main phase learning rate to `4e-3`, zero phase weight decay and
   separate phase/electronic gradient clipping. Every epoch records circular
   physical phase motion, not only raw-parameter drift.
9. Added a one-batch gradient audit, parameter accounting, resume-safe
   checkpoints, final optical-off/random-phase/electronic-skip-off evaluations,
   and explicit flags recording that neither the full Qwen nor a token cache was
   used.

## Run sequence

The run sequence is extraction -> smoke -> 100k supervised screen -> 90-epoch
full ImageNet pretraining. Server commands and any subsequent results are kept
in this file so changes can be replayed without relying on shell history.

## 2026-08-23: extraction, equivalence and smoke results

- Extracted checkpoint: 988,160 frozen parameters, 3,955,552 bytes.
- Static equivalence input: deterministic 224x224 RGB image.
- Official Qwen processor output: 196 patches of width 1536, grid `[1,14,14]`.
- Official Conv3D+position versus extracted Conv2D+position maximum absolute
  error: `1.9073486328125e-6`.
- Mean absolute error: `9.361252750750282e-8`; RMS error:
  `1.5390361340905656e-7`; equivalence passed at `atol=rtol=1e-4`.
- Unit tests: 3 passed.
- GPU smoke: two exact-BP optimizer steps completed, all eight phase-gradient
  norms finite and nonzero. Norms ranged from 0.0954 to 0.8373 before clipping.
- Training-state mean circular physical phase motion after two steps:
  `0.006729 rad`. This confirms the `4e-3` phase schedule can move the masks.
- Exact accounting: 1,204,224 phase parameters; 1,194,587 trainable electronic
  parameters; optical fraction of newly trainable parameters: 50.2009%.
- Full Qwen loaded during training: no. Hidden-state cache used: no.

The final training runner separately exports `checkpoints/backbone.pt` without
the ImageNet readout. Its stable contract is the final three-bank optical field
plus the tuple of all eight OEO stage maps.

## 2026-08-23: 100k screen result

The single-GPU screen used 100 images per class for training, 10 per class for
validation and five supervised epochs. Validation Top-1 increased monotonically:

| epoch | train Top-1 | validation Top-1 | mean phase motion |
|---:|---:|---:|---:|
| 1 | 0.13% | 0.42% | 0.0643 rad |
| 2 | 0.52% | 1.60% | 0.1960 rad |
| 3 | 1.19% | 2.15% | 0.2762 rad |
| 4 | 1.82% | 2.90% | 0.2982 rad |
| 5 | 2.24% | **3.96%** | **0.3023 rad** |

Final validation Top-5 was 11.59%. Causal evaluations on the same 10,000-image
validation subset were:

- normal: 3.96% Top-1;
- optical off: 0.33% Top-1;
- random phase: 0.21% Top-1;
- electronic residual off: 2.31% Top-1.

At the selected checkpoint, 74.02% of phase pixels had moved by more than 0.1
rad. The screen therefore passed both convergence and optical-dependence checks.

## 2026-08-23: full ImageNet launch

- configuration: `configs/pretrain_90e.yaml`;
- epochs: 90, full 1,281,167-image training split and 50,000-image validation;
- GPUs: physical 3 (RTX 4090) and 5 (RTX 3090), two-rank DDP;
- per-rank batch: 28; global batch: 56;
- launch shell PID: 1122715;
- log: `logs/p08_imagenet1k_pretrain_90e.log`;
- run: `runs/p08_imagenet1k_pretrain_90e`;
- initial validation: 0.10% Top-1 / 0.45% Top-5;
- observed memory after training began: approximately 3.0GB / 2.8GB;
- resume: command `04_train_imagenet_90e.sh` always passes `--resume`.

The job was confirmed beyond process creation: both CUDA ranks were resident,
both GPUs showed active utilization, and batch 250/22,878 of epoch 1 had been
logged before handoff.

## 2026-08-23: batch-size calibration and optimized restart

The initial per-rank batch 28 was a conservative stability setting rather than
a throughput optimum. Five candidates were measured on physical GPU 1 (RTX
4090) for 300 complete training updates, including the online frozen stem,
eight optical stages, exact backpropagation, AMP, augmentation and AdamW.

| per-rank batch | images/s | peak allocated MiB | peak reserved MiB |
|---:|---:|---:|---:|
| 64 | 518.172 | 5127.4 | 5214.0 |
| **96** | **550.457** | 7652.6 | 7772.0 |
| 128 | 525.451 | 10177.7 | 10294.0 |
| 160 | 537.712 | 12702.8 | 12836.0 |
| 192 | 538.924 | 15228.0 | 15372.0 |

Memory was not the limiting resource: throughput peaked at 96 and regressed
for every larger candidate. The actual heterogeneous physical GPU 3 (RTX 4090)
+ GPU 5 (RTX 3090) pair was then measured for 300 DDP updates:

| per-rank batch | global batch | global images/s | max peak allocated MiB |
|---:|---:|---:|---:|
| 64 | 128 | 845.153 | 5136.6 |
| **96** | **192** | **884.936** | 7661.7 |

The original global-batch-56 run processed 757.6, 764.9 and 770.6 images/s in
epochs 1--3. Its validation Top-1 reached 6.83%, 12.18% and **14.33%**, with
0.7919 rad mean phase motion after epoch 3. That checkpoint and history remain
under `runs/p08_imagenet1k_pretrain_90e`; the incomplete fourth epoch was
stopped deliberately after the calibration established a faster setting.

Selected formal configuration: per-rank batch 96, global batch 192, validation
batch 192 and 90 epochs. Because the global batch increased by 3.43x, learning
rates use a conservative approximately square-root scaling instead of the risky
linear jump: phase `7e-3`, adapter `5e-4`, residual `3.5e-4`, head `9e-4`, with
one full warmup epoch and the existing gradient clipping. The phase LR remains
large enough to produce visible physical mask motion.

Reproducible launcher:

```bash
PHYSICAL_GPU_INDICES=3,5 bash experiments/qwen3_vl_patch_stem_8stage_optical_imagenet_backbone/commands/08_launch_imagenet_90e_bs96.sh
```

New run directory: `runs/p08_imagenet1k_pretrain_bs96_90e`. New log:
`logs/p08_imagenet1k_pretrain_bs96_90e.log`. The expected training-only epoch
time from the DDP benchmark is about 24.1 minutes, versus 27.7--28.2 minutes for
the old run.
