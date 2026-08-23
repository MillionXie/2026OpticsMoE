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
