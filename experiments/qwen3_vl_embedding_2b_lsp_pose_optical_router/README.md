# LSP Vision-only optical Router ablation

This project checks whether the Caltech101 Router result transfers to a dense
human-pose task. It is a new experiment; it does **not** modify or silently
reinterpret `qwen3_vl_embedding_2b_lsp_pose_optical_moe16`.

## Model contract

Only the Vision path is executed:

```text
224x224 person crop
  -> frozen Qwen3-VL patch embedding
  -> packed Vision tokens [sum(T),1024]
  -> image_grid_thw restores the 2-D token topology
  -> input adapter 1024 -> 192
  -> Block 1: 2-D electronic Mixer || MoE4 optical expert plane
  -> learned bounded fusion (optical fraction >= 5%)
  -> Block 2: 2-D electronic Mixer || global optical plane
  -> learned bounded fusion (optical fraction >= 5%)
  -> spatial 192-D feature map
  -> progressive 14-joint heatmap decoder, 56x56
```

The optical body is instantiated directly from
`qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust`.
It is not the older 16-um `vision2_hybrid_dense` core. The fixed physical
contract is 17 um, 10 cm, 518x518 simulation canvas, canonical 478x478 CCD,
224x224 expert tile, 2x2 MoE4, 16-pixel input/phase/CCD displacement bounds,
and a 192-wide 2-D depthwise Mixer.

Position handling needs one precise qualification. Qwen still constructs the
packed token order and `image_grid_thw`, but all native Vision attention blocks
are replaced. The rotary-position kwargs that Qwen would pass into native
attention are accepted and ignored by the capture/bypass block; they are not
an extra positional neural layer in this student. Spatial position is retained
mainly by reconstructing the Qwen block-major token order into its 2-D grid and
then applying the 2-D Mixer/pose decoder on that topology.

Electronic E1/E2/E4 and optical O2 all use `power_l2`: the selected amplitude
weights have unit L2 norm, so changing `k` does not silently change routed
optical power. The corrected straight-through expression is
`hard.detach() + dense - dense.detach()`.

The optical Router adds one time-multiplexed exposure before the two Vision
feature exposures:

```text
central 224x224 amplitude + learned Router phase
  -> 10 cm propagation
  -> same canonical 478x478 CCD
  -> four fixed 59x59 energy windows
  -> standardize four energies -> softmax -> hard Top-2
```

Softmax and Top-k remain electronic. No additional optical path or CCD crop is
introduced. The Vision-global plane reuses the decision from Vision-expert.

## Data and periodic-test checkpoint protocol

The original dataset loader provides HR-LSPET 9,428 + the first 1,000 LSP
images as training, and the last 1,000 LSP images as test. The lab protocol
keeps this split unchanged:

- all 9,428 HR-LSPET + first 1,000 LSP: training (10,428 total);
- no validation split;
- final 1,000 LSP: periodic test and checkpoint-selection split.

EMA weights are tested at epoch 1, every 5 epochs, and the final epoch. The
checkpoint is selected directly by maximum test PCK@0.2-torso, then lower
test torso NME, lower test loss, and finally the earlier epoch. Intermediate
epochs do not run test. The optional explicit `evaluate` command reloads the
selected checkpoint and saves detailed per-sample outputs.

This protocol deliberately uses test for model selection, as requested for
the current laboratory comparison. Its reported test score is therefore a
best-observed/model-selection result rather than a leakage-free estimate of
future generalization.

All four variants load one shared, untrained trainable-body/pose-head
initialization. Router state is deliberately absent from the shared
initialization.

## Important limitations

- This is currently one optimization/data seed (42). It is a transfer check,
  not evidence that variance has been eliminated. Repeat seeds should follow
  only after this first matrix is complete.
- Person cropping is computed from ground-truth joints, following the existing
  LSP experiment. Therefore the result evaluates pose estimation within a
  supplied person crop; it is not an end-to-end person detection benchmark.
- Test is visible every five epochs and selects the checkpoint. Quote this
  protocol next to the result; do not describe it as a sealed-test score.

See `RUN_COMMANDS.md` for the exact order.
