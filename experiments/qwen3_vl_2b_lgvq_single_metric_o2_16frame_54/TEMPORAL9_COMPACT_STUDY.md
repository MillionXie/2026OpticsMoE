# Temporal 9-frame / 3x3 compact optical study

## Conclusion

The formal Temporal input has been reduced from 16 frames arranged as 4x4 to
9 frames arranged as 3x3 without changing the 478x478 logical active field,
532 nm wavelength, 17 um sampling pitch, 10 cm propagation, four optical
experts, optical Top-2 routing, or the no-Attention/no-Transformer student
contract.

Two checkpoints are intentionally retained:

- Maximum-accuracy 9-frame checkpoint: test SRCC 0.817582. Its per-frame
  optical router is well balanced, but the final serial router selects the
  same two experts for all test videos.
- Recommended routing trade-off: test SRCC 0.810740. The per-frame router is
  well balanced and the final serial router uses three experts with aggregate
  selected shares 0.3414 / 0.2043 / 0.0000 / 0.4543. This is only 0.006842
  SRCC below maximum accuracy and avoids the fixed-pair collapse.

The recommendation is the routing trade-off checkpoint unless only the
largest simulated SRCC matters.

## Geometry

All sizes below are logical 17 um pixels inside the same 478x478 active field.
The surrounding simulation canvas remains 518x518.

| Input | Frame grid | Lane size | Lane gap | Expert size | Expert gap | Expert-area use |
|---|---:|---:|---:|---:|---:|---:|
| 4 frames | 2x2 | 232 | 14 | 109 | 14 | 83.20% |
| 9 frames | 3x3 | 156 | 4 | 77 | 2 | 93.42% |
| 16 frames | 4x4 | 114 | 6 | 54 | 6 | 81.68% |

The 9-frame expert is 77x77 logical pixels, or 1.309x1.309 mm. Its 2-pixel
intra-lane gap is 34 um, corresponding to about 4.25 pixels on the 8 um phase
SLM. The 4-pixel inter-lane gap corresponds to about 8.5 phase-SLM pixels.
Thus the tighter packing gives area back to the experts while preserving a
nonzero hardware-separable guard gap.

The nine frames are sampled at normalized video times 0.1, 0.2, ..., 0.9.
There is no random temporal sampling mismatch between cache construction and
formal inference.

## Optical route and anti-collapse design

The vision router performs one optical four-region energy decision per frame,
so each video has nine parallel Top-2 decisions. At the recommended checkpoint
the four aggregate selected shares are 0.2220 / 0.1998 / 0.2778 / 0.3004.

The final serial field contains valid multimodal tokens in its first rows and
zero padding afterwards. With 20% coherent unmodulated leakage, routing the
whole padded 109x109 field places a strong zero-order footprint near one pair
of detector windows. The recommended version therefore:

1. crops the actual valid token rows;
2. uses a fixed adaptive resampling to a centered 45x45 optical router input;
3. keeps the learned 109x109 phase plane and the same 478x478 CCD ROI;
4. places four 55x55 energy windows outside a central guard band;
5. applies softmax and hard Top-2 to those four measured optical energies.

This changes neither the feature-propagation field nor the physical 10 cm
path. No electronic MLP, Attention, or Transformer is used for routing.
Stronger routing penalties, uniform-reference flat-field division, and
non-trainable four-channel standardization were also tested. They did not
improve the joint accuracy/diversity operating point, so they are recorded as
diagnostics rather than selected as the formal result.

## Formal result

There is no validation split. The fixed 558-video test split is evaluated
every five epochs and the best test SRCC checkpoint is selected, following the
project's requested protocol.

| Model / same checkpoint mode | SRCC | KRCC | PLCC | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| Temporal-16 optical | 0.840591 | 0.633062 | 0.859658 | 7.2170 | 5.6309 |
| Temporal-9 maximum accuracy | 0.817582 | 0.606122 | 0.830077 | 7.7816 | 5.9278 |
| Temporal-9 recommended, optical on | 0.810740 | 0.598099 | 0.821763 | 7.9605 | 6.1014 |
| Temporal-9 recommended, optics bypassed | 0.498122 | 0.341768 | 0.458939 | 14.0074 | 11.9774 |

For the recommended checkpoint, enabling the optical paths raises SRCC by
0.312617 over the same checkpoint with optics bypassed. The four learned
fusion alphas are 0.5607, 0.5598, 0.5689, and 0.5709; the model therefore does
not obtain the result by numerically suppressing its optical branches.

Recommended checkpoint:

`runs/lgvq_temporal_qwenfront_o2_9f77_dc20_zero_order_crop/best_observed_test_checkpoint.pt`

Checkpoint epoch: 55  
Checkpoint SHA256: `2cb43fdfeb90aa626e9730485d1ea1ae56a9a4dae2c2a057b3519cfb70951bffc`

Maximum-accuracy checkpoint:

`runs/lgvq_temporal_qwenfront_o2_9f77_dc20_balanced/best_observed_test_checkpoint.pt`

Checkpoint epoch: 35  
Checkpoint SHA256: `56039361f9ab776b26ebbfd585b4e2a8addbe396681ae57985b94cd1d7ef95ebb`

## Data and reproducibility

The exact Qwen-front cache contains 2,808 videos and has tensor shapes
`vision=[2808,9,49,1024]` and `quality=[2808,9,49,14]`. It uses the official
Qwen3-VL-2B-Instruct processor, frozen patch embedding and interpolated visual
position embedding, frozen text token embedding, zero Qwen blocks, and no
Attention in the student.

Cache:

`artifacts/qwen3vl_front_9f_49x1024_quality14.pt`

Cache size: 2,571,361,590 bytes  
Cache SHA256: `48b0d916da8959c063a13bfd4f57546b57e72e945105b6af5bf84be4aac9156d`

Train/test counts are 2,250/558. Temporal MOS is 49.754 +/- 13.580 for train
and 50.690 +/- 13.692 for test.

Selected config:

`configs/release/temporal9_compact_zero_order_crop.yaml`

Run from the repository root:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.run \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/temporal9_compact_zero_order_crop.yaml \
  --phase train
```

## Figures

Paper-ready PNG (400 dpi) and vector PDF files are generated under
`runs/lgvq_temporal_qwenfront_o2_9f77_dc20_zero_order_crop/figures/`:

- `temporal_frame_packing_geometry`: 4/9/16-frame physical packing comparison;
- `temporal_dataset_and_sampling`: MOS distributions and temporal sampling;
- `temporal9_representative_frame_sequences`: low/mid/high MOS videos, all nine frames;
- `temporal9_selected_phase_planes`: all six learned optical phase-plane groups;
- `temporal9_metrics_router_training`: correlations, optics-off control, router use, and training trajectory;
- `temporal9_metrics_table.csv` and `temporal_dataset_summary.json`: plotting hand-off data.

All plot text uses Arial when installed, with Liberation Sans as the metrically
compatible fallback, and a base font size of 7 pt.

