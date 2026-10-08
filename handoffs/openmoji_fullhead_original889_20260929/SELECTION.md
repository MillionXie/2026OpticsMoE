# OpenMoji full-head candidate selection (2026-09-29)

This set uses the original shared electronic readout (381,976 parameters), not
the later low-rank 234,520- or 160,792-parameter versions. All four models use
the same 17 µm / 10 cm optics and zero-preserving bounded amplitude contract;
the BMP bridge must use `round(255 * amplitude)` without another peak scale.
Nothing in this set has been deployed or measured on the optical bench.

| Display group | PT | Added training condition | Clean TRAIN holdout VAL (1000) | Common clean TEST (1000) | SHA-256 |
| --- | --- | --- | ---: | ---: | --- |
| G1 ideal / G2 direct deployment | `g1_g2_base.pt` | No added CCD/DC/grid perturbation; ordinary phase dropout 0.08 | 0.9620 (historical) | 0.8880 (fresh confirmation; original report 0.8890) | `579b19e78befe93cd258ce02308235f1a9176316f37bc42fc2e1b6ff06d38e9f` |
| G3 | `g3_ccd.pt` | CCD gain/offset/read-noise profile | 0.9665 | 0.8935 | `135fc7e436f6bcf04c610ccd90ecdf28fa56cb64db9ccd232738c067abcb0eb4` |
| G4 | `g4_ccd_dc30.pt` | G3 + coherent DC30% profile | 0.9650 | 0.8965 | `9f3527b4dec5a6f59fd64f83ff863450f6430a4befcc63a2fb7d476e9e628048` |
| G5 | `g5_ccd_dc30_grid.pt` | G4 + differentiable 17→8→17 raster proxy | 0.9655 | 0.8960 | `dd5d175b60a884254dd4f38a47ba95d71be8608233696b132fcf90ac73cbfd7c` |

G1 and G2 intentionally share one PT; there are five display conditions and
four weights. All TEST results above use **the same clean r0 inference profile**:
no extra detector noise, DC30%, or raster proxy is applied at evaluation. G3–G5
received their listed interventions *during training*. Thus this clean table
does not demonstrate physical or perturbed-condition robustness.

G3–G5 are short, one-minibatch continuation probes of the full-head base model.
They were chosen before TEST by comparing the fixed TRAIN-derived holdout to
the base 0.9620 validation level, then each checkpoint was evaluated once on
the original TEST. These probes are useful for identifying near-0.889 clean
checkpoints, **not sufficient evidence that the interventions improved
robustness**. The base warm start had previously seen TRAIN and the original
TEST was used historically during development, so none of these TEST scores
is a new blind external benchmark. The G5 grid is only a differentiable raster
proxy, not exact 8 µm propagation.

Per-group protocol/split and clean validation/TEST JSON files sit beside the
PTs. `report.json` is the independent G1/G2 confirmation. Do not deploy until
the user confirms which candidates to test.
