# Model card

## Model and inputs

This temporal quality reference model uses a frozen Qwen-derived visual front,
two Top2 routers, four independent linear experts at each stage, two simulated
optical global stages, and a two-layer five-grade readout. The fixed prompt
embedding and frozen visual tensors are embedded in the standalone checkpoint.
Five TRAIN-derived grade scores are combined using predicted probabilities to
produce one mean opinion score (MOS) per video.

The parameter count is 5,732,678. Each input contributes four 448×448 RGB
frames. Ordered groups of sixteen videos share a coherent optical simulation.
The order and cyclic tail-padding convention affect the physical field.

## Evaluation scope

The LGVQ temporal evaluation uses 2250 TRAIN videos and 558 original TEST
videos. The original TEST split was used for development selection. Padding
samples are excluded from metrics.

| Evaluation | SRCC |
|---|---:|
| Original ordered groups, cached features | 0.7991007779 |
| Direct video input, full 558 videos | 0.7992393302 |
| Mean over sixteen cyclic slot shifts | 0.7901063063 |
| Minimum over sixteen cyclic slot shifts | 0.7857706830 |
| Numerical 8-bit phase quantization | 0.7989927886 |

The direct-video and cached-feature predictions have a mean absolute MOS
difference of 0.017690 and a maximum of 2.016068. Eight videos exceed a 0.1 MOS
difference. The largest case has a Top2 boundary change in its first frame.
This field-based model exhibits input-order and numerical-boundary sensitivity.
The lowest SRCC among the tested two-pixel detector shifts is 0.7823957337.

## Global ablations

These are post-training forward interventions with the remaining modules fixed.
Each random condition uses ten fixed seeds, with the same selected model and
558-video grouping. The standard deviation is the sample standard deviation
over seeds.

| Intervention | Vision only | Joint only | Both stages |
|---|---:|---:|---:|
| Entire GlobalStage identity bypass | −0.31554 | −0.79331 | 0.39344 |
| Uniform random phase mask, mean ± SD | 0.73820 ± 0.00650 | 0.77988 ± 0.00394 | 0.65240 ± 0.01611 |
| Default random raw phase, mean ± SD | 0.78086 ± 0.00219 | 0.78990 ± 0.00377 | 0.76441 ± 0.00548 |
| Random mappings and uniform phase, mean ± SD | −0.00443 ± 0.07594 | 0.01936 ± 0.28158 | 0.00132 ± 0.02413 |
| Complete default random GlobalStage, mean ± SD | −0.02306 ± 0.18993 | 0.05852 ± 0.37629 | 0.00677 ± 0.03910 |

An entire GlobalStage includes token-to-field mapping, phase modulation,
coherent propagation, detection, intensity processing, and electronic projection
back to tokens. Its identity bypass therefore measures the contribution of
that complete trained block. Random-phase interventions retain its trained
electronic mappings and optical geometry.

The default raw-phase initialization is N(0,0.25²), mapped through
`2π·sigmoid(raw)`. Uniform-phase controls sample physical phase independently
over [0,2π). SRCC retains its sign throughout the reported comparisons.

Per-seed aggregate metrics and condition definitions are provided in `results/`.

## Training and release

Visual-router and linear-expert tuning used TRAIN-only mixture/output
distillation and Top2 frequency feedback. The final phase stage selected epoch
86 from 260 epochs. Original raw phase parameters were initialized at zero,
which maps to physical phase π through the sigmoid parameterization. The final
stage continued from previously learned phase masks.

The public checkpoint has the same model tensors as the selected research
checkpoint. Its metadata describes the public model, upstream components,
evaluation scope, and tensor fingerprint. Source checkpoint and public-export
identities are recorded in `MANIFEST.json`.

## License and upstream components

The project uses Apache-2.0. The checkpoint contains components derived from
Qwen3-VL-2B-Instruct, including frozen patch-projection tensors, interpolated
position tensors, and a fixed-prompt embedding. Attribution and the upstream
license are included in `NOTICE` and `licenses/QWEN_LICENSE.txt`.
[Official model](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
