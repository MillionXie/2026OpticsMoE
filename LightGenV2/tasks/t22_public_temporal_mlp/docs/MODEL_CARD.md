# Model card

## Model and inputs

This temporal quality reference model uses a frozen Qwen-derived visual front,
two Top2 routers, four independent linear experts at each stage, two simulated
optical global stages, and a two-layer five-grade readout. The fixed prompt
embedding and frozen visual tensors are embedded in the standalone checkpoint.
The text adapter projects the prompt at runtime. Five TRAIN-derived grade
scores are combined using predicted probabilities to produce one mean opinion
score (MOS) per video.

The parameter count is 4,164,998. Each input contributes four 448×448 RGB
frames. Ordered groups of sixteen videos share a coherent optical simulation.
The order and cyclic tail-padding convention affect the physical field.

The frozen visual components provide patch projection and interpolated position
tensors. The two spatial mean-pooling operations, input adapters, mean/max
frame merger, and detector processing are task-specific components. The frame
merger applies mean/max pooling over 49 tokens, LayerNorm over 384 channels,
a 384-to-192 linear map, and ReLU.

## Detector processing and readout

Each detector region uses relative intensity

\[
I_{\mathrm{relative}}=I/\max(\operatorname{mean}(I),10^{-6}).
\]

Adaptive average pooling supplies 49×96 values per vision frame and 42×96
values per joint-stage video. Each pooled token passes through LayerNorm96 and
ReLU. The vision stage then applies a 96-to-192 projection. The joint-stage
projection and the first readout matrix are merged into a single linear map.
The resulting readout is Flatten4032, Linear4032→128, ReLU, and Linear128→5,
followed by softmax and the grade expectation.

The optical input encoders, coherent propagation, phase parameters, routers,
experts, text adapter, and frame merger retain the selected parent model's
values. The current refinement optimizes the readout parameters.

## Evaluation scope

The LGVQ temporal evaluation uses 2250 TRAIN videos and 558 original TEST
videos. The original TEST split was used for development selection. Padding
samples are excluded from metrics.

| Evaluation | SRCC |
|---|---:|
| Original ordered groups, cached features, full-model forward | 0.7989651231 |
| Direct video input, full 558 videos | 0.7992074548 |
| Mean over sixteen cyclic slot shifts | 0.7960946657 |
| Minimum over sixteen cyclic slot shifts | 0.7901525881 |
| Numerical 8-bit phase quantization | 0.7993673839 |

Full-model evaluation determines the reference metric. Head-only cached-feature
evaluation during fine-tuning has a slightly different GPU matrix shape.
Development selection applies the requested 0.790–0.799 SRCC range to cached-feature
full-model evaluation. Direct-video input is evaluated separately. Input-order and
numerical-boundary sensitivity are evaluated on the selected full model.

The direct-video/cache mean absolute MOS difference is 0.018583,
with maximum 1.472984. The lowest SRCC among the tested
two-pixel detector shifts is 0.7780928779.

## Phase and propagation controls

These are post-training forward interventions with the remaining modules fixed.
Phase-only and propagation-only controls retain the trained input and output
mappings. Constant or random phase masks replace the stated phase parameters.
Whole-stage randomization reinitializes the current GlobalStage parameters
and retains the final task readout.
Propagation controls replace the angular-spectrum operator with the identity
on the optical field while preserving intensity detection and electronic
processing.

| Both-stage intervention | Development SRCC |
|---|---:|
| Learned phases | 0.7989651231 |
| Constant phase pi | 0.7944393854 |
| Identity propagation | 0.6455167380 |
| Uniform random phase, ten seeds | 0.3831020843 ± 0.0669603449 |
| Default random raw phase, ten seeds | 0.7042872013 ± 0.0258067855 |

Random-condition SD is the sample SD across ten fixed seeds. The constant-phase
and identity-propagation conditions fix the remaining trained modules.
Per-condition definitions, seeds, and metrics are supplied in `results/`.

## Training and release

The selected parent was prepared through visual-router and linear-expert tuning,
followed by phase optimization. The current model merges the final detector
projection into the readout, selects 128 ReLU hidden units using TRAIN activation
contributions, and fine-tunes that readout for 400 epochs. Development selection
chooses epoch 225 within the requested SRCC range. Optimization and
teacher-prediction distillation use the 2250 TRAIN videos. The distillation loss
consumes TRAIN teacher predictions.

The selected phase masks and retained parameters outside the readout are verified
against the parent. The public checkpoint has the same model tensors as the
selected research checkpoint. Its metadata describes the public model, upstream
components, evaluation scope, and tensor fingerprint. Source checkpoint and
public-export identities are recorded in `MANIFEST.json`.

## License and upstream components

The project uses Apache-2.0. The checkpoint contains components derived from
Qwen3-VL-2B-Instruct, including frozen patch-projection tensors, interpolated
position tensors, and a fixed-prompt embedding. Attribution and the upstream
license are included in `NOTICE` and `licenses/QWEN_LICENSE.txt`.
[Official model](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
