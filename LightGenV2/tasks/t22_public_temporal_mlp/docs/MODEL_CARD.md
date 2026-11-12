# Model card

## Model and inputs

This temporal quality reference model uses a frozen Qwen-derived visual front,
two Top2 routers, four independent linear experts at each stage, two simulated
optical global stages, and a two-layer five-grade readout. The fixed prompt
embedding and frozen visual tensors are embedded in the standalone checkpoint.
The text adapter projects the prompt at runtime. Five TRAIN-derived grade
scores are combined using predicted probabilities to produce one mean opinion
score (MOS) per video.

The parameter count is 3,777,350. Each input contributes four 448×448 RGB
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
The resulting readout is Flatten4032, Linear4032→32, ReLU, and Linear32→5,
followed by softmax and the grade expectation. The readout has 129,221 parameters.

The optical input encoders, coherent propagation, phase parameters, routers,
experts, text adapter, and frame merger retain the selected parent model's
values. The current refinement optimizes the readout parameters.

## Evaluation scope

The LGVQ temporal evaluation uses 2250 TRAIN videos and 558 original TEST
videos. The original TEST split was used for development selection. Padding
samples are excluded from metrics.

| Evaluation | SRCC |
|---|---:|
| Original ordered groups, cached features, full-model forward | 0.7988976462 |
| Direct video input, full 558 videos | 0.7992454773 |
| Mean over sixteen cyclic slot shifts | 0.7953460324 |
| Minimum over sixteen cyclic slot shifts | 0.7885972250 |
| Numerical 8-bit phase quantization | 0.7993216603 |

Full-model evaluation determines the reference metric. Head-only cached-feature
evaluation during fine-tuning has a slightly different GPU matrix shape.
Development selection applies the requested 0.790–0.799 SRCC range to cached-feature
full-model evaluation. Direct-video input is evaluated separately. Input-order and
numerical-boundary sensitivity are evaluated on the selected full model.

The direct-video/cache mean absolute MOS difference is 0.019374,
with maximum 2.122620. The lowest tested two-pixel
detector-shift SRCC is 0.7747974614.

## Electronic-capacity check

A companion electronic model uses the input adapters, custom mean/max frame
merger, position parameters, and a two-layer 8064→32→5 readout. Its parameter
count is 942,341, and its development SRCC is 0.7937765654. Its retained
electronic layers are optimized on the 2250 TRAIN videos for 400 epochs from
the corresponding parent initialization. The optical reference optimizes the
129,221 readout parameters over 400 epochs and uses a 4032-dimensional head
input. This capacity comparison has different head input dimensions and
optimization scopes, which must be accounted for in interpreting the result.

The electronic control selects hidden units from all 256 parent units using
its TRAIN activations. It uses seed 713 and selects its highest evaluated
development SRCC without an upper bound.

| Metric | Optical reference | Electronic capacity control |
|---|---:|---:|
| SRCC | 0.7988976462 | 0.7937765654 |
| PLCC | 0.8005052856 | 0.8074254715 |
| RMSE | 8.4795709019 | 8.2045499911 |
| MAE | 6.3281161469 | 6.2657914521 |

Both models reach the 0.79 SRCC range in this single-seed comparison.

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
| Learned phases | 0.7988976462 |
| Constant phase pi | 0.7945205413 |
| Identity propagation | 0.6436202435 |
| Uniform random phase, ten seeds | 0.4412689405 ± 0.0662973695 |
| Default random raw phase, ten seeds | 0.7101837287 ± 0.0184968802 |

Random-condition SD is the sample SD across ten fixed seeds. The constant-phase
and identity-propagation conditions fix the remaining trained modules.
Per-condition definitions, seeds, and metrics are supplied in `results/`.

## Training and release

The selected parent was prepared through visual-router and linear-expert tuning,
followed by phase optimization. The current model merges the final detector
projection into the readout, selects 32 ReLU hidden units using TRAIN activation
contributions, and fine-tunes that readout for 400 epochs. Development selection
chooses epoch 290 within the requested SRCC range. Optimization and
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
