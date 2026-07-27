# Architecture

## 1. Design goal

This baseline keeps the Qwen3-VL multimodal interface and the MoE16-224
electronic-routing design while reducing each optical stack to exactly two
trainable phase planes:

1. one plane containing 16 independent expert masks;
2. one shared global phase plane.

Vision and Language instantiate separate optical cores and do not share masks,
adapters, routers, or detector statistics.

## 2. Frozen electronic teacher

```text
RGB image + task prompt
-> Qwen processor/chat template
-> full electronic Qwen3-VL vision stack
-> native vision merger and DeepStack injection
-> full electronic Qwen3-VL language stack
-> final RMSNorm
-> last valid prompt token [2048]
-> LayerNorm(2048) + Linear(2048, 1)
-> normalized quality score
```

All Qwen parameters remain frozen and in eval mode. Only the teacher regression
head is trained. Teacher vision taps, answer hidden, and predictions are cached.

## 3. Shared one-stage optical core

For a hidden group `X ∈ R[T,D]`, where `T <= 224`:

```text
X
-> Linear(D,224)
-> LayerNorm(224)
-> Softplus
-> zero row padding
-> A ∈ R+[224,224]
```

No interpolation or token truncation is used. More than 224 visual or language
tokens raises an error.

### 3.1 Electronic input-dependent routing

The router pools `A` to `14 × 14`, applies input LayerNorm and a trainable linear
projection to 16 logits, selects Top-4 experts, then renormalizes the selected
probabilities. The selected weighted copies of `A` are loaded directly on the
amplitude SLM:

```text
4 × 4 experts
expert size = 224 × 224
pitch = 254
gap = 30
active footprint = 986 × 986
canvas = 1026 × 1026
```

Unselected regions are exactly zero.

### 3.2 Expert phase and OEO reload

All 16 selected/unselected expert masks occupy one physical plane:

```text
weighted amplitude copies
-> independent expert phase masks
-> 10 cm angular-spectrum propagation
-> square-law intensity detection
-> independent non-affine LayerNorm for each expert region
-> ReLU
-> multiply the same routing weight again
-> hard-zero unselected experts
-> zero-phase amplitude reload
```

The routing decision is computed only once. No new router is evaluated between
the expert plane and global plane.

### 3.3 Global phase and final CCD

The reloaded amplitude is treated as co-planar with the global phase mask:

```text
reloaded amplitude
-> one shared 986 × 986 global phase
-> 10 cm angular-spectrum propagation
-> square-law CCD intensity on the 986 × 986 active ROI
-> adaptive average pooling to 224 × 224
-> non-affine LayerNorm independently over each token row
-> ReLU
```

The numerical simulation does not add a propagation for the ideal 4f relay.
The relay is an identity copy between amplitude and phase SLM planes.

Readout takes the first `T` rows and applies `Linear(224,D)`. With residual
enabled:

```text
Y = X + Delta_optical
```

The identity coefficient is fixed at 1 and has no trainable scalar.

## 4. Vision replacement

Qwen vision patch embedding produces packed RGB visual tokens
`[sum(T_i),1024]`. The optical core maps them through:

```text
[T_i,1024]
-> optical input [224,224]
-> one expert phase plane
-> OEO reload
-> one global phase plane
-> [T_i,1024]
```

Qwen provides three native DeepStack paths plus one final output. This
single-stage student intentionally keeps only one auxiliary path:

- the decoded pre-global expert-stage output is passed through frozen
  `deepstack_merger_list[0]` and used as the sole auxiliary feature;
- native auxiliary paths 1 and 2 are disabled for student forward;
- the decoded post-global output is passed through the normal frozen merger and
  used as the main vision embedding.

The frozen native Qwen vision merger is still responsible for converting vision
hidden size 1024 to language hidden size 2048 at its normal locations.

Vision distillation therefore has exactly two targets: the teacher's first
native DeepStack tap and teacher final vision output. The full electronic
teacher still runs all three native auxiliary paths when producing its final
answer hidden.

## 5. Language replacement

The original language stack has 28 layers. Although native Qwen can inject
auxiliary visual tensors after slots 0, 1, and 2, this student passes a list of
length one and therefore injects only after slot 0.

The baseline therefore maps layers as follows:

```text
language slot 0: bypass -> the sole auxiliary visual injection
language slot 1: the single Language optical stage + global readout
language slots 2..27: bypass
frozen final RMSNorm
```

The optical language input therefore contains text embeddings, the main visual
embedding, and one auxiliary visual injection. For each valid sequence:

```text
[S_i,2048]
-> optical input [224,224]
-> one expert phase plane
-> OEO reload
-> one global phase plane
-> [S_i,2048]
-> fixed residual
```

Padding tokens do not enter optical fields or hidden losses.

## 6. Regression head and losses

Teacher and student use the same head architecture:

```text
LayerNorm(2048)
-> Linear(2048,1)
-> raw normalized score
```

There is no sigmoid. Dataset targets are normalized to `[0,1]`; reported
predictions are converted back to the 0–100 scale. The student head is freshly
initialized and is not copied from the teacher head.

Student loss:

```text
1.0 * mean(normalized MSE over 2 vision targets)
+ 1.0 * normalized final-answer hidden MSE
+ 0.5 * prediction SmoothL1 distillation
+ 1.0 * ground-truth SmoothL1 regression
+ 0.03 * vision/language router balance
```

The two vision targets are the first original Qwen DeepStack tap and final
vision output. Ranking, Norm-in-Norm, importance, SAM, phase dropout, and weight
decay are disabled.

## 7. Trainable parameter accounting

Per optical stack:

| Component | Vision | Language |
|---|---:|---:|
| expert phase plane | 802,816 | 802,816 |
| global phase | 972,196 | 972,196 |
| input adapter | 229,600 | 458,976 |
| adapter LayerNorm | 448 | 448 |
| output adapter | 230,400 | 460,800 |
| router | 3,152 | 3,152 |
| total | 2,238,612 | 2,698,388 |

The head has 6,145 parameters, for a total of 4,943,145 trainable student
parameters in the main Vision+Language optical mode.

## 8. Reproducibility guards

`Settings.validate()` rejects:

- any expert stage count other than 1;
- DeepStack tap mapping other than `[1]`;
- enabled native attention;
- disabled fixed residual;
- SAM, phase dropout, or nonzero weight decay;
- ranking or Norm-in-Norm auxiliary losses;
- historical student checkpoint initialization;
- geometry or propagation distances that differ from this baseline.

The resolved runtime architecture and these switch states are saved in
`config_resolved.json` and `model.json`.
