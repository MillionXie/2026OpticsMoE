# Architecture

## Teacher

```text
RGB image + task prompt
-> Qwen3-VL processor/chat template
-> frozen electronic Qwen3-VL-2B vision stack
-> native vision merger and three DeepStack injections
-> frozen electronic language stack
-> final RMSNorm answer hidden [B,2048]
-> shared normalized linear regression head
```

Teacher cache captures four vision targets at original vision block indexes
`[5, 11, 17, 23]` and the final answer hidden. The Qwen backbone remains in
`eval()` and is never optimized.

## Student stack replacement

Vision uses optical stage outputs `[1,2,3,4]` for those four teacher targets.
The first three outputs preserve native Qwen three-point DeepStack injection
timing; stage 4 is the final vision output passed to the frozen merger.

Language uses four optical stages. Stage 0 receives the multimodal language
input. Native DeepStack deltas are added before stages 1, 2 and 3. This keeps
Qwen's three injection events while replacing the original 28 decoder blocks.

No additional attention layer is used. Both stacks use the fixed residual:

```text
Y = X + OpticalMoE(X)
```

There is no learned residual coefficient and no post-residual activation.

## Optical geometry

```text
expert:        224 x 224
grid:          4 x 4
gap:           30
pitch:         254
footprint:     4*224 + 3*30 = 986
global phase:  986 x 986
outer padding: 20 per side
FFT canvas:    1026 x 1026
CCD ROI:       986 x 986
readout:       AdaptiveAvgPool -> 224 x 224
```

The 1026 canvas is only the padded propagation grid. It is not treated as the
physical effective detector area. Final detector intensity saved by the core
is the cropped 986×986 ROI.

## Four optical stages

Router selection is computed once from the 224×224 input field and reused in
all stages. No new router is evaluated between stages.

```text
InputAdapter -> LN -> Softplus -> [224,224]
-> router probabilities -> sparse top-4 weights
-> direct weighted amplitude copies

for stage in 1..4:
    16 independent 224x224 phase masks
    -> shared 10 cm angular-spectrum operator on 1026 canvas
    -> intensity |E|^2
    -> per-selected-expert non-affine LN
    -> ReLU
    -> multiply original routing weight
    -> hard-zero unselected experts
    -> zero-phase complex reload

-> 986x986 global phase
-> 10 cm propagation
-> |E|^2
-> crop 986x986
-> AdaptiveAvgPool2d(224,224)
-> non-affine LN + ReLU
-> OutputAdapter
-> residual add
```

Because all hops are 10 cm, each core reuses one immutable angular-spectrum
transfer-function buffer. This does not share trainable masks or alter the
propagation equation; it only avoids five duplicate 1026×1026 buffers.

## Loss

The training objective is unchanged from the source experiment:

```text
L = lambda_vision * normalized vision hidden MSE
  + lambda_answer * normalized answer hidden MSE
  + lambda_prediction * SmoothL1(student prediction, teacher prediction)
  + lambda_regression * SmoothL1(student prediction, ground truth)
  + lambda_balance * router balance
  + lambda_importance * router importance
```

The default head output is linear. Scores are trained in normalized `[0,1]`
units and reported on the original 0–100 SPAQ scale.
