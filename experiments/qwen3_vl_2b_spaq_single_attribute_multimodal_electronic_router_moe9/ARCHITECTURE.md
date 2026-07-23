# Architecture

## Teacher

```text
RGB SPAQ image + task-specific prompt
-> Qwen processor/chat template
-> frozen complete Qwen3-VL-2B vision stack
-> native vision merger and three DeepStack visual features
-> frozen complete Qwen language stack
-> final RMSNorm
-> last valid prompt token [2048]
-> LayerNorm(2048) + Linear(2048,1)
-> normalized attribute prediction
```

Teacher Qwen remains frozen. Only the small regression head trains.

## Student block alignment

For each replaced stack, the default block-aligned form is:

```text
A = X + Attention(Norm1(X))
Delta = OpticalMoE(Norm2(A))
Y = A + Delta
```

There is no learned residual scale and no post-residual activation. Vision and language attention modules are independent. By default their projection weights are randomly initialized and trainable; teacher initialization is an explicit configuration option.

Vision MoE exposes stage 1/3/4 and final outputs at Qwen's native DeepStack provider block indices. Language MoE consumes the three native DeepStack deltas at the same ordering points as Qwen.

## Optical planes

```text
electronic top-k router
-> amplitude SLM: sparse weighted copies in 3x3 expert layout
-> ideal 4f identity relay (zero modeled propagation)
-> co-planar expert phase SLM
-> 0.10 m propagation
-> CCD / per-expert LN / activation / route weight / hard mask / amplitude reload
-> next co-planar expert phase SLM
```

This repeats for five phases. After phase 5 and its 0.10 m CCD/reload, the reloaded amplitude is co-planar with the global phase. Global phase then propagates 0.10 m to the final CCD.

There is deliberately no `prompt_phase`, fan-out lens, grating, or complex prompt transmission. The phase SLM contains only expert/global learned masks.

## Normalization map

- Adapter input: `LayerNorm(120)` over channels before Softplus.
- Each OEO plane: independent `LayerNorm((120,120), affine configurable)` inside each selected expert.
- Hard route masking: after activation; unselected expert fields are exactly zero.
- Routing coefficient: reapplied after per-expert normalization, because normalization otherwise erases relative amplitude.
- Final detector readout: `LayerNorm(120, affine=False)` independently for every token row.
- Hidden distillation loss: non-parametric LayerNorm over hidden dimension, only inside the loss.

This separation avoids the former full-field final LayerNorm failure mode while preserving the established inter-expert OEO behavior.

## Loss

```text
L = lambda_vision * normalized vision hidden MSE
  + lambda_answer * normalized answer hidden MSE
  + lambda_prediction * SmoothL1(student, teacher)
  + lambda_regression * SmoothL1(student, target)
  + lambda_balance * router balance
  + lambda_importance * router importance
```

Labels and model outputs use the normalized 0–1 training scale; MAE is reported on 0–100. SRCC and PLCC are also reported.
