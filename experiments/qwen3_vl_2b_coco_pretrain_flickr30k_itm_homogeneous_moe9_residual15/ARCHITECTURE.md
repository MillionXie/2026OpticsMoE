# Architecture

## Stage A: generic multimodal distillation

```text
COCO RGB image + selected human caption
  ├─ frozen electronic Qwen3-VL teacher
  │    ├─ vision targets: native tap 1, tap 2, tap 3, final stack output
  │    └─ language target: final valid-token answer hidden
  └─ optical student
       ├─ frozen Qwen vision patch embedding
       ├─ FrozenNorm -> Vision MoE9 residual15
       ├─ frozen vision merger + native three-point DeepStack injection
       ├─ FrozenNorm -> Language MoE9 residual15
       └─ frozen final RMSNorm -> answer hidden
```

Teacher parameters and Qwen student-side backbone parameters are frozen. Only optical adapters, routers, phase masks, configured OEO affine terms (disabled by default), output adapters and any explicitly trainable optical components update. There is no downstream task head in Stage A.

## Stage B: Flickr30k binary matching fine-tuning

```text
Flickr image + per-pair caption prompt
 -> same initialized optical student
 -> answer hidden [B,2048]
 -> normalized binary head
 -> raw logit [B]
```

The fixed source experiment protocol is preserved: balanced positive/negative manifests, raw-logit teacher KD, BCEWithLogits, and test-AUROC checkpoint selection as explicitly requested in the preceding Flickr experiment. This is selection-biased and is recorded in output metadata.

## Physical and logical depth

Each vision/language surrogate contains 15 physical phase planes per expert, arranged as 5 logical stages × 3 physical layers. A logical-stage call runs all three physical layers in sequence. OEO conversion remains between physical planes. The final physical layer then enters the existing global phase, propagation, square-law detector, 480→120 pooling, non-affine per-token LayerNorm, nonlinearity and output adapter.

The grouping matters: Qwen sees five replacement blocks, not fifteen. Native DeepStack injections occur between logical language stages, never between arbitrary physical layers.

## Token/field mapping

```text
[T,H] -> Linear(H,120) -> LayerNorm(120) -> Softplus -> zero-pad rows -> [120,120]
```

No token-field interpolation is used. `T>120` or language sequence length `S>120` raises an error. Batch samples are encoded as independent fields; batching does not concatenate samples into one optical plane.

## Residual alignment without attention

The copied source block normalization is frozen. With the default config:

```text
branch = OpticalMoE(FrozenNorm(X))
Y = X + branch
```

There is no learnable residual coefficient and no post-add activation. `native_pre_attention_enabled=false` guarantees that attention is not part of the student. The optional attention implementation remains in code only for compatibility experiments and is disabled in all supplied configs.

## Generic cache identity

Generic manifests and caches bind to COCO source/revision/fingerprint, selected split values, stable caption selection seed, prompt template, processor budget, model id and manifest SHA256. Flickr caches independently bind to the fixed pair manifests. A mismatch aborts instead of silently reusing representations.

## SAM execution

For SAM-enabled configs, each optimizer batch performs:

1. normal forward/backward;
2. gradient-norm perturbation with radius `rho`;
3. a second forward/backward on the same cached teacher targets and inputs;
4. restore parameters and apply AdamW.

SAM touches only student trainables. Frozen Qwen and teacher tensors remain unchanged.
