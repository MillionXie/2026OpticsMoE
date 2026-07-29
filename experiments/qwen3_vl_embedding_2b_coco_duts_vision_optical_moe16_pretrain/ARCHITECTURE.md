# Architecture

## 1. Teacher target

```text
COCO image [3,224,224]
→ frozen Qwen image processor
→ frozen patch/position embedding
→ all frozen native Qwen Vision blocks
→ final pre-merger hidden [T,1024]
→ offline fixed PCA_vision
→ target [T,224]
→ FP16 sharded cache
```

The PCA coordinate system is fitted once from COCO train calibration tokens.
It is not a Student module and has no trainable parameters.

## 2. Student backbone

```text
Qwen patch/position hidden [T,1024]
→ input adapter / LN / Softplus
→ zero-pad rows to [224,224]
→ electronic Top-4 routing
→ direct weighted amplitude copies into four selected apertures
→ Expert phase plane 1 / propagation / OEO
→ Expert phase plane 2 / propagation / OEO
→ Expert phase plane 3 / propagation / OEO
→ global phase / propagation / CCD
→ detector readout [224,224]
→ residual CCD recombiner [224,224]
→ read first T rows [T,224]
```

The router runs once. Its selected expert set and weights are reused at every
OEO reload. No optical prompt phase, grating, amplitude prompt, attention,
language model, PCA, or hidden-size output adapter is present in Student.

## 3. CCD recombiner

For each token row:

```text
delta = Linear_224x224(LayerNorm_224(Fccd))
Fout  = Fccd + alpha * delta
alpha_init = 0.1
```

Both signs are allowed in `delta` and `Fout`. `Fccd` itself is the validated
nonnegative physical detector readout after crop, adaptive pooling, per-token
non-affine LN, and ReLU.

## 4. Parameter ownership

Trainable during COCO:

- Vision input adapter and input LayerNorm;
- electronic Top-4 router;
- 3×16 expert phase masks;
- one 986×986 global phase mask;
- optional OEO affine values (zero in the default non-affine config);
- CCD recombiner LayerNorm, Linear, and alpha.

Frozen:

- all native Qwen parameters;
- PCA;
- tokenizer and language model;
- Qwen Vision merger;
- segmentation head (absent in COCO).

During DUTS warmup only the segmentation head is trainable. During joint
fine-tuning all items in the COCO backbone list and the head are trainable,
while native Qwen remains frozen.

Default phase count:

```text
expert phase = 3 × 16 × 224 × 224 = 2,408,448
global phase = 986 × 986             =   972,196
total phase                              3,380,644
```

The report generated at runtime is authoritative for all electronic and
trainable parameter counts.

## 5. Spatial mapping

The runtime `image_grid_thw` controls all grouping:

```text
packed valid rows [sum(T_i),224]
→ split by runtime T_i
→ first T_i CCD rows
→ reshape to [B,224,H_i,W_i]
```

Any disagreement among Qwen grid, cached teacher grid, Student token lengths,
or mask-logit shape raises immediately.
