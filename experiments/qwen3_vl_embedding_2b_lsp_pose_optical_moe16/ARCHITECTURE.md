# Architecture and physical contract

## Why heatmaps rather than coordinate regression

The task requires preserving spatial order. Both paths retain the processor-provided `image_grid_thw`, restore packed Qwen tokens into a 2-D feature map, and predict one spatial heatmap per joint. A global vector or a 28-value MLP would discard the key inductive bias and is deliberately not used.

## Frozen electronic upper bound

The electronic model hooks the final native Qwen Vision block before the frozen merger. Qwen remains in `eval()` with `requires_grad=False`; only the pose head is optimized. No language instruction, token embedding, merger output, or Language Model is used.

## Optical replacement

The optical model keeps only the frozen Qwen patch/position stem. The first native Vision block is replaced by a capture block and all later blocks are identity bypasses. The capture block runs exactly one expert phase stage, the configured OEO conversion, one global phase plane, free-space propagation, and square-law CCD readout. The CCD result remains nonnegative before the detector LayerNorm/nonlinearity.

Physical defaults:

| item | value |
|---|---:|
| wavelength | 532 nm |
| pixel pitch | 8 µm |
| experts | 16 (4×4) |
| expert size | 224×224 |
| expert pitch | 254 px (30 px gap) |
| active footprint | 986×986 |
| FFT canvas | 1026×1026 |
| selected experts | top-4 |
| expert phase layers | 1 |
| global phase layers | 1 |
| propagation distance | 10 cm |
| CCD active observation | 986×986 |
| CCD readout | 224×224 |

The ideal amplitude-to-phase 4f relay is represented as a co-planar identity mapping and therefore has no explicit propagation operator. The electronic router is evaluated once from the input feature; later reload uses the same routing weights.

## Shape contract

```text
Qwen packed visual hidden      [sum(T_i), Dv]
optical input fields           [B, 224, 224]
full optical canvas            [B, 1026, 1026] complex
active physical CCD intensity  [B, 986, 986]
pooled detector readout        [B, 224, 224]
runtime spatial feature        [B, 224, Hq, Wq]
pose heatmaps                  [B, 14, 56, 56]
```

`Hq` and `Wq` are never hardcoded; they come from `image_grid_thw`. A mismatch between the packed tokens, optical valid rows, and runtime grid aborts execution.

