# Architecture and isolation guarantee

The probe stops immediately before `visual.blocks[0]`. It explicitly calls the frozen Qwen patch embedding and positional embedding code, reconstructs per-image groups from `image_grid_thw`, and passes each group to the trained vision input adapter.

```text
[T_v,1024] -> Linear -> LayerNorm -> Softplus -> zero-pad [64,64] -> flatten [4096] -> probe
```

No optical phase mask or detector is evaluated. The `conversions` modules exist only so the source checkpoint remains structurally compatible; the feature API never invokes them. Feature-cache metadata records both executed and skipped module lists so a run can be audited.

