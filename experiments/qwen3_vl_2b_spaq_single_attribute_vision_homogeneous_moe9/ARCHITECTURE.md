# Architecture

All three configurations use the same architecture and differ only in the selected SPAQ label and isolated output directory.

```text
SPAQ RGB image
-> Qwen image processor (25600 pixel budget)
-> frozen Qwen3-VL-2B visual patch embedding
-> teacher: complete frozen electronic vision stack
   student: homogeneous optical MoE9x5 replacement
-> [T,1024] vision hidden
-> mean across valid visual tokens
-> LayerNorm(1024) -> Linear(1024,1)
-> Brightness OR Colorfulness OR Contrast
```

The student retains the verified 480x480 canvas, 3x3 experts, top-3 input-dependent router, five phase layers per expert, global phase, detector readout, and output adapter from the SPAQ MOS source experiment. No Qwen language layer is called.

Loss:

```text
hidden-state distillation
+ teacher scalar-prediction distillation
+ ground-truth SmoothL1 attribute regression
+ router balance / importance terms
```

All physical, router, optimizer, dropout, logging, and visualization settings remain configurable in the grouped JSON schema.
