# Architecture

```text
224×224 RGB
→ frozen Qwen3-VL-Embedding-2B patch/position stem
→ Optical MoE16 stage 1 (top-4)
→ OEO
→ Optical MoE16 stage 2 (same routing decision reapplied)
→ OEO
→ Optical MoE16 stage 3 (same routing decision reapplied)
→ global phase
→ 10 cm propagation
→ physical CCD intensity
→ detector pooling/normalization to [224,224]
→ Fccd + alpha·Linear(LayerNorm(Fccd))
→ restore visual token grid [224,Htoken,Wtoken]
→ lightweight segmentation decoder
→ [1,224,224] mask logits
```

The Qwen Vision Transformer blocks and language model are not executed by the
student. The exact optical geometry comes from the COCO/DUTS source
configuration and is checked against checkpoint metadata before weights load.

