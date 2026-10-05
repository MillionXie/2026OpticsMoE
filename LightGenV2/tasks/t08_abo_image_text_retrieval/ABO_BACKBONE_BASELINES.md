# ABO frozen-backbone baselines

This runner preserves the three published Qwen retrieval protocols instead of
merging their different candidate sets:

| Direction | Fixed protocol | Published frozen-Qwen reference |
|---|---|---:|
| image to text | easy100: 2,400 test images rank 100 official titles | R@1 0.7358 |
| image to image | ABO-200: 800 queries rank 1,600 enrolled images | R@1 0.8513 |
| text to image | easy100: 100 titles rank 2,400 test images | Hit@1 0.8200 |

`CLIP ViT-B/32` uses its native image and text embeddings. `DeepSeek-VL2-Tiny`
uses the final decoder layer at the last valid prompt token, matching the Qwen
readout position. Both are completely frozen and have no fitted retrieval head.

`YOLO11s` is visual-only. Its image-to-image baseline uses the frozen global
average of the layer-10 C2PSA feature map. For cross-modal retrieval only, it is
reported as the explicit composite `YOLO11s + frozen CLIP text encoder + linear
alignment`: YOLO and CLIP remain frozen; a bias-free 512-by-512 matrix is fitted
on the easy100 training split. This fitted composite must not be described as a
zero-shot YOLO result.

The executable writes resumable feature shards, a final `report.json`, exact
predictions, parameter counts, the data/manifest identity, and the Qwen source
commit references.
