# 数据与硬件合同

本工程不改变 10 cm robust 工程的数据和硬件格式。完整说明见：

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval_10cm_robust/DATA_PIPELINE.md
```

本版本额外固定以下规则：

- 正式 split：2625 train、200 sealed test、30 gallery；
- 训练期间不读取 sealed test；
- quick 最后一层：100 train、100 test、10 gallery，共 210 张；
- amplitude：478×478 compact PNG，实验室重建为 1024×1024 8-bit L BMP；
- 振幅极性：255=白/亮/透光，0=黑/暗/遮光，不反相；
- phase：478×478 compact PNG，导出为 1920×1200 8-bit L BMP；
- phase 中心 `(980,590)`，导出前竖直翻转；
- CCD：与 amplitude 同 stem，478×478 uint8 灰度 PNG；不做背景扣除；
- 每层 session 和 manifest 独立，不得把 robust、warmstart5、formal、quick 的 CCD 混用。
