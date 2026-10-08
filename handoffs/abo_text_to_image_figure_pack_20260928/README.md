# ABO 文搜图：四组模型 Top-3 画图素材

这份包使用同一个 ABO easy100 测试协议：100 条官方商品标题作为文本 query，在 2,400 张测试图片中检索。每个标题对应 24 张同 SKU 的测试图片。绿色框表示检索到相同商品 ID，红色框表示不同商品 ID；它不是人工主观的“看起来像不像”判定。

| 模型 | 本包采用的版本 | 全部 100 个标题 Hit@1 |
|---|---|---:|
| LightGenV2 (Ours) | **10 cm** compact 光学仿真，epoch 12，64D | **0.86** |
| Qwen3-VL-Embedding-2B | 冻结权重，动态长宽，完整 2048D | 0.82 |
| DeepSeek-VL2-Tiny | 冻结权重，原生 processor/末层隐藏状态 | 0.08 |
| OpenAI CLIP ViT-B/32 | 冻结权重，原生图文 embedding | 0.64 |

`overview_top1.png` 是 10 个候选 prompt 的一页预览，`cases_overview.csv` 可按商品和模型筛选。`panels/` 中每个 prompt 有一张四模型 Top-3 图，可直接交给排版同学。`originals_by_case/` 按案例和模型/名次放了逐张**原始 JPG 字节副本**；`asset_index.csv` 提供每张原图的 sample ID、原数据集路径与 SHA256。`raw_dataset/images/` 保留这些原图在 ABO 数据集中的目录结构。`rankings.csv` 是逐项明细，`selection.json` 含完整 prompt、商品 ID、Top-3 与源文件哈希。

建议优先看看案例 16（两款相近的梯凳）、19（硬盘收纳包）、52（垃圾桶）、71（平板支架）和 17（床头柜）；其余 5 个提供不同产品类别的备选。10 个案例均满足：Ours Top-1 同 SKU，而 Qwen、DeepSeek、CLIP 的 Top-1 均为其他 SKU。**这些是刻意筛选的示例**，不能作为模型总体胜率的统计证据；总体性能应引用上表的完整测试集结果。

数据来源与复核：`source/ours_10cm_predictions.csv`、`source/ours_10cm_final_report.json`、`source/ours_10cm_run_manifest.json` 来自 2026-09-25 的 10 cm 运行；Qwen 来自已保存的 `dynamic_2048_predictions.json`。DeepSeek/CLIP 逐 query 排名原缓存未保留，本次用冻结权重和原 easy100 测试集重算，文搜图 Hit@1 分别复现为 0.08/0.64。三者的预测和报告、原始 `test.csv`/`titles.csv` 均在 `source/`。重新计算时最多占用两张 GPU；现已全部释放。

图注草稿："Representative text-to-image retrieval examples on ABO easy100. For each official product-title query, the top three images from LightGenV2 (10 cm optical simulation), frozen Qwen3-VL-Embedding-2B, frozen DeepSeek-VL2-Tiny, and frozen CLIP ViT-B/32 are shown. Green indicates the queried product ID; red indicates a different product ID. Examples were selected where LightGenV2 ranked a matching test view first and all three frozen baselines did not. Overall Hit@1 on all 100 queries is 0.86, 0.82, 0.08, and 0.64, respectively."

复现素材筛选和拼图：`python build_pack.py prepare`，下载 `image_paths.txt` 列出的原图到 `raw_dataset/` 后执行 `python build_pack.py render`。拼图只进行 EXIF 方向校正、等比例缩放和留白；`originals_by_case/` 不经过像素处理。
