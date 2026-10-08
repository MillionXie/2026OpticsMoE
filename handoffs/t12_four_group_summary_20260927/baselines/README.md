# 两组文生图 baseline 图片

这里汇集固定 TEST 集的原生 256×256 无损 PNG，供论文排图使用。两组各有 2304 张，`test_00000.png` 至 `test_02303.png` 文件名逐张对应；身份、编辑模式和 prompt 已全部核对一致。图片从原始交付逐字节复制，未重新编码、锐化或放大。

- Qwen28 + decoder：`qwen28/images/generated/`；逐图指标和 prompt 在 `qwen28/sample_metrics.csv`。
- pix2pix-Turbo：`pix2pix_turbo/images/generated/`；逐图指标和 prompt 在 `pix2pix_turbo/sample_metrics.csv`。

同一个编号的输入和目标图可从上级 `baseline/images/reference/`、`baseline/images/target/` 取得。两个 CSV 是原始表的未修改副本，其 `generated_image` 列在各自目录内有效；`reference_image` 和 `target_image` 列请以上述公共目录为准。每组 `report.json` 记录模型权重与完整 TEST 汇总指标。

Qwen28 和 pix2pix-Turbo 都是仿真 baseline，不要与光路实拍结果混称。同样本对照时只配对相同的 `test_index`，不要凭图像相似程度手动重新匹配。
