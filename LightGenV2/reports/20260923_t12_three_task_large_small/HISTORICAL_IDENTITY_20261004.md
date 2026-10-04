# 历史测速与 baseline：身份说明

此目录是 **2026-09-23 的历史七模型对照**，不是当前 T12 部署入口。
原 README 原样保留，其中 “current primary/final bundle” 只描述当时版本。
当前 T12 从 [任务复现入口](../../tasks/t12_text_to_image/reports/reproduction/README.md) 进入。

## 必须保留的内容

- 完整电子 baseline：Qwen28 + 原电子 decoder，计数 1,824,200,950 参数。
- 三个历史大版：约 282M；三个历史小版：18,975,128 参数。
- 更早的 3.27M byte-GRU/no-Qwen 对照只作为消融保留，不冒称正式小版。
- 原计时 JSON、质量摘要、原图与对照图全部保留，包括服务器没有的本地独有图片。
- 新版 149.76M / 9.96M / 17.03M 不能沿用此处旧模型的速度或质量。

## 测速边界

七模型同一 A100-PCIE-40GB、batch=1、20 次预热、100 次 CUDA-event 测量。
计时从首个保留语言 block 输入至生成 RGB，排除模型加载、分词、词嵌入查表、H2D。
电子 baseline 均值 63.738 ms；历史光电模型的非光学部分为实测，
另外加的六层 6.2682 ms 是合同估算，**不是实验台端到端实测**。
原始 p50/p95/min/max 仍在 `latency_a100_matched.json`，没有重跑或重算。
大型 latent MSE 与小型 RGB MSE 不能直接比较。

## 本轮核验与保护

服务器已验证运行工作树 `t12_physical_robust_v2_20260927` 的 HEAD 为
`7093ec46082eed2fae127ec5028d3e2e8548b592`。2026-10-04 只读核验发现
服务器此目录 14 份文件与本地逐字节 SHA256 相同；本地另有 12 份文件，
一律保留，未仅因服务器缺失而删除。

主线此前缺少历史 README；从已保全引用
`ac7e3ecce0bef96ae7d91f8de6e478542913bd4e` 进行 missing-only 纳入，
9082 字节，SHA256 `ab615c3e7fc6804eed542058492a020463fcf3548fbdf7dff5b19ba363aa0565`。
工作目录中用户原 README 未改写；图片、计时 JSON、权重均不入 Git。
逐文件资产身份见
[保护清单](../../../maintenance/storage/T12_HISTORIC_TIMING_PRESERVATION_20261004.json)。

这是保全旧证据，不是重新验证七份权重、复测 A100、发布新模型或恢复外部上传。
旧权重的完整 SHA/依赖闭合仍需后续审计；原报告中的权重路径继续保留。
