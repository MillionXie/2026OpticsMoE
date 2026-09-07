# ABO similarity-10 冻结 Qwen A100 baseline

## 结论

这是表格 03 的 ABO 图搜图 baseline：一张未见过商品的图片检索 120 个 train 商品
centroid，同类别的 12 个 gallery 商品都视为相关。完整 480 张 test query 的结果如下。

| 指标 | 结果 |
|---|---:|
| unrestricted R@1 = Hit@1 = P@1 | **0.9521** |
| R@5 / R@10 | **0.9875 / 0.9979** |
| P@5 / P@10 | 0.9208 / 0.8548 |
| positive Recall@5 / @10 | 0.3837 / 0.7123 |
| mAP@10 / NDCG@10 | **0.8325 / 0.8807** |
| category-prototype route accuracy | **0.9208** |

这里的 R@K 是“Top-K 中是否至少有一个同类别商品”的 query 命中率；positive Recall@K
才是平均找回 12 个相关商品中的比例。必须保留这个区别，不能把 `positive Recall@1 =
0.0793` 错写为表格中的 R@1。

## A100 速度与能耗

| 项目 | 结果 |
|---|---:|
| 性能测试量 | 480 queries |
| 计时量 | 200 queries，类别均衡、不重复 |
| 显式 warm-up | 50 forwards |
| 核心时延 mean / median / P95 | **49.174 / 46.157 / 68.927 ms/query** |
| 推理吞吐（由 mean 折算） | 20.34 queries/s |
| idle / active mean / active peak | 61.47 / **82.96** / 101.11 W |
| active 能耗 | **4.079 J/query** |
| idle-subtracted 能耗 | **1.057 J/query** |
| 250 W 额定上界 | 12.294 J/query |

设备为物理 GPU 6 `NVIDIA A100-PCIE-40GB`，PyTorch 2.6.0+cu124，bfloat16，batch 1。
时延从第一个原生 Vision Transformer block 的输入开始，到完整 Qwen Vision/Language
blocks、2048D L2 normalization、120 个 gallery cosine、10 个 category-prototype cosine
及排序结束；不含文件读取、JPEG 解码、processor、patch embedding、gallery 构建和模型加载。

## 模型与数据合同

- 模型：完整 `Qwen3-VL-Embedding-2B`，2,127,532,032 参数，全部冻结；没有 LoRA、
  微调或训练读出头。
- 输入：224×224；gallery 与 query 使用完全相同的图像 embedding 指令。
- 数据：10 类、200 商品、每商品 12 视角。train/val/test 分别为 120/40/40 个商品，
  商品身份严格不重叠。
- gallery：120 个 train 商品；每个 centroid 是该商品 12 个视角 2048D embedding 的
  平均值再 L2 normalization。
- query：40 个未见 test 商品的 480 个视角；validation 不参与本 frozen baseline。
- 数据 ZIP SHA256：`c8f0f79cbd8ceb3c42162092136254a1f44f6d13b3329e0e3b1c5dfc458c1485`。

## 证据

- 正式 run：`tasks/t07_abo_image_retrieval/runs/simulation/qwen_frozen_a100_20260907`
- 代码 commit：`354c53e54b9d26b1d7835a3ba51b60929425baa8`
- `evidence/baseline_report.json`：完整合同、性能、时延、功率及环境。
- `evidence/per_category_metrics.csv`：每类 48 条 query 的独立指标。
- `a100_baseline_overview.png/.pdf`：性能、分类别 P@1 和原始 active 功率曲线。
- `evidence_manifest.json`：归档证据 SHA256。

原始的 480 条预测、200 条计时、1254 条功率样本和 gallery index 保留在服务器正式 run，
不提交 Git。
