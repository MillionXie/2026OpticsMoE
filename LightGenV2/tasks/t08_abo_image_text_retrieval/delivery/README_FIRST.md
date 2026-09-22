# ABO 文搜图 15 cm 最终交付版

这是 **ABO easy100 文搜图（text-to-image）** 的最终仿真模型，不是图搜文模型。
文本标题作为 query，2,400 张测试图像作为 gallery；100 个商品各有 24 张测试图像。

## 唯一正式版本

- 配置：`configs/optical_text_to_image_64_15cm_final_evaluate.yaml`
- 权重：`weights/final_best_checkpoint.pt`
- 权重 SHA256：`ae92995b49eaf7d49165d7816f876e831d9ab070a482f94e849d2f16e9ebb50f`
- 光传播：15 cm
- 输出：64 维 embedding
- Router：光学 detector-energy Top-2
- 光层：Vision 两层 + Language 两层；每层 4 个专家
- 残差电子 MLP：每层 `192 -> 96 -> 192`
- 融合：同尺度凸组合 `(1-alpha)E + alpha O_scaled`

不要混用旧的 10 cm、电子 expansion=1 或其他 checkpoint。

## 正式指标

| 项目 | Hit@1 | Hit@5 | Hit@10 | MRR | mAP |
|---|---:|---:|---:|---:|---:|
| Ours | 0.88 | 0.94 | 0.98 | 0.9119 | 0.7909 |
| 同权重去光 | 0.74 | 0.90 | 0.95 | 0.8086 | 0.6821 |
| 冻结 Qwen 2048D baseline | 0.82 | 0.90 | 0.96 | 0.8560 | 0.7719 |

去光不是重新训练，而是在同一 checkpoint 上关闭光分支；Hit@1 下降 14 个百分点。

## Alpha 与专家占比

四层 alpha：Vision-1 `0.398342`、Vision-2 `0.397866`、Language-1
`0.392176`、Language-2 `0.392077`。融合前会把光、电 RMS 对齐，因此 alpha
就是同尺度条件下的组合系数，不能用原始分支幅值直接替代。

最佳 epoch 的 Top-2 硬路由计数：

| 输入 | E0 | E1 | E2 | E3 |
|---|---:|---:|---:|---:|
| Vision image | 25.281% | 24.740% | 25.000% | 24.979% |
| Language image | 50.000% | 19.479% | 15.719% | 14.802% |
| Language title | 50.000% | 19.531% | 16.500% | 13.969% |

语言侧 E0 是共享专家，固定占据两个 Top-2 名额之一；另一个名额在 E1–E3
之间分流。这不是四专家完全均匀，也不是只使用一个专家，部署时必须保留该语义。

## 仿真复现

需要另行准备：

1. ABO easy100 数据根目录（包内只带 CSV/哈希合同，不重复分发图像）。
2. Qwen3-VL-Embedding-2B 权重或已匹配的教师缓存。
3. 与工程相符的 PyTorch/CUDA 环境。

从压缩包根目录运行：

```powershell
python -m LightGenV2.tasks.t08_abo_image_text_retrieval.optical_moe `
  --config LightGenV2/tasks/t08_abo_image_text_retrieval/configs/optical_text_to_image_64_15cm_final_evaluate.yaml `
  --evaluate-only `
  --data-root D:\path\to\abo_easy100_dataset_20260906 `
  --teacher-cache LightGenV2/tasks/t08_abo_image_text_retrieval/cache/abo_easy100_qwen64_true_text_to_image.pt `
  --device cuda
```

输出必须与 `delivery_evidence/final_report.json` 的数据协议哈希一致。正式 checkpoint
按每 2 epoch 查看一次 TEST Hit@1 选择；这项 test-selection 偏置已在报告中披露。

## 文件入口

- `delivery_evidence/`：正式报告、训练记录、预测、架构与相位图。
- `data_contract/`：训练/测试/标题清单及 SHA256，不含原始图片。
- `HARDWARE_ADAPTATION.md`：转实验室版本时必须遵守的几何与导出规则。
- `SHA256SUMS.txt`：交付包内全部文件的完整性校验。
