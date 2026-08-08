# Grocery10 Optical Retrieval

这是 10 种包装商品的图像到图像检索实验，不是十分类器。Frozen Qwen3‑VL‑Embedding‑2B 提供 64D teacher embedding；Student 同时替换 Vision 和 Language stack，最终输出允许正负值的 64D L2-normalized retrieval embedding。

## 当前推荐版本

当前实物实验使用 2×2、四专家、Top‑2 的 MoE4：每个 Vision/Language stack 都只有一个 expert phase plane 和一个 global phase plane。

可加载的推荐 checkpoint：

```text
runs/qwen3_vl_embedding_2b_grocery10_moe4_from31_epoch40_replay/ema_last_checkpoint.pt
```

完整复评：Top‑1 67.69%，Top‑3 87.31%，MRR 79.16%。历史 Top‑1 73.46% 属于 4×4/MoE16，不是同一个四专家结构，checkpoint 与 phase mask 均不可混用。

## 四层实物数据流

```text
Vision routed amplitude + Vision expert phase
→ propagation → CCD-1
→ per-expert LN → ReLU → same routing weights → hard-zero unselected experts
→ Vision reload amplitude + Vision global phase
→ propagation → CCD-2
→ detector pooling/LN/ReLU → output adapter/residual → frozen Qwen visual bridge
→ Language routed amplitude + Language expert phase
→ propagation → CCD-3
→ per-expert LN → ReLU → same routing weights → hard-zero unselected experts
→ Language reload amplitude + Language global phase
→ propagation → CCD-4
→ detector pooling/LN/ReLU → final RMSNorm → LN/Linear64 → L2 normalize
→ cosine retrieval
```

CCD 文件已经是平方律强度，电子桥不会再次平方。

## 固定全量硬件 manifest

正式硬件配置使用 `selection.mode: full_dataset`，一次性导出并固定：

- gallery 登记图；
- 全部训练图；
- 全部测试 query；
- 四个共享 phase BMP；
- 所有样本逐层同名的 amplitude BMP、理论 CCD 和元数据。

微调默认只使用 gallery + train 实测结果，test 仅作最终独立评测。配置 `adaptation.include_test_split: true` 可让 test 参与硬件域校准，但所得 test 指标会被标记为 transductive/selection-biased。

## 两条互不混淆的实物路线

1. 不微调：每层采集后运行 `hardware_pipeline --phase process_<stage>`，只生成下一层输入，第四层直接评测。
2. 逐层微调：每层采集后运行 `hardware_finetune --capture-stage <stage>`，冻结本层及上游，只训练下游 100 epoch，再导出更新后的 mask、checkpoint 和下一层全量振幅。

若只想完成一次不微调的纯推理，可在 prepare 时使用 `--selection-mode test_only`，只导出 gallery + 全 test；该 manifest 会被微调入口明确拒绝。

所有从零训练、BMP 导出、四层采集、checkpoint 串联和最终推理命令统一见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。物理相机与 SLM 播放见 [共享硬件工程](../hardware_sdk/README.md)。
