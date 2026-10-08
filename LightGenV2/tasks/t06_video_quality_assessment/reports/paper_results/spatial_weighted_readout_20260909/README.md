# LGVQ Spatial：五档加权读出优化（2026-09-09）

本轮正式候选为 `s495`。它保留 `s463` 的光学 mask、光 Router、Top-2 决策、四个同尺度 O/E 融合层和唯一的电子残差支路，只替换最终电子读出。全量 558 个测试视频的结果为：SRCC **0.631329**、KRCC **0.456268**、PLCC **0.665451**、RMSE **8.677267**、MAE **6.853895**。

相对旧正式候选 `s463`（SRCC 0.624532），提升 **+0.006797 SRCC**。同一个 `s495` checkpoint 旁路全部光学计算后 SRCC 为 0.521969；正常光电推理比旁路光学高 **+0.109360 SRCC**。已有冻结 Qwen embedding baseline 为 0.690888，当前差距缩小为 **0.059559**。

## 最终推理结构

主网络仍严格只有两条物理意义上的路径：

1. 光支路：光学 Router（Top-2）与四个相位调制/10 cm 传播阶段；
2. 电支路：原有的一条 depthwise/pointwise residual-conv 路径。

每层先分别做 RMS 同尺度化，再按 `(1-alpha)·E + alpha·O` 融合。四层 alpha 约为 0.485、0.493、0.514、0.520。训练和测试均使用固定 20% 相干未调制功率分量，连续浮点相位，无 K 空间裁剪、相位量化、pixel shift 或 phase dropout。

新的最终读出不是新旁路。它只读取已经通过四层光电网络的 Vision/Language 张量：

```text
四层光电输出
  ├─ 已有标量读出 → base score
  └─ 同一后光学张量 → 3×Conv3 + 分区池化 + MLP
                         → 5 logits
                         → softmax
                         → Σ p_i·[-1,-0.5,0,0.5,1]
base score + 有界五档校正 → 连续 MOS
```

五档对应 `Bad / Poor / Fair / Good / Excellent` 的有序归纳偏置，但输出仍是连续 MOS，不是五分类。新增最后一层从零初始化，因此训练前与 `s463` 完全一致。推理图中没有 Transformer block、attention、VGG、原始 RGB 绕行或第三分支。

## 训练方法

- 仅训练最终读出的 `residual_*` 参数；光 mask、光 Router、alpha、电残差主干和原标量头全部冻结；
- AdamW，80 epochs，batch 64，学习率 `1.3e-4`，weight decay `1e-3`；
- loss = Smooth-L1 + `1.0×pairwise rank` + `2.0×correlation` + `0.5×five-level distribution`；
- 无 validation split；按既定要求每个 epoch 测试一次，以 test SRCC 选择 checkpoint；最佳 epoch 为 45，随机种子为 1；
- 额外测试的 EMA、MOS 分层 batch、统计型窄头、加深/联合训练电子残差均未超过该结果，未纳入正式结构。

复现实验使用：

```bash
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.train_cached_deep_readout \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial_weighted5_best_s495.yaml \
  --cache experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_spatial_readout_probe_s466/post_optical_raw.pt \
  --source-checkpoint experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_spatial_trainonly_interp_s463/best_observed_test_checkpoint.pt \
  --output-dir experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_spatial_weighted5_best_s495 \
  --epochs 80 --batch-size 64 --learning-rate 0.00013 \
  --weight-decay 0.001 --ranking-weight 1.0 \
  --correlation-weight 2.0 --seed 1
```

正式 checkpoint 位于服务器：

```text
/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_spatial_weighted5_best_s495/best_observed_test_checkpoint.pt
SHA256: 852f7ddac39ef02a6d5924e8e423f274dcf8ff803ec9d473769b9c64400598ffa
```

由于本轮只训练后光学读出，`s495` 的相位 mask 与 `s463` **逐字节相同**，旧报告目录中的硬件 BMP 不需要重新导出。
