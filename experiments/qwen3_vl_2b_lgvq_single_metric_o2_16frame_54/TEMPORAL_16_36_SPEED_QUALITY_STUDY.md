# Temporal 16/36 帧并行：速度、质量与路由均衡

## 结论

在相同的 `478×478` 有效光场、相同的四专家 Top-2 光路由和六次串行光传播下，两套候选均达到目标：

| 方案 | Temporal SRCC | KRCC | PLCC | RMSE | MAE | Qwen 时间 | 六次光传播 | 理想计算加速 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 帧 | 0.8374 | 0.6278 | 0.8612 | 7.043 | 5.418 | 476.591 ms | 9.084 ms | 52.46× |
| 36 帧 | 0.8454 | 0.6394 | 0.8650 | 7.183 | 5.451 | 1133.494 ms | 9.084 ms | 124.78× |

这里的加速严格定义为 `给定的 Qwen 推理时间 / 六次光传播时间`。它不是实验台端到端吞吐量，尚未计入 SLM 刷新、稳定等待、CCD 曝光/读出、传输和最终电子读出头。

同一 checkpoint 直接屏蔽全部光支路，而不另训电子模型时：

| 方案 | 正常光电 SRCC | 同权重去光 SRCC | 差值 |
|---|---:|---:|---:|
| 16 帧 | 0.8374 | 0.5087 | 0.3286 |
| 36 帧 | 0.8454 | 0.2333 | 0.6122 |

这个消融只回答“已训练模型依赖光支路多少”，不能替代重新训练一个纯电子基线。

## 专家是否集中

所有数值均来自 558 个测试视频的硬 Top-2 选择统计，而不是训练损失的软概率。

| 方案 | 路由层 | E1 | E2 | E3 | E4 | 熵等效专家数（最大 4） |
|---|---|---:|---:|---:|---:|---:|
| 16 帧 | 帧级视觉光路由 | 29.56% | 24.94% | 23.54% | 21.95% | 3.975 |
| 16 帧 | 视频级序列光路由 | 25.36% | 28.23% | 23.57% | 22.85% | 3.986 |
| 36 帧 | 帧级视觉光路由 | 26.30% | 29.99% | 19.91% | 23.80% | 3.957 |
| 36 帧 | 视频级序列光路由 | 25.99% | 24.55% | 25.27% | 24.19% | 3.998 |

第二级路由使用的是四个 CCD 能量区的固定、非仿射标准化。均值和方差仅在训练集拟合，没有可学习缩放、偏置，也没有电子路由网络；决策仍由光传播后的四区能量和 Top-2 完成。

## 固定面积内的布局

两套方案的物理画布都是 `518×518`，中央有效区都是 `478×478`。

- 16 帧：`4×4` lane；lane 为 `114×114`，pitch 为 120，外围 offset 为 2。每个 lane 内为 `2×2` 四专家，每专家 `54×54`，专家 pitch 为 60。
- 36 帧：`6×6` lane；lane 为 `77×77`，pitch 为 79，外围 offset 为 3。每个 lane 内为 `2×2` 四专家，每专家 `37×37`，专家 pitch 为 40，内部缝隙仅 3 像素。
- 36 帧占用跨度为 `5×79+77=472` 像素，位于 478 有效区中央，两侧各留 3 像素；没有扩大 ROI 或改变光路。
- 后续视频级串行层保持 `109×109` 专家、pitch 123，不随帧数变化。

每条推理路径共六次光传播：帧级光路由、帧级 Top-2 专家、帧级全局 mask、视频级光路由、视频级 Top-2 专家、视频级全局 mask。训练仿真包含 20% 名义未调制/零级直流功率以及位置、相位和 CCD 扰动；学生网络中没有 Attention 或 Transformer block。

## 数据和选择口径

- 数据划分：训练 2250、测试 558，不另划验证集。
- 目标：只优化 LGVQ Temporal 分数。
- 文本条件保留 Temporal prompt：`Please evaluate the temporal quality ... Excellent, Good, Fair, Poor, or Bad.`
- 16/36 帧都从原视频均匀抽帧并通过冻结 Qwen3-VL-2B-Instruct 前端生成独立缓存；36 帧缓存不是由 16 帧缓存插值。
- 每 5 epoch 测试一次。36 帧正式权重锁定在 epoch 35；达到 SRCC 和路由均衡目标后停止其余重复训练，避免无意义占卡。

## 服务器结果位置

项目根目录：

```text
/DATA/DATA1/guest3/2026OpticsMoE/experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54
```

集中整理后的文件：

```text
artifacts/temporal16_36_study/
  models/
    temporal16_balanced_calibrated.pt
    temporal16_balanced_calibrated_report.json
    temporal36_balanced_calibrated.pt
    temporal36_balanced_calibrated_report.json
  temporal16_optical_fields/       # 六个 CCD 平面，6 个代表视频，PNG/PDF/原强度 CSV
  temporal36_optical_fields/
  temporal16_phase_masks/          # phase_preview、逻辑 mask、1920×1200 BMP、布局图
  temporal36_phase_masks/
  tradeoff/
    temporal_16_36_speed_quality_router.png
    temporal_16_36_speed_quality_router.pdf
    temporal_16_36_speed_quality_router.csv
    temporal_16_36_tradeoff_report.json
```

checkpoint SHA256：

```text
16f  efb5960985235f5ffc80efa48ccbbfbfab47676a163cee96c19ff07f6f3be4a8
36f  159b1d8cd31aa5f817d274f2930129601d4f0a365f01c430a8fefcc5989c8730
```

## 复现关键命令

以下命令均从仓库根目录执行。36 帧真实缓存约 9.6 GB。

```bash
P=experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54

CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_qwen_front \
  --config "$P/configs/release/temporal36.yaml" \
  --model-path /ABSOLUTE/PATH/TO/Qwen3-VL-2B-Instruct \
  --device cuda --batch-size 2 --chunk-rows 16

CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54 \
  --config "$P/configs/release/temporal36.yaml" --phase train

CUDA_VISIBLE_DEVICES=0 python -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.calibrate_router_channels \
  --config "$P/configs/release/temporal36.yaml" \
  --checkpoint /PATH/TO/CANDIDATE.pt \
  --output-checkpoint /PATH/TO/CANDIDATE_calibrated.pt \
  --report /PATH/TO/CANDIDATE_calibrated_report.json --device cuda
```

光场与相位导出分别使用 `visualize_optical_fields.py` 和 `export_hardware_masks.py`。PNG/PDF 的逐平面 p99.7 截断和 `log1p` 只用于显示；CSV 保存未做显示映射的原始强度统计，推理路径没有加入该显示处理。

## Spatial 包已传输

师姐服务器上的文件为：

```text
/root/autodl-tmp/lgvq_handoff/lgvq_spatial4_balanced_full_lab_20260904.zip
```

SHA256：

```text
57b9151d0151653d52528488c737aa0bae00c4f8da608f02066854e8e56236ae
```

目标服务器已重新计算并核对通过；同目录保留 `.zip.sha256`。
