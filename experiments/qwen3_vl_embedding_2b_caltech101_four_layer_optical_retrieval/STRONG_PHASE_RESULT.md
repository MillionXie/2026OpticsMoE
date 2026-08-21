# 17 µm strong-phase 联合训练结果

完成日期：2026-08-21。正式训练使用 GPU 4，从头联合训练 60 轮，未加载普通 17 µm
组或纯电子 checkpoint。配置为
`configs/release/caltech101_four_layer_optical_joint_17um_strong_phase.yaml`。

## 固定 checkpoint 结果

部署结果使用第 56 轮 `ema_best_train_loss_checkpoint.pt`。该 checkpoint 只按最低
训练总损失选择，没有使用测试集挑选，因此 `selection_biased=false`。

| 配置 | phase LR | 初始光学融合 | 固定 Top-1 | Top-3 | MRR | 最终 phase std |
|---|---:|---:|---:|---:|---:|---:|
| 普通 17 µm | 5e-4 | 0.05 | 88.0% | 96.5% | 0.9262 | 0.0554 rad |
| strong phase | 4e-3 | 0.15 | 83.0% | 93.5% | 0.8877 | 0.3624 rad |

strong-phase 的总 phase 标准差约为普通组的 6.5 倍，固定 Top-1 下降 5.0 个百分点。
反复观察测试集得到的最高值为 live 84.0%、EMA 84.5%，这些峰值仅用于诊断，不能
作为无偏正式结果。

## 四层 phase 诊断

| 光学层 | phase std | 从本次初始化移动的 RMS | `|delta|>0.01 rad` 比例 |
|---|---:|---:|---:|
| Vision expert | 0.4521 rad | 0.4520 rad | 93.01% |
| Vision global | 0.4297 rad | 0.4295 rad | 89.69% |
| Language expert | 0.2632 rad | 0.2626 rad | 53.05% |
| Language global | 0.2614 rad | 0.2607 rad | 52.76% |

- 最终 phase gradient RMS：`8.19e-5`；
- sigmoid raw phase 的 `|raw|>4` 饱和比例：0；
- `trainable_tensors_without_gradient=0`；
- Vision/Language router 的未选择专家数均为 0；
- 无 NaN、非有限梯度或 CUDA OOM。

`phase_preview.png` 显示去除每个相位面圆周均值后的相对相位。专家之间的固定零填充
间隔会显示为灰色，不参与 rad 标准差和色标计算。保存的张量与导出的 BMP 没有做该
显示变换。

## 服务器产物

```text
experiments/qwen3_vl_embedding_2b_caltech101_four_layer_optical_retrieval/
└── runs/caltech101_four_layer_moe4_joint_17um_strong_phase/
    ├── ema_best_train_loss_checkpoint.pt
    ├── student_metrics.json
    ├── metrics/evaluation_summary.json
    ├── metrics/phase_training_latest.json
    └── hardware_phase_export/
        ├── phase_preview.png
        ├── compact_phase/       # 四张 478×478 PNG
        └── phase_bmp/           # 四张 1920×1200、8-bit BMP
```

硬件 BMP 继续使用 `17/8` 物理坐标 nearest 栅格化、纵向翻转和相位中心
`(980,590)`。改变硬件中心只需重新导出，不需要重新训练。

对齐图位于：

```text
experiments/hardware_sdk/generators/slm_patterns/generated/
└── dual_slm_17um_8um_alignment/
```

其中有 25 张调幅 BMP、19 张调相 BMP，以及 6 组严格配对的
`registration_checker` 黑白棋盘与 `registration_checker_xy` 横纵 `0/π` 光栅。
每档必须拍 primary 和 complement，确保所有相位格至少被照亮一次。
