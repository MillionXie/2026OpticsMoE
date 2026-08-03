# Grocery10：当前维护命令

所有命令均在仓库根目录 `2026OpticsMoE/` 下执行。日常只需要下面三组。

## 1. 推荐的 MoE4 训练

该版本使用 2×2 专家、Top-2、raw phase 全零初始化（物理相位 π）、
0.65° K-space 通带、2×2 CCD 像素积分，并显式关闭 phase-DC loss。
不使用 wrapped-phase smoothness。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_hardware_robust.yaml \
  --phase all
```

一轮 smoke：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_hardware_robust_smoke.yaml \
  --phase all
```

## 2. 导出可直接播放的硬件文件

训练完成后执行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_pipeline \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_hardware_robust_export.yaml \
  --phase prepare
```

输出目录：

```text
experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_runs/grocery10_moe4_hardware_robust/
├── amplitude_bmp/       # 四个播放阶段的 1920×1080 振幅 BMP
├── phase_bmp/           # 四张 1920×1200 相位 BMP；导出前已上下翻转
├── theoretical_ccd/     # 四个阶段的仿真 CCD PNG
└── 00_manifest/         # 播放顺序及简要审计信息
```

振幅编码使用正像素 P95 截断与 gamma=0.8，提高弱有效像素亮度；严格为零的
未选专家、间隙和 padding 保持为 0。

## 3. 生成光路对齐/标定 BMP

```bash
python -m experiments.slm_calibration_bmp_generator \
  --config experiments/slm_calibration_bmp_generator/configs/slm_956.yaml
```

生成棋盘格、十字、圆孔、字母 A、5 cm/10 cm 透镜相位等常用图案。

## 测试

```bash
/home/guest3/miniconda3/envs/xml/bin/python -m pytest \
  experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests \
  experiments/slm_calibration_bmp_generator/tests -q
```

## 历史复现（仅在需要对照时使用）

历史最高结果的三阶段 Grocery31→Grocery10→EMA 配方仍保留：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.reproduce_best \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_best_reproduction.yaml
```

旧 `phase_dc_*` 配置仅用于复现实验，不再推荐用于实物光路。
