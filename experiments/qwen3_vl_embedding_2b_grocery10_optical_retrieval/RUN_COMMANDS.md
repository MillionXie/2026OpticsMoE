# Grocery10：保留版本与硬件命令

所有命令都从仓库根目录 `2026OpticsMoE/` 执行。现在只保留两套配置：

| 名称 | 结构 | 用途 |
|---|---|---|
| `grocery10_moe16_best` | 4×4、16 experts、Top-4 | 历史最佳，保存的 EMA checkpoint Top-1 73.46% |
| `grocery10_moe4_latest` | 2×2、4 experts、Top-2、2×2 CCD integration | 当前最新实物鲁棒版本 |

## 1. 训练

当前 MoE4 从零训练：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest.yaml \
  --phase all
```

历史 MoE16 最佳是从 epoch-141 checkpoint 继续 40 epoch 得到。保留起点后可复现实验段：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe16_best.yaml \
  --phase all \
  --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_continue_epoch141_stronger_augmentation_ema/checkpoints/pre_resume_epoch_0141/resume_checkpoint.pt
```

该命令输出到独立 `...moe16_best_rerun`，不会覆盖保存的历史最佳。

## 2. 一键仿真验证硬件流程

先用少量样本验证四次曝光、四段电子处理、命名和指标：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_automation \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --simulate --yes --sample-limit 4 \
  --session-dir experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_simulation_smoke
```

MoE16 将配置名改为 `grocery10_moe16_best_hardware.yaml` 即可。

## 3. 真实 SLM/CCD 自动采集

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_automation \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --session-dir experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_physical_001
```

每层会提示：

```text
[vision_expert] 请准备并确认相位 mask：...bmp
输入 y 开始播放本层全部振幅；输入 q 安全退出：
```

输入 `y` 后自动完成本层全部播放与拍照。每张振幅加载后等待 40 ms，再调用相机；本层结束后自动计算 CCD-vs-theory MSE/PCC 和下一层振幅。

中断后继续同一目录：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_automation \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --session-dir experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_physical_001 \
  --skip-prepare
```

已有 capture 会按 basename 自动跳过。

## 4. 厂商 SDK 设置

当前配置使用：

- 振幅 SLM：HOLOEYE `showDataFromFile`；
- 相位 SLM：手工换 mask；
- 相机：DVP Python 3.5 持久子进程，原始帧保存为 `.npy`；
- SDK 放在实验目录的 `sdk/`，已被 Git 忽略。

若系统没有 `python3.5`，将硬件 YAML 中 `python_executable` 改成厂商环境的绝对路径。更换厂家时只需在 `hardware_devices.py` 添加并在 YAML 选择新 driver，不改模型后处理。

## 5. 常用标定 mask

```bash
python -m experiments.slm_calibration_bmp_generator \
  --config experiments/slm_calibration_bmp_generator/configs/slm_956.yaml
```

## 6. 测试

```bash
python -m pytest experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests -q
```
