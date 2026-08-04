# Hardware SDK commands

以下命令均从仓库根目录 `2026OpticsMoE/` 执行。

## 1. 生成常用振幅/相位标定 BMP

```bash
python -m experiments.hardware_sdk.slm_calibration_bmp_generator \
  --config experiments/hardware_sdk/slm_calibration_bmp_generator/configs/slm_956.yaml
```

输出位于 `experiments/hardware_sdk/slm_calibration_bmp_generator/generated/slm956_calibration/`。

## 2. 振幅 SLM + CCD 数字顺序 demo

只生成 0–9 的 MNIST 振幅 BMP，不连接设备：

```bash
python -m experiments.hardware_sdk.amplitude_camera_demo \
  --config experiments/hardware_sdk/configs/amplitude_camera_digit_demo.yaml \
  --generate-only
```

连接 HOLOEYE 振幅 SLM 和 DVP 相机后播放、拍摄：

```bash
python -m experiments.hardware_sdk.amplitude_camera_demo \
  --config experiments/hardware_sdk/configs/amplitude_camera_digit_demo.yaml
```

查看 `experiments/hardware_sdk/demo_outputs/amplitude_camera_digits/`：

```text
input_bmp/                  # 000_digit_0.bmp ... 009_digit_9.bmp
ccd_captured/               # 同序号无损原始帧
capture_order.csv           # 实际命令/拍摄顺序和时间
resolved_devices.json       # 相机与 SLM 实际设置
input_vs_capture_order.png  # 左输入、右实拍的顺序总览
```

## 3. 相位 SLM demo

先检查尺寸、上下翻转和 WFC 合成结果：

```bash
python -m experiments.hardware_sdk.phase_slm_demo \
  --config experiments/hardware_sdk/configs/phase_slm_demo.yaml \
  --dry-run
```

在连接 Meadowlark Blink 1920 HDMI 的 Windows 控制机上真实播放：

```powershell
python -m experiments.hardware_sdk.phase_slm_demo `
  --config experiments/hardware_sdk/configs/phase_slm_demo.yaml
```

主 Grocery 四平面流程仍不会自动操作相位 SLM。

## 4. Grocery 四平面实物流程

MoE4 仿真回放检查：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_automation \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --simulate --yes --sample-limit 4 \
  --session-dir experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_simulation_smoke
```

真实采集：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval.hardware_automation \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest_hardware.yaml \
  --session-dir experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/hardware_sessions/moe4_physical_001
```

中断后续跑时增加 `--skip-prepare`，已有 capture 会按 basename 跳过。

## 5. 测试

```bash
pytest experiments/hardware_sdk/tests \
  experiments/hardware_sdk/slm_calibration_bmp_generator/tests \
  experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests -q
```
