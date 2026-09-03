# 主服务器：DC20 最终三卡实验

本页对应当前唯一正式口径：Spatial 4 帧一项、Temporal 16 帧两项。三个训练都是
正常光电模型；“去光”只在各自选中的同一个 checkpoint 上旁路光分支，不另训纯电子
模型。当前服务器从仓库根目录 `/DATA/DATA1/guest3/2026OpticsMoE` 执行。

## 1. 环境和固定路径

```bash
cd /DATA/DATA1/guest3/2026OpticsMoE
source /home/guest3/miniconda3/etc/profile.d/conda.sh
conda activate xml
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

固定资源：

```text
LGVQ: /DATA/DATA1/lixinyue/xyli/data/LGVQ
Qwen: /DATA/DATA1/lixinyue/code/adapt2026/video_inf_time/Qwen/Qwen3-VL-2B-Instruct
```

`CUDA_DEVICE_ORDER=PCI_BUS_ID` 不能省略；省略时 PyTorch 的逻辑 GPU 编号可能不等于
`nvidia-smi` 左侧编号。

## 2. 两份冻结前端缓存

Spatial 与 Temporal 的抽帧数不同，必须分别生成。缓存按 16 个视频保存一个可恢复
分片，中断后重复同一命令即可继续。

GPU 5：Spatial 4 帧。

```bash
CUDA_VISIBLE_DEVICES=5 python -u -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_qwen_front \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/spatial.yaml \
  --model-path /DATA/DATA1/lixinyue/code/adapt2026/video_inf_time/Qwen/Qwen3-VL-2B-Instruct \
  --batch-size 2 --chunk-rows 16 --device cuda
```

GPU 4：Temporal 16 帧。

```bash
CUDA_VISIBLE_DEVICES=4 python -u -m \
  experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.cache_qwen_front \
  --config experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/configs/release/temporal.yaml \
  --model-path /DATA/DATA1/lixinyue/code/adapt2026/video_inf_time/Qwen/Qwen3-VL-2B-Instruct \
  --batch-size 2 --chunk-rows 16 --device cuda
```

## 3. 三项训练

先确认 2、4、5 号卡空闲，然后运行启动器：

```bash
nvidia-smi
bash experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/server/launch_dc20_train.sh
```

| GPU | 任务 | 配置 | 推理结构 |
|---:|---|---|---|
| 5 | Spatial | `spatial.yaml` | 4 帧、2×2 lane、109×109、光 Top-2 |
| 4 | Temporal 基准 | `temporal.yaml` | 16 帧、4×4 lane、54×54、光 Top-2 |
| 2 | Temporal accuracy | `temporal_accuracy.yaml` | 与基准相同；只加宽最终电子读出头 |

训练时每个相位面的未调制功率比例从 `[0.20,0.35]` 均匀采样，测试固定为 `0.20`。
每 5 epoch 测完整 test、按最高 test SRCC 选权重，同时保存一个 phase-only `.pt`。

## 4. 监控

单次查看：

```bash
bash experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/server/monitor_dc20_runs.sh
```

每 30 秒刷新：

```bash
bash experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/server/monitor_dc20_runs.sh \
  --interval 30
```

`Ctrl+C` 只退出监控，不会停止后台训练。

## 5. 输出合同

每项任务独立生成：

```text
best_observed_test_checkpoint.pt
test_metrics_optical_on.json
test_metrics_optical_off.json
optical_contribution_same_checkpoint.json
train_history.json
phase_snapshots/epoch_0005.pt
phase_snapshots/epoch_0010.pt
...
phase_snapshots/manifest.json
phase_training_diagnostics.json
training_summary.json
```

相位 `.pt` 的精确字段、公式和读取代码见 [MASK_EVOLUTION.md](MASK_EVOLUTION.md)。
训练结束后用 `RUN_COMMANDS.md` 最后一条命令生成 Arial 7 pt 的 PNG/PDF 图和 CSV/JSON
汇总表。只有实测 `test_metrics_optical_on.json` 可用于声明是否达到 Temporal SRCC>0.8
或 Spatial SRCC≈0.64；配置目标不等同于实验结果。
