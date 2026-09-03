# 主服务器：缓存一次，四张 GPU 并行训练

本页只负责服务器启动和监控，不改变模型、配置或训练逻辑。四个正式任务是：

1. Spatial 正式鲁棒性；
2. Spatial 正式鲁棒性、独立 seed 43；
3. Spatial 强鲁棒性候选；
4. Temporal 正式鲁棒性。

四者均为正常光电训练。程序最终会用各自选中的同一个光电 checkpoint 再执行一次
`optical_off` 旁路推理，用于衡量光学贡献；不会启动、也不会训练一个独立的纯电子模型。

## 前置条件

- 从仓库根目录操作；
- 已激活包含 PyTorch、Transformers、OpenCV 等依赖的环境；
- 数据集固定为 `/DATA/DATA1/lixinyue/xyli/data/LGVQ`；
- 配置引用的训练集 soft-target 文件必须已经存在；
- 必须知道服务器本地 `Qwen3-VL-2B-Instruct` 目录。启动器不会猜路径，也不会联网下载；
- 需要四张空闲 GPU。默认使用 0、1、3、4，可通过参数修改。

先检查当前 GPU：

```bash
nvidia-smi
```

## 一条命令启动

下例中的 Qwen 路径必须替换成服务器上的真实绝对路径：

```bash
bash experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/server/launch_four_runs.sh \
  --qwen-model /ABSOLUTE/PATH/TO/Qwen3-VL-2B-Instruct \
  --cache-gpu 0 \
  --spatial-gpu 0 \
  --spatial-seed43-gpu 1 \
  --robust-gpu 3 \
  --temporal-gpu 4
```

执行顺序是固定的：

1. GPU 0 生成/恢复 16 帧共享 Qwen Vision front 和 Spatial prompt cache；
2. GPU 0 复用上述 Vision cache，只补 Temporal prompt cache；
3. 串行检查 Spatial seed 42、Spatial seed 43、Spatial 强鲁棒、Temporal 四份
   preflight，四份都必须为 `ready`；
4. 预先确认四张训练 GPU 均空闲，然后用 `nohup` 同时启动四项训练。

缓存结束后，缓存 GPU 可以立即用于 Spatial，因此默认 `cache-gpu` 和 `spatial-gpu` 都为 0。
四个训练 GPU 参数则必须互不相同。

启动器的安全限制：

- 任意被本工程 PID 文件记录的旧进程仍存活时拒绝重复启动；
- 任一 GPU 已有计算进程时，在启动任何任务之前拒绝；
- 四个正式输出目录中任一个已有 checkpoint、训练日志或 summary 时拒绝覆盖；
- 数据集路径、soft target、配置和缓存/preflight 有一项不符合即停止；
- 四个任务若有一个启动后立即退出，会明确报错并保留日志。

如果确实要重跑某个已有实验，请先把该配置对应的整个旧输出目录改名归档，不要只删除某个
checkpoint 来绕过检查。

## 监控

单次查看：

```bash
bash experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/server/monitor_four_runs.sh
```

每 30 秒刷新一次：

```bash
bash experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/server/monitor_four_runs.sh \
  --interval 30
```

按 `Ctrl+C` 只会退出监控，不会终止训练。启动合同、缓存日志、preflight 日志、四个 PID
和四个训练日志位于：

```text
experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/server_jobs/<UTC时间>/
```

正式训练结果仍分别写入四份配置各自的 `output_dir`。重点文件为：

```text
best_observed_test_checkpoint.pt
metrics_best_observed_test_optical_on.json
test_metrics_optical_on.json
test_metrics_optical_off.json
optical_contribution_same_checkpoint.json
phase_training_diagnostics.json
training_summary.json
```

其中 `test_metrics_optical_off.json` 不是另训的纯电子模型，而是最佳光电权重的四层光学旁路
结果，因此能够公平回答“拿掉光以后下降多少”。
