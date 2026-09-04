# LGVQ Temporal 六档性能测试

正式时间和正式性能均在同一台 **RTX 5090 D** 上生成。实验室服务器只保存最终工程副本，
不得把实验室服务器上的耗时或中断任务结果混入正式表格。六档使用相同的固定
2250/558 train/test 划分，Qwen 完全冻结，只训练共享的 `Linear(2048,1)` 读出头
50 epoch。测试集每个 epoch 都评估，保留观测到的最高
test mean(Spatial SRCC, Temporal SRCC)；最终帧数比较提取 Temporal 的
SRCC/KRCC/PLCC/RMSE/MAE。

采样位置、65% 中心裁剪、448×448 输入和 Temporal prompt 均与测速实验一致。性能缓存
允许顺序读取短视频及并行解码，因为这不改变任何选中帧或模型输入；缓存构建时间不作为
推理速度证据。正式速度仍以 `results/per_video_measurements*.csv` 为准。

## 5090D 预检查

```bash
cd /root/qwen_temporal_performance_workspace
/root/miniconda3/envs/qwen_vqa/bin/python -m \
  experiments.qwen3_vl_2b_lgvq_temporal_framecount_timing.performance_driver audit \
  --config experiments/qwen3_vl_2b_lgvq_temporal_framecount_timing/performance_config.5090.json
```

## 串行运行

一次只运行一个帧数。先完成当前命令，再把 `--frames` 依次改为
36、25、16、9、4；正式运行顺序不会改变指标含义。

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/qwen_vqa/bin/python -m \
  experiments.qwen3_vl_2b_lgvq_temporal_framecount_timing.performance_driver all \
  --config experiments/qwen3_vl_2b_lgvq_temporal_framecount_timing/performance_config.5090.json \
  --frames 49
```

不同帧数写入不同目录并支持断点续建特征缓存，但正式实验不并行跨 GPU。

## 六档完成后汇总

```bash
/root/miniconda3/envs/qwen_vqa/bin/python -m \
  experiments.qwen3_vl_2b_lgvq_temporal_framecount_timing.performance_driver report \
  --config experiments/qwen3_vl_2b_lgvq_temporal_framecount_timing/performance_config.5090.json
```

输出目录：

```text
/root/autodl-tmp/qwen3vl_lgvq_temporal_framecount_timing_performance/
```

汇总完成后才把小型正式证据（指标、预测、训练历史、图表和报告）复制到实验室服务器：

```text
/DATA/DATA1/guest3/2026OpticsMoE/experiments/
qwen3_vl_2b_lgvq_temporal_framecount_timing/
```

`qwen_prompt_features.pt` 和分片缓存体积较大，只留在 5090D 的临时盘，不复制到实验室
服务器，也不提交 Git。
