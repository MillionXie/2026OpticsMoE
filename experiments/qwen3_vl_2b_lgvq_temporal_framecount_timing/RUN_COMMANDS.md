# 5090D 复现实验命令

在工程根目录执行：

```bash
cd /root/qwen3_vl_2b_lgvq_temporal_framecount_timing
python benchmark.py --config config.json 2>&1 | tee benchmark.log
```

输出位于：

```text
/root/autodl-tmp/qwen3vl_lgvq_temporal_framecount_timing/
```

重点文件：

- `RESULTS.md`：直接阅读的速度表；
- `timing_summary.json`：包含均值、中位数、P05、P95、显存、输入形状和软件环境；
- `timing_summary.csv`：便于绘图的汇总表；
- `per_video_measurements.csv`：每个视频每次测量的原始耗时；
- `run_identity.json`：配置、manifest 哈希和本次固定样本 ID。

这个脚本只使用 Temporal prompt。表中的总耗时表示一个视频从打开 MP4、抽完 4/9/16
帧、裁剪缩放、Qwen processor、GPU 传输、完整 Qwen 前向到线性层连续值回到 CPU 的总和。
模型加载时间不混入逐视频耗时，而是在 JSON 中单列。

正式测试覆盖全部 558 个 test 视频，每个视频在三种帧数下各运行一次。

## 36帧及条件49帧扩展

先完整测试36帧。如果36帧的平均端到端总耗时不超过910 ms/video，脚本才继续测试
49帧；否则严格按预设规则跳过49帧：

```bash
cd /root/qwen3_vl_2b_lgvq_temporal_framecount_timing
python benchmark_extended.py --config extended_config.json 2>&1 | tee benchmark_extended.log
```

结果为`RESULTS_EXTENDED.md`、`timing_summary_extended.json/csv`以及对应帧数的逐视频CSV。

第一轮正式运行中，36 帧均值为 1133.494 ms/video，超过 910 ms/video，因此当时按
预设条件跳过 49 帧。

## 明确追加的25帧与49帧

用户随后明确要求补测 49 帧并增加 25 帧。复现命令为：

```bash
cd /root/qwen3_vl_2b_lgvq_temporal_framecount_timing
python benchmark_additional.py --config additional_config.json 2>&1 | tee benchmark_additional.log
```

该命令不重跑 4/9/16/36 帧，只测全部 558 个 test 视频的 25/49 帧。完整结果为
`results/RESULTS_ALL.md`、`timing_summary_all.json/csv`、
`per_video_measurements_25.csv` 和 `per_video_measurements_49.csv`。结论解释见
`TIMING_REPORT.md`。
