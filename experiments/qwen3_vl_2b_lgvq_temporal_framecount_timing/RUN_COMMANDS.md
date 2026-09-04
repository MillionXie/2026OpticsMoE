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
