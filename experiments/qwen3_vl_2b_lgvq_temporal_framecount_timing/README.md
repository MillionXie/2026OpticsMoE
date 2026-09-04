# Qwen3-VL LGVQ Temporal 4/9/16-frame timing

本工程只回答一个问题：在相同的完整 Qwen3-VL 时序 baseline 中，一个视频分别抽取
4、9、16 帧时，从原始 MP4 到一个连续质量分数总共需要多久。

计时边界：

```text
打开 MP4
→ 对每个目标帧执行随机 seek + read
→ 65% 中心裁剪并缩放到 448×448
→ 官方 Qwen processor 与 Temporal prompt tokenizer
→ CPU 到 GPU
→ 完整 Qwen Vision tower + main merger + Language backbone
→ 最后有效 token
→ Linear(2048,1)
→ 连续值回到 CPU
```

这里没有双 prompt。一个视频只使用一次 Temporal prompt，只产生一个连续输出。

为了遵守“baseline 不做额外优化”：

- batch size 固定为 1 个视频；
- 沿用原三个 LGVQ 工程逐采样点随机 seek 的解码方式；
- 不使用顺序解码、多线程解码、帧缓存或 Qwen 特征缓存；
- 不使用量化、`torch.compile` 或手工 FlashAttention override；
- 完整 558 个 test 视频全部参加测试，每个视频在 4/9/16 三档各测一次；
- 558 个视频分成三组并轮换 4/9/16 的测试顺序，避免某个帧数永远首先运行；
- shape warmup 与模型加载单独记录，不混入逐视频耗时。

抽帧和三个旧工程的关系见 [FRAME_CONTRACT_AUDIT.md](FRAME_CONTRACT_AUDIT.md)，复现命令见
[RUN_COMMANDS.md](RUN_COMMANDS.md)，正式结论见 [EXPERIMENT_RECORD.md](EXPERIMENT_RECORD.md)。结果证据保存在本目录 `results/`，服务器原始证据
同时保存在 `/root/autodl-tmp/qwen3vl_lgvq_temporal_framecount_timing/`。
