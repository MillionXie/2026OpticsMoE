# Qwen3-VL-2B 448px quality-token baseline

这是 T06 的冻结 Qwen 纯电子 baseline，不是光电模型。每个视频均匀抽取 4、9 或 16 帧，
每帧先做固定 65% 中心裁剪，再缩放为 **448×448**。Qwen3-VL-2B-Instruct 全部冻结；
方案一只训练 `Linear(2048,1)`，方案二只训练五个质量词对应的 `5×2048=10,240`
个输出参数，然后以 softmax 概率加权五个质量等级得到 Temporal MOS。

## 正式结果

GPU 为 RTX 5090 D，batch size 为 1。每个方案/帧数组合均为独立进程：模型加载一次，
不做显式 warmup，不做 test 前推理，第一条包含在全部 558 个测试视频的统计中。主时间
使用同步 CUDA event，从 Vision Transformer block 0 输入开始，到相应方案的标量分数
在 GPU 上就绪；MP4 解码、裁剪/resize、tokenizer、H2D 和模型加载分栏保存。

| 帧数 | 方案 | mean ms | median ms | P95 ms | SRCC | KRCC | PLCC | RMSE | MAE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | Linear 评分头 | 56.760 | 63.062 | 76.723 | 0.7415 | 0.5450 | 0.7537 | 9.110 | 7.405 |
| 4 | 五质量词 | 65.433 | 71.508 | 79.927 | **0.7693** | **0.5678** | **0.7797** | **8.682** | **6.735** |
| 9 | Linear 评分头 | 66.007 | 65.445 | 76.663 | 0.7514 | 0.5546 | 0.7586 | 8.974 | 7.276 |
| 9 | 五质量词 | 65.133 | 63.946 | 75.914 | **0.7745** | **0.5771** | **0.7829** | **8.690** | **6.468** |
| 16 | Linear 评分头 | 85.092 | 84.842 | 90.771 | 0.7636 | 0.5635 | 0.7696 | 8.792 | 7.125 |
| 16 | 五质量词 | 86.078 | 85.963 | 91.724 | **0.7787** | **0.5814** | **0.7925** | **8.574** | **6.298** |

五质量词方案在三种帧数下均有更高相关性和更低误差。9 帧两方案均值相差不足 1 ms，
不能解释为质量词方案结构上更快；独立 dataset-pass 会受到 GPU 动态频率影响，应同时
查看 median、P95 和逐视频分布。

## 输入 token 几何

当前模型配置为 16×16 patch、2×2 spatial merger、temporal patch size 2。448 可被
有效空间步长 32 整除：

| 输入帧 | Vision block 0 输入，1024维 | merger 后视觉 token，2048维 |
|---:|---:|---:|
| 4 | 1,568 | 392 |
| 9 | 3,920 | 980 |
| 16 | 6,272 | 1,568 |

9 帧在 temporal patch 阶段补齐为 5 个帧对。merger 后的视觉 token 与 Temporal prompt
的文本 token 一起进入 28 层语言模型。

## 图和证据

- `framecount_latency_performance.*`：模型时间、P95、性能和预处理成本。
- `scheme2_prediction_scatter.*`：五质量词方案的 558 视频目标—预测散点。
- `latency_ecdf.*`：两方案逐视频延迟的完整经验分布；对数横轴保留了首条推理离群值。
- `result.json`：论文图对应的机器可读数值、输入几何和源证据 SHA256。

逐视频 CSV、原始日志和每组完整报告不提交 Git，保存在同名 T06 run 及实验数据包中：

```text
LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/
  qwen3vl_quality_tokens_r448_dataset_once_5090d_20260906/
```

该 run 是既有 5090D 正式测试证据的只读导入；复现入口、448 profile 和训练/计时代码已
迁入 T06。源数据包的 SHA256 为
`769fe41a5fd73d58ad8a06d6df7fdff0155cadac8f8d539973f646efcf824a0b`。
