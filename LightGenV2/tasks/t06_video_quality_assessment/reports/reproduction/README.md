# t06_video_quality_assessment 复现说明入口

- [DeepSeek-VL2-Tiny 冻结主干 LGVQ baseline（2026-09-22）](DEEPSEEK_VL2_TINY_LGVQ_20260922.md)

[Baseline 复现说明](BASELINE_METHODS.md)：按最新表格指标核对的论文式技术正文（2026-09-15）。

本目录集中保存baseline及主方法的可复现性证据。2026-09-14已完成固定Temporal权重的六层SHS实测：
558个测试视频、210张正式CCD，SRCC 0.7977138739；详见 [本轮实测报告](TEMPORAL_SHS_20260914.md)。
这是固定权重硬件评估，不是重新训练，也不代表其他baseline已经复现。
同日空间六层实测完成：558视频、3348张有效CCD，SRCC 0.5786364901；第4/5层使用1600μs，
其余400μs，明确保存继承来源。详见[空间实测报告](SPATIAL_SHS_20260914.md)，未达到仿真0.6711。
当前任务结构和已有结果见 [任务README](../../README.md)。每项完成声明须有对应证据。

后续每个正式结果需在这里记录：

1. baseline定义，冻结/训练的参数，预处理、输出头与主方法差异。
2. 原始数据版本、train/test清单与SHA256、标签生成方法、指标与选模口径。
3. 源码commit、完整命令、依赖环境、模型及checkpoint的SHA256和获取位置。
4. 固定权重复评与从头重训分别报告；标明样本数、seed、运行ID、结果和误差。
5. 速度/能耗的硬件、计时边界、功率积分口径；未测的不得填估计值冒充实测。

原始日志及逐样本结果留在本任务runs，文档只引用。参考 [SALICON复现说明](../../../t03_saliency/reports/reproduction/README.md)。
