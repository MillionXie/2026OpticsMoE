# t02_keypoint_detection 复现说明入口

[alpha≥0.4热图蒸馏](HEATMAP_DISTILLATION.md)：0.72793来源、训练专用教师缓存、梯度检查、完整命令。

[最后global相位替换为固定噪声](GLOBAL_NOISE.md)：干预范围、5个seed、低/高alpha的独立评估与命令。

[两级alpha≥0.5硬约束续训](ALPHA50.md)：初始化、分阶段训练、迁移评估合同及命令。

[Qwen 轻量读出头预算对照](QWEN_HEAD_BUDGET.md)：完整冻结视觉主干，头由110.3万缩为13.84万参数，保留原baseline。

[同结构分阶段续训与架构对比](STAGED_REFINEMENT.md)：从核实的 0.7305 权重继续训练，单列训练策略与电子结构改动。

[Baseline????????](BASELINE_METHODS.md)?????baseline??????????????????????2026-09-09?

本目录集中保存baseline及主方法的可复现性证据；本次只建立入口，**尚未进行本任务的新一轮复现**。
当前任务结构和已有结果见 [任务README](../../README.md)。不得因为存在本文件就声称已复现。

后续每个正式结果需在这里记录：

1. baseline定义，冻结/训练的参数，预处理、输出头与主方法差异。
2. 原始数据版本、train/test清单与SHA256、标签生成方法、指标与选模口径。
3. 源码commit、完整命令、依赖环境、模型及checkpoint的SHA256和获取位置。
4. 固定权重复评与从头重训分别报告；标明样本数、seed、运行ID、结果和误差。
5. 速度/能耗的硬件、计时边界、功率积分口径；未测的不得填估计值冒充实测。

原始日志及逐样本结果留在本任务runs，文档只引用。参考 [SALICON复现说明](../../../t03_saliency/reports/reproduction/README.md)。
# 新增：alpha≥0.4

见[ALPHA40.md](ALPHA40.md)：电子结构/参数量不变，PCK≥0.73为待验证目标。
