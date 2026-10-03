# t02_keypoint_detection 复现说明入口

[2026-10-03完整版本身份](IDENTITIES_20261003.md)：区分原DC20公平对照、73.48%历史交付、alpha40蒸馏及个人图片pilot；入口按固定服务器源码收敛，不直接替换PT。

后续运行记录：[分阶段微调](STAGED_REFINEMENT.md)、[alpha40](ALPHA40.md)、
[热图蒸馏](HEATMAP_DISTILLATION.md)、[个人照片](PERSONAL_DOMAIN.md)、
[alpha50](ALPHA50.md)、[全局噪声](GLOBAL_NOISE.md)、[读出头预算](QWEN_HEAD_BUDGET.md)。
这些是历史协议与结果，旧分支/工作树说明不覆盖仓库当前单main协作规则。

[Baseline 复现说明](BASELINE_METHODS.md)：按最新表格指标核对的论文式技术正文（2026-09-15）。
[Baseline 代码交接包](../../../../reports/20260915_baseline_methods/CODE_HANDOFF.md)：对应版本源码、配置、运行入口及哈希清单。

本目录集中保存baseline及主方法的可复现性证据；本次只建立入口，**尚未进行本任务的新一轮复现**。
当前任务结构和已有结果见 [任务README](../../README.md)。不得因为存在本文件就声称已复现。

后续每个正式结果需在这里记录：

1. baseline定义，冻结/训练的参数，预处理、输出头与主方法差异。
2. 原始数据版本、train/test清单与SHA256、标签生成方法、指标与选模口径。
3. 源码commit、完整命令、依赖环境、模型及checkpoint的SHA256和获取位置。
4. 固定权重复评与从头重训分别报告；标明样本数、seed、运行ID、结果和误差。
5. 速度/能耗的硬件、计时边界、功率积分口径；未测的不得填估计值冒充实测。

原始日志及逐样本结果留在本任务runs，文档只引用。参考 [SALICON复现说明](../../../t03_saliency/reports/reproduction/README.md)。
