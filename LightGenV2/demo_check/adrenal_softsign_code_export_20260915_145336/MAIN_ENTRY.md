# 历史病理结构／对照源码：不是新任务入口

2026-10-07补齐主线中的原导出源码和配置，原字节与服务器导出MANIFEST、既有Git历史
逐项核验。T10复现说明引用这里的model／models／prompt作结构核查，T16仅使用此前已
登记的optical_reference依赖。它不是T11或T16的另一套最终版，不用其结果覆盖现用模型。

原PROTOCOL、配置和run.sh保持原样，包含旧服务器绝对路径、CUDA设备0及完整训练流程。
这些是历史身份，不是当前机器可直接执行的命令；本轮没有训练、评估PT或打开设备。
不要依据旧导出说明自动新建独立工程或工作树。日常入口仍是仓库根START_HERE与对应任务。

原数据、结果、环境记录、字体、PT及导出MANIFEST保持原位置，未删除或代替。
原环境/资产没有据此全部补齐；Python语法检查和源码身份不代表科学实验已重新复现。
原说明README_EXPORT_zh.md继续保留，旧数据路径和历史重跑要求按原文审阅。

逐文件来源：[身份收据](../../../maintenance/storage/ADRENAL_HISTORICAL_SOURCE_ADOPTION_20261007.json)。
