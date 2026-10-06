# remote_staging baseline补丁：用途和主线取舍

2026-10-06；比较基准main `fc2e8f556d200aeffdfd3cf12e6c0f2cd52f9783`。
原补丁目录和其所有报告／数据原位保留。本轮不采集、不训练、不测GPU、不改正式PT。

| 原文件 | 原SHA256 | 核验及处理 |
| --- | --- | --- |
| abo_backbone_baselines.py | e3b98c42ec9fbe0575c31a12631b7767794041c7637a14578c82e8c92a457961 | 与T08同名主线源码逐字节一致；三方向候选集不同，不整合成同一结果 |
| test_abo_backbone_baselines.py | 6f68e1c6a1a7223290e1dc82e42a612d783cbb69393dfe9834117debf837dbf5 | 与T08测试逐字节一致 |
| frozen_backbone_dense_tasks.py | 5d2feeb5d83c6ff56adbf6241ab8d6af876d4b92e3ead4548a8a7342780b3682 | 仅两处OpenMoji导入改为baseline_shared_readout／baseline_settings；归一这两处后整文件AST相同。保留主线专用baseline导入，不退回可能随现用光模型变化的共享入口 |
| aligned_baseline.py | 77f17a40e162f39b5a0ed1f1cba8e41e30b3b345317676a77e0e3095c3eabf33 | 本地补丁增固定LR与显式间隔，主线默认为服务器分阶段版本；按显式可选参数补齐，不覆盖默认协议 |

## SALICON两种训练身份

主线原默认：配置中的adapter／decoder不同LR、调用原staged_epoch；
epoch1、配置指定间隔、末轮评估。原50轮报告仍绑定这一身份，不能改成固定LR。

本机旧补丁：固定LR时不执行staged_epoch，显式间隔只在倍数轮与末轮评估。
主线只有显式给参数才切换对应行为；正有限LR、正整数间隔在创建run或加载模型前检查。
head类AST与原main逐项相同，冻结主干、decoder规格、GT训练函数、指标与TEST选模口径未改。
run manifest现在记录真实LR策略、间隔和epoch1策略，而不是硬编码每5轮。

12项纯CPU协议测试通过：默认日程委托／评估轮次、历史间隔、固定LR不被日程覆盖、
非法LR和间隔拒绝。不是完整数据复现。
本机现有Torch CPU环境运行T03模块测试时因缺scipy而在收集阶段停止；未安装依赖或启动Qwen。
随后服务器实际main `5254a4e58136fb336b37d86e0fdb78eb21229ad9` 在既有xml环境、
CUDA不可见条件下通过25项CPU测试（7.50s）：上述12项协议检查，加
`test_alpha_and_aligned_head.py`、`test_training_support.py`。tracked状态保持不变。
这是选中模块的CPU回归，不是全部T03测试、训练重跑、完整数据评估或设备验证。
此前296项测试仍为对应原版本证据，不冒充本次新增选项的完整测试。

该目录还含ABO和LSP／SALICON／OpenMoji baseline结果说明，所以不作为整目录删除候选。
LSP异常低分仍按原报告诊断口径保留，不能因其它任务成绩较高把它改为正式达标结果。
