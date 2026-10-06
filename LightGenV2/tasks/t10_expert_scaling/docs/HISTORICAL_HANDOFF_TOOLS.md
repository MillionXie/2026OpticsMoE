# 历史Top-k绘图工具：单一源码入口

三个本地历史报告转换工具已纳入主线，原版保存在Git提交
`417a90397fb1eb7cf49f138b69c752ddc5a31afd`，不再需要维护另一份源码副本。
这不是替换实际服务器训练/评估实现，也不重新运行TEST模型。
2026-10-06只读检查服务器主目录及 `t10_corrected_20260918` 均无同名三文件，
因此是保留本地独有交付工具，不冒称它们是该服务器实际训练版本的一部分。

| 模块 | 所读历史证据 | 输出口径 |
| --- | --- | --- |
| `export_topk32_handoff` | 40组原训练报告、metadata/history、数据manifest | TRAIN/VAL、两seed均值与样本SD；验证选k，不当TEST成绩 |
| `update_topk32_test_handoff` | 事先锁定的16份TEST报告、40组训练表 | 只填锁定条目，未选配置TEST字段为空 |
| `update_topk32_full_test_handoff` | 全网格锁定的40份TEST报告 | 完整网格；若根据网格再选k，属于开发分析而非独立测试 |

从仓库根以模块形式运行，例如：

```text
python -m LightGenV2.tasks.t10_expert_scaling.update_topk32_full_test_handoff --package ORIGINAL_EVIDENCE_PACKAGE --output NEW_DERIVED_REPORT_DIRECTORY
```

三个模块现在均要求显式的新输出目录，拒绝原包内输出或已有目标，原包只读。
派生目录只放报告/CSV，不复制大数据、PT、原始预测或证据树，**不是自足的实验室包**。
需要回溯时仍使用传入的原证据包及其锁定清单。无自动覆盖/删除选项；失败产生的
新派生文件保留供诊断，工具不自动清理。

6项CPU测试覆盖覆盖拒绝、数学辅助函数与原版一致、合成40组和16组报告转换、
输入包逐SHA不变及未选TEST字段空值。没有运行原始数据、模型、GPU或硬件。
原报告、逐样本预测、best/last与全部测速继续保留；完整T10矩阵缺项仍以
[当前版本说明](../CURRENT_VERSION_20261003.md)为准，不用这些工具补造结果。
