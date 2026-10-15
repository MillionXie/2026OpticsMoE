# 双方100后仅残差接续：全部结束，三层达标、二/四层平台暂停

2026-10-15北京时间约19:39只读确认root all_depths_stopped_targets_or_plateau，CPU controller2446199、三个worker和全部GPU生产/测试子均实际退出，原始worker及producer rc0。全T18 GPU0/2/4/5项目0，A1000，无failure。所有185来源current/original与Git逐项一致；未重新训练/审核/测试。以下是最终选中同一模型的训练集评估、验证和测试，不是MixUp训练batch accuracy。选中轮次不是提前停止轮次。

| 深度 | 残差实际/额外轮数；选中E | 无残差实际轮数；选中E | 残差训练/验证/测试 | 无残差训练/验证/测试 | 总体测试差 | macro有/无；差 | 目标结果 |
|---|---:|---:|---:|---:|---:|---:|---|
| 2 | 325/225；E232 | 100；E98 | 76.7411/75.1592/78.9962% | 67.9369/64.9682/77.6869% | +1.3093pp | 75.0158/71.8949%；+3.1209pp | 4pp：平台暂停，未达标 |
| 3 | 125/25；E123 | 100；E100 | 90.9330/81.5287/87.4523% | 79.8949/75.1592/80.2510% | +7.2013pp | 84.6657/75.2939%；+9.3718pp | 6pp：达标停止 |
| 4 | 575/475；E488 | 100；E100 | 99.6058/87.2611/93.3988% | 84.7569/78.3439/83.9062% | +9.4926pp | 90.1104/80.6888%；+9.4216pp | 10pp：平台暂停，未达标 |

结论：二层总体差+1.3093未4；三层+7.2013达到6；四层+9.4926未10，不能四舍五入宣称10成功。四层575已完整，并非选E488就488停止；二层325完整选E232；三层125完整选E123。所有无残差仅完整100，二层按100内firstminimum选E98，三/四选E100，此后冻结且未再训练/重复测试。

## 停止原因与科学边界

无预设总epoch上限。每25轮是完整保存与独立审核的操作分段。四层最后LR2e-5，验证balancedNLL无.0002足够改善87轮；二层同LR触底、停滞93轮。二/四触发既定75新轮平台条件，保留全部结果暂停，不是人为设575/325轮上限。三层达到目标停止。不会因没达到4/10点自动恢复旧worker、改阈值或追加新方案。

有残差完整恢复E100 raw/EMA/Adam4800/scheduler100，MixUp.2->.3，其余设置保持；无优化器/EMA/scheduler重置、无prefix重播。额外残差预算不同于冻结100对照，这是人工要求的产品候选比较，不能当新同预算因果残差消融。原双方100公平基线另见[100轮完整报告](../baseline_final/README.md)。四层训练99.6058%、验证87.2611%，仍有明显泛化差，不能称过拟合已经解决；二层训练76.7411%、验证75.1592%，拟合不足仍在。

选模始终整个前缀首次最低验证EMA balancedNLL，不挑最大test epoch。测试反馈控制继续/停止，所以整体是test-informed development，不是独立未见泛化。二/三/四层累计测试不同selected参数digest分别7/2/12次（包含新100基线各一次）；相同参数真实复用已存结果，最后二/四段均未新增forward。绝不把多次开发测试称作只测试过一次。固定十类和原角色/ID/窗口/光学结构不变，不换类或削弱控制。test1833类支持315/80/678/680/20/16/13/9/14/8，macro和全逐类CSV同时报告；精确跨角色重复0并不证明近重复或主体独立，详见[数据与测试分数核查](../correction/README.md)。

## 真实身份与最终收据

run caltech10_equal100_residual_open_20261015，185训练来源b03fcdfa941669e53b6f67b4db8193f00972db85=current/original。基线caltech10_depth432_s17_equal100_correction_20261015的181首次发布e87、root metadata实际d61142f91605bd33fbbd5180541c4d488aeeb5ac，源SHA一致。报告HEAD不是训练身份。独占CPU admission SHA9bf1136c08332c643790c718dbbfb6d761d0735e0685d4ed364cf326e4e878fe，0models/forward/test；每个生产模型自己的恢复态微批概率/动态phase梯度准入实际passed。

| 深度 | 最后独占science SHA | selected测试result SHA | 独立保存测试audit SHA |
|---|---|---|---|
| L4 | a880d39b277494a2503f320be7f7de3dc49df3a7b68f0215de9d8a5eaa2c3afb | 4d20da3245da1072b4c88b409d9db0a689cb5453cfa9a4e73e8e0c2d9c0f0367 | 0f79d6375caa733049ee4a363164229af50f1ee1df95313a71303def6577e9b7 |
| L3 | d7f031f1c35c95f6d6f91eb735495e52aaa8986998afa268f071d27e22463797 | 0d66618da94e308c1c6261249961dae400e9b56053596b7ee0c90ecc84951dd5 | da15441ba8f71a3e94fe8f82425d02a2ed7db1167dafc4dc8ae6a03be9ad84ca |
| L2 | b63586e325276093f9e0c0a38504363bc73086dfa94609f298c89e4016b9bf4a | a5246021bf397a8c2bcd4ff6e65d7d685bb82c04491b7f93754e60fc1ac9ddc6 | 516a27abd21d4d2465acbb19087e3eaae288708626453fdaa51d57e41d53e2dc |

以上各science actualauditrc0与completed SHA逐项核一致，生产实际rc0/退出；CPU只读saved ID/label/argmax/CM/support/macro/NLL核验，0额外模型forward/test图。每个测试result/saved_audit/completed真SHA一致。CPU父实际退出，三个worker exit原始rc0/actual_exit，所有failure不存在，GPU项目0。所有历史来源、数据、PT、history、失败、claims和图永久保留；原始JSON、完整日志、逐样本预测只私有，不进Git。本次只新增本最终聚合MD/CSV并更新任务入口，不重跑任何旧作者或审核/传输helper。保留t18与每日09 ACTIVE，后续普通已结束和已通知结果静默。
