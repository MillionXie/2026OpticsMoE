# 冻结十类四层E149配对：各一次仿真测试

用户明确授权当前选好有/无残差模型各跑测试。双方父训练各150完整、独占science passed；firstminimum验证EMA balancedNLL均选E149，测试前已冻结双方PT/来源/类别/划分，不按test再选模型。1833张相同ID测试，两生产子2291195/2291261实际rc0退出，CPU独立核保存预测ID/label/argmax/CM/support/macro/NLL，0第二次forward/test图。test_evaluations_per_model1/test_used_for_selectionfalse，completed真实SHA匹配，无failure。

| 指标 | 有光学残差 | 无光学残差 | 差值 |
|---|---:|---:|---:|
| 测试准确率 | 92.7987% | 88.3797% | +4.4190pp |
| 测试宏平均召回 | 89.7096% | 86.2398% | +3.4698pp |
| 测试balanced NLL | 0.5380722493 | 0.7531026623 | 越低越好 |
| 正确/总数 | 1701/1833 | 1620/1833 | 净多81 |

准确率超80，但差距4.4190pp未达到用户10pp目标；macro差3.4698pp也未10。原验证86.6242/78.9809、差7.6433不能冒充测试差。测试支持不均，人脸/摩托/飞机占1673/1833，必须同时看macro和全部逐类；不能只总体92.80宣传每类92.80。旧Caltech测试已观察、本十类验证知情筛选，是开发仿真结果，不是未经观察的泛化/原20或全101验收。

| 类别 | 测试支持 | 有残差recall | 无残差recall | 差pp |
|---|---:|---:|---:|---:|
| Faces | 315 | 90.79% | 82.86% | +7.94 |
| Leopards | 80 | 88.75% | 78.75% | +10.00 |
| Motorbikes | 678 | 92.48% | 93.66% | -1.18 |
| airplanes | 680 | 95.29% | 87.35% | +7.94 |
| bonsai | 20 | 85.00% | 75.00% | +10.00 |
| grand_piano | 16 | 75.00% | 75.00% | +0.00 |
| revolver | 13 | 76.92% | 76.92% | +0.00 |
| schooner | 9 | 100.00% | 100.00% | +0.00 |
| trilobite | 14 | 92.86% | 92.86% | +0.00 |
| yin_yang | 8 | 100.00% | 100.00% | +0.00 |

父run caltech10_gap_s17_current4_pair150_gpu425_20261015，169训练Git259cdcfdec6edc9b8ca29f9c7ddf18fb3668cffa，父SCI617825b1c2d628b7a2cefdd13874d4c4ca3e468ed16338def5561b3d723051db。新测试173来源25fd39981fdba0a686cf8df977add484791327c7，不改原169或旧测试工具。GPU2/4两子、一卡一子、2GiB+800MiB，A1000；测试前wrongrun拒绝准入SHA d29266a8e127245a2863b1b15950e04fb21956af72bebc02b07c72b001afc539只工具准入，不是测试成绩。

输出父run/paired_once_test_20261015，CPU2290327已结束后只读核；paired_resultSHA eb7776325d0ad0366bd4286c7fa04c218dcda497cc212977509510c36f9fb779。
残差PT8f7037445a77defbec297e5d24f8abfb639a89565e7992216976314a971c6815、独立保存测试auditSHA3401940457af323c81a042f1fad33cd5c918a05a895fa718a0d4f632062b6e9d。
对照PT95886e6af086bafb2915baf050cf37e2e3241f4ec503f1d539b5f09d069970ea、auditSHAb8a4f38af0236ab92073a28854596262fc384ee0b579ea67843d975bd9adae1a。

当前两个模型的这次测试已结束，绝不重测/删除claim/按test换epoch或类别。新人工允许一次残差MixUp.4有界续训（先固定方案、仅val选模）和随后三/二层配对；新候选冻好后各最多一次test，不无限追10pp，不削弱对照。有残差额外预算后的比较要明确标注额外优化，不能替代本同150消融。原数据/PT/history/source/claims与外国任务保持，rawJSON/预测逐样本ID/完整日志私有；本报告仅聚合MD/CSV。
