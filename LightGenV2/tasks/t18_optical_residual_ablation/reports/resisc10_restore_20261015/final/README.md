# 原十类一致接续140轮：最终核验结果

本轮已经完整完成E101–140，maximum_epochs结束；CPU queue1816946与GPU2 producer1817647实际均退出，producer原始记录returncode0/actual_exit true。root状态complete_science_passed，自己的独占保存科学首次真实passed，审核completed与原始audit SHA一致；无failure。GPU2/5项目0、A1000、test0。

| 比较对象 | 选中轮次 | 训练准确率 | 验证准确率 | 验证balancedNLL |
|---|---:|---:|---:|---:|
| 原有残差100轮 | 100 | 87.6607% | 78.1429% | 0.7785573853 |
| 本次有残差140轮 | 139 | 89.8750% | 78.7143% | 0.7545565714 |
| 无残差历史前100轮参考 | 100 | 83.3393% | 76.2857% | 0.8483807700 |
| 无残差历史完整120轮选中参考 | 113 | 84.8571% | 77.1429% | 0.8206857000 |

按事先固定的首次最低验证EMA balancedNLL在完整140轮中选择E139；选择139不是提前停止，E140实际已完成，其末轮验证为78.2857%。不改为最高accuracy轮。

相较父E100，选中验证提高0.5714个百分点（700张净多4张正确），NLL从0.7785573853降至0.7545565714；仍未达到80%。本次训练/验证差11.1607个百分点，训练提升大于验证，泛化差仍明显；不能宣称过拟合已解决，也不能将验证结果当测试结果。

对无残差前100轮历史数值差+2.4286个百分点，对旧完整120轮选中参考差+1.5714个百分点。这些仅为明确披露的历史参考：本次残差获得额外40轮，没有新增同预算无残差对照，因此不是新的匹配预算残差因果增益。原同100轮差仍是+1.8571个百分点。

## 本轮身份和边界

run `resisc10_original_rho03_steady40_mixup02_20261015`，stage steady40，训练来源`6ddd95fa90691714ef9d0e5ba96a78f38af41e05`。原十类bridge/desert/harbor/intersection/island/lake/meadow/tennis_court/thermal_power_station/wetland、原split/网络/输入100/8bit及光学rho.3保持，无可训练电子分类支路。

完整恢复父complete100的raw/EMA/Adam35000/scheduler100/LR.0005；保持MixUp.2和原scheduler，无额外E101手动LR减半，无重播/重置/early。新增40轮实际拟合1279.13383秒，低于1800秒安全上限，累计Adam49000步。fit_result.seconds累计包含保留前100轮，为4626.90259秒，不能误当本次新增耗时。

自己的独占科学核验prefix100、来源、模型形状/初始化、rawEMA/11梯度、订单与affineMixUp.2、LR replay、预算、保存ID/label/argmax/CM/support/NLL和firstminimum；CPU1模型仅shape/seed，0forward/test。没有重复执行审核。

- audit SHA `139891ea639703b8c7e02d23a95db04df626570d582bbb1f5889cabd3071bb7d`
- selected PT SHA `12c1893245b4f86930232658e0e5c70b61adb876ad2642fe33f364c3f7a5de60`
- history SHA `7fdb729eeef018c9174c4578721c4558e2869f3bad54fcbb31ba7f602b22b6fd`

逐类聚合见[class_recall.csv](class_recall.csv)，每类train560/val70。逐样本记录、完整JSON/日志及PT保留私有，不进Git。

此次一次有界授权已结束，不自动扩预算/重试/新候选/换类/电子分支/无残差补训/A100/正式六组/native/lab或测试。旧原十类E100一次测试77.8571%是旧权重结果，本次E139未测试，不能套用或自动再测。保留t18与09晨报ACTIVE，后续普通完成状态和已通知旧失败静默。
