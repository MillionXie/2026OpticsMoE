# 固定Caltech十类：四层正则与三/二层完整配对最终结果

本次人工范围已全部真实结束：原四层E149两模型各一次测试；残差E151-200一次MixUp.4正则完整+自己科学；三/二层四臂各完整150+自己独占科学+val-selected模型各一次测试。CPU campaign2314182与所有GPU实际退出，producer原始rc0，rootstate complete_science_and_frozen_tests_passed，globalGPU项目0/A1000，无failure。不是计划/排队/临时验证。

所有测试基于相同1833ID、固定十类、无电子分类器。每模型按首次最低验证EMA balancedNLL冻结，未挑最高accuracy/gap或用test选epoch/类别。表内val为有/无，四层正则后参数未变复用原已测E149而不是再测。

| 深度 | 有/无选中E | 有/无验证 | 有残差test | 无残差test | test差 | macro差 |
|---|---:|---:|---:|---:|---:|---:|
| 4层 | 149/149 | 86.62/78.98% | 92.7987% | 88.3797% | +4.4190pp | +3.4698pp |
| 3层 | 149/149 | 82.17/75.80% | 88.8707% | 82.7605% | +6.1102pp | +10.4842pp |
| 2层 | 146/149 | 71.34/68.79% | 79.0507% | 79.4326% | -0.3819pp | -1.1844pp |

**本轮四/三/二层都未达到测试10个百分点目标。**无论是否达到，真实正负结果全部保留，不削弱对照/换类别/用test继续搜索或扩预算。三层macro差10.4842pp达到该宏平均口径10，但总体accuracy差仅6.1102pp；不能在看到测试后换指标冒称总体10pp成功。原四层test92.7987/88.3797差4.4190、macro89.7096/86.2398差3.4698；原val差7.6433不当test差。单seed17、验证知情筛类且此前Caltech测试曾观察，是开发仿真证据，不代表未观察独立泛化/原20/全101/实拍。测试支持极不均、每类最少8张，必须看macro与全部逐类，不只总体accuracy。

## 正则负结果与匹配预算

四层从已审父150 raw/EMA/Adam7200/scheduler/LR.001完整接续50、仅MixUp.2->.4，没有重播/重置或额外手动LR减半，新增fit238.30716秒在1800上限内，无early；新增最低valNLL.582891164886未胜父E149 .570893883949，firstmin仍E149，模型参数digest07ddefec9212bf7465df2673a1c77a02d9ea039e0fc0118f010f97fff3be6611实际一致，regularize_test_reused真实复用旧test，0第二次forward。不能声称缓解过拟合已成功；不新增第二正则候选。额外残差预算不能当新同预算因果消融。

三/二层各freshseed17、同深度初值/order/affineMix/MixUp.2/LS.08/TV.05/capture.2/EMA.95/lr.002/scheduler/effective16/micro8，每E48Adam尾9、完整150/7200步，无early，各3600fit秒内；三层E-G-E+router680068相位/10张量，二层E-G+router479364/6，仍四224专家/full478/RGB100/联合8bit/十原窗口、角色和划分。GPU0/2/4/5四臂曾实际并行、一卡一个/2GiB+800MiB/A1000，不杀foreign进程。不同深度参数量由合同决定，不声称所有908552/11张量。

## 独立科学与唯一测试身份

run caltech10_regularize_then_L3L2_s17_20261015，177运行来源084e7be07be831d627bbafa98f87a378a2cf97a1=current/original，不是报告HEAD。源码只Git，所有旧源码/PT/data/history/失败/claims/图保持。新CPU恢复态/黑图10路归一与shape/seed准入、GPU0五种配置all10/macro16/tail9数值/梯度proof及各own productiongate都已一次passed；不是用工具准入替代最终科学。各stage自己的独占claim/log/audit/completed真存在，独立核source/config/PT/rawEMA/shape/init/11或10或6phasegrads/Adam9600或7200/orderaffineMix/LR/预算/保存IDlabelargmax10CMsupportmacroNLL/firstminimum，无审核forward/test图。

- regularize: 4e2ae5722d27c63266714a1de4c0a49b67feac3e8a414a8ab9e818fc3daa0c9a / actualauditrc0, completed一致。
- L3: 5a43b4412ea7b0437da59cacb435ad7d7f48d1adbd3ac8f67a5591590849d549 / actualauditrc0, completed一致。
- L2: 927196efc9fc2633dbf053f4b29496f78b8eaf61dd3d306edc81709ae89341c4 / actualauditrc0, completed一致。

四个新浅层测试各有exclusiveclaim/test_started/result/test_predictions/producer_exit/saved_test_audit/completed，原始exitrc0+actualexit、resultSHA与独立保存审查SHA和completed匹配；CPU用保存预测/标签独立核ID/argmax/CM/support/macroNLL，无第二次forward或test图。未重测相同四层参数。

- L3_rho03: selectedE149, PT 08edcbcfe3df3107d3c52a6f4f8d34c89f9f33069f90e00f3a38315e6b1517fd, result 46888e75928b64477966278e13e068ec79279253c43277911d1b4b79ed81a1f9, independent saved audit da3e80fb5c5ff9c67235b4590ffe1460a3016db8942f0bb04c1553a3a3c8adc8; actual producer rc0/exit, test1.
- L3_rho0: selectedE149, PT 364d4bea08ffefd459288d76cf3a4da81326130ad45e74d20953d9b7ba84ca01, result 2c8c62f376bcd97f0f0ddfc7acbc6949d162d7b33e1231a7637e446ba2a7127e, independent saved audit 2b27440c7f07a337a441f86710349fe958dd3bd7aebd9064eee3b9498e1a4635; actual producer rc0/exit, test1.
- L2_rho03: selectedE146, PT 6c37cfcb1db0f4fd5f8a3763d776370d38bb80fa533c619d50bac4b18c29f4ea, result 21b98846ccaa52e2b67bc5ff25de7cc15affbdb505aed3c91fd2fafeefecd934, independent saved audit 0e6f4447155fa708324a366a1634c6f23e472cf577e5cbeeabee66031bd5aabf; actual producer rc0/exit, test1.
- L2_rho0: selectedE149, PT efc8358ccf7cdbd90915dac370962f6c8fdbf158c65fa8f33393eae1a3da2c7a, result e10b15eda92fb98ebd8e13d93d3d22e821904278cfe90e6c7522bbe3e15cf6c2, independent saved audit 6589c7a14b0e3f126a00c5b916832390288c338c3d5a40404febb9ab899c1739; actual producer rc0/exit, test1.

[所有深度指标CSV](depth_comparison.csv)、[全部逐类test支持/召回与差CSV](class_recall_test.csv)、[原冻结四层配对测试](../../caltech10_gap_20261015/paired_once_test/README.md)、[正则科学负结果](../regularize_final/README.md)。原始JSON/逐样本预测ID/PT/完整日志仅private，不进入Git。

本次新人工范围已全结束，禁止自动fit/start/queue/campaign/prepare/cpu/numeric/gates/admission/audit/测试/图/markers/重试/换类/第二正则候选/扩预算/A100/native/formal6/lab；未来新人工授权再另按范围处理。保留t18与每日09晨报ACTIVE，完成本turn报告一次、普通完成/旧失败静默。
