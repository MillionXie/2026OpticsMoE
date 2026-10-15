# 三层与二层：固定十类同150轮四GPU配对已实际启动

用户明确要求四层完成后跑三/二层并允许四卡，已按顺序执行：原冻结四层配对测试通过、单次正则完整200与自己的独占science通过但没有改善、相同selected参数复用原test后，CPU campaign2314182自动启动四个fresh臂。run caltech10_regularize_then_L3L2_s17_20261015，177训练来源084e7be07be831d627bbafa98f87a378a2cf97a1=current/original，不是报告HEAD。

| case | 物理GPU | 实际子PID | 参数/相位张量 |
|---|---:|---:|---:|
| L3_rho03 | 0 RTX4090 | 2323784 | 680068/10 |
| L3_rho0 | 2 RTX3090 | 2323855 | 680068/10 |
| L2_rho03 | 4 RTX4090 | 2323921 | 479364/6 |
| L2_rho0 | 5 RTX3090 | 2323987 | 479364/6 |

四个own production gate都已实际passed，maxgradrel依次6.4789e-7/7.2346e-7/2.4168e-7/2.5883e-7，所有各自对应相位梯度finite非零；不是拿独立GPUproof充数。当前都在早期训练（最近E13-15），不跨epoch拼临时gain/当最终test。global实际4个项目GPU子、一卡一个、A1000，未杀外国进程。PID仅历史，健康应实时proc/UUID核。

每深度rho.3/0同seed17/初值/完整150orderANDaffineMix；仍761train/157val、每E48Adam尾9/micro8/effective16，各7200更新、150maximum_epochs/noearly，各3600fit秒上限不扩/不retry。三层E-G-E加光router，二层E-G加光router，四224专家/full478/RGB100/原10窗口/联合8bit，原10名单与角色ID/缓存/划分不变，无电子分类器。MixUp.2/LS.08/TV.05/capture.2/EMA.95/lr.002/原scheduler保持。10pp是目标，不削弱对照或为了gap挑epoch。

全部实际producer rc0/exit后CPU自动各一次L3/L2_audit_claim/log，自己的L3/L2_science/audit.json +对应audit_completedSHA真passed，独立核source/config/动态shape/init/phase10或6grads/7200Adam/orderaffineMix/LR/预算/保存IDlabelargmax10CMsupportmacroNLL/firstminimum，CPU0forward/test。随后各val-selected模型冻结PT/来源/类别后各一次test，保存独占claim/test_started/result/predictions/producer_exit/saved_test_audit/completed，CPU独立保存预测审查无第二次forward。每深度paired_test保存真实accuracy/macro差/是否10pp，负结果如实。

不能另起campaign/queue/fit/audit/测试/marker；CPU2314182负责全部排程。任何failure保存、停止下游并通知，不重试/放宽阈值/延长预算；总本轮新增fit<=4.5GPUhours。source177冻结，仅报告可精确Git提交，不SCP源码/rawJSON/IDs/PT。旧所有资产及foreignT06保护。全部完成要rootstate complete_science_and_frozen_tests_passed、CPU父也实际退出、每producer rc0/科学与测试completedSHA正确；普通健康静默，真正完成/新失败主动报，t18与每日09 ACTIVE保留。
