# 新人工授权的残差正则与三/二层实际启动

run caltech10_regularize_then_L3L2_s17_20261015，177训练/审核来源084e7be07be831d627bbafa98f87a378a2cf97a1=current/original；CPU campaign2314182实际启动，GPU4残差子2315182实际运行，最近真实E158val86.6242/NLL.597259（临时，未完成200/科学）。从已审父E150 lastSHA3dd438d7dd6b59ed3625ce20a140e38fb1a0841b658fe1e84a035b0285a33e74/history2db7ed346978b4c40079a9b00167ee47d48c655a09709d2f44b58fd0cc94d154恢复raw/EMA/Adam7200/scheduler150/LR.001，只有MixUp.2->.4，E151-200新50轮、1800新增fit秒/noearly，不手动额外降LR，不重置/重播。firstminimum验证EMA balancedNLL在保留150+50里选；旧E149参数若仍被选则复用已存测试不再forward，真新selected参数最多一次冻测，复用固定对照测试。

四层科学/冻测之后自动fresh三层E-G-E与二层E-G，每深度rho.3/0两臂，各150完整、3600fit秒、seed17/sameinit/order/affineMix/config、48Adam每E尾9、7200步，无early。三层680068相位/10张量、二层479364相位/6张量，均四224专家、478global/RGB100/8bit/原10坐标窗口，不虚称浅层908552/all11。实际四臂目前pending，尚未训练，按CPU campaign后续自动排程，不能另起queue。

新CPU恢复态与2/3深度黑图十路归一/shape/seed/8syntheticforward准入已一次passed，SHA17e3db1d52460bb4c35a093bf958230a8051d314d3b97847c84681e52ca452bb；新GPU0含4层恢复态/3层两rho/2层两rho all10+macro16+tail9，最大gradrel7.1874e-7、所有对应相位梯度finite/nonzero、optimizer0/test0，SHA79629de6758c972b3eb1376d856f3abde357c0fcce09b2b565b45fe8a4691a43。CPU/GPUproof及claims不可重跑，真实生产自己的gate每臂独立执行。

只GPU0/2/4/5最多4个T18子一卡一个、2GiB+800MiB/A100禁止、foreign任务不动。四层reg完成后GPU4子退出rc0才自己独占regularize_audit_claim/log -> regularize_science/audit.json + regularize_audit_completedSHA；三/二层各臂实际rc0/150才自己L3/L2独占科学，随后各一次冻结测试与CPU保存预测独立核验。任何failure保留、停下游、不重审/重试/阈值放宽/扩预算，不能因为差距未10继续候选或选test。四层额外优化预算比较明确标注，不能冒充新同预算因果消融。

原冻结四层配对test已经各一次完成：92.7987/88.3797差4.4190pp、macro89.7096/86.2398差3.4698，未10；[完整测试与逐类](../../caltech10_gap_20261015/paired_once_test/README.md)。这是验证知情筛类且旧Caltech测试已观察的开发仿真任务；不声称完整101、原20或完全未见泛化。固定10类和所有角色/IDs/source/PT/history/claim保持，不增加电子分支/native/lab/formal6/更深层或A100。新独立科学与测试真完成后主动汇报实际正负结果，t18/09 ACTIVE留存，普通健康静默。数据/PT/rawJSON/逐样本ID/完整日志私有，Git只源码/聚合。
