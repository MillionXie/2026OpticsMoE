# 一次残差MixUp.4接续：完整200、科学通过、无可验收提升

父十类L4残差已完整150并真正独立SCI，fullE150 raw/EMA/Adam7200/scheduler150/LR.001完整恢复，不重播/重置，不额外手动LR减半。仅MixUp.2->.4，E151-200实际完整50/maximum_epochs，无early，新增fit238.30716秒低于1800上限。GPU4子2315182已实际rc0退出，自己的独占regularize科学首次passed，auditSHA4e2ae5722d27c63266714a1de4c0a49b67feac3e8a414a8ab9e818fc3daa0c9a与completed实际一致、auditrc0；CPU shape/seed/保存数学核0forward/test。

新增50最低验证EMA balancedNLL0.582891164886未优于父E149 0.570893883949，因此全prefix150+new50的firstminimum仍选E149：train99.3430/val86.6242/macro86.3258。不能挑新增最高accuracy轮称提升或说过拟合已解决。这个方案没有产生可验收收益，不继续自动尝试/换配置/扩预算。

所选model参数digest与已冻结测试E149完全一致：07ddefec9212bf7465df2673a1c77a02d9ea039e0fc0118f010f97fff3be6611；已真实写regularize_test_reused.json，复用原测试resultSHAbc3767a7f1fa02cf854f65e3b2679857cf8eae4a4fecf91011e939ecb26be1b0。没有第二次forward/test、没有把新PT容器字节变化当新参数。原测试仍92.7987/88.3797、差4.4190pp，未10；macro差3.4698。完整[冻结测试与逐类](../../caltech10_gap_20261015/paired_once_test/README.md)。

run caltech10_regularize_then_L3L2_s17_20261015/regularize_L4_rho03；177来源084e7be07be831d627bbafa98f87a378a2cf97a1=current/original，不改原169。真实独立核exactprefix150/rawEMA/Adam9600/11grad/50新轮独立MixUp.4、旧150独立MixUp.2/orderaffineMix/原LRreplay/新增预算/IDlabelargmaxCMNLL/firstminimum。自己的恢复态生产gatepassed maxgradrel5.4872e-7/all11finite非零，不拿独立numeric替代。原有PT/data/history/claims全部保留，不重审/重试/改源。

按用户顺序该阶段结束后，三/二层四臂已真实并行启动，见[实际四卡配对](../depth_pairs_start/README.md)。旧完成禁令不阻止本轮新人工授权；不自动追加四层候选或test重测。rawJSON/PT/IDs/完整日志私有，Git仅聚合。
