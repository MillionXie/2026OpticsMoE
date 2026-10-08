# OpenMoji 师姐排查版：真实CCD末端适配

已完成1000条原TRAIN的六层6000CCD采集，800拟合、200验证；原1000条TEST不参与梯度或轮次选择。仅训练shared_readout.decoder，上游相位、router、alpha、前端及电子权重冻结且哈希审计通过。

50轮中验证选第47轮，验证Changed-cell Accuracy为0.9350。原TEST实拍基线0.6710，适配后0.8375；Cell Accuracy为0.9921111，Object F1为0.9526515，Scene Exact Match为0.7500。同原权重未适配仿真Changed-cell为0.9365，不能把它当适配后的仿真成绩。

原基线完整CCD重放精确复现0.671；缓存末端输入与选定权重完整前向的抽检误差为0。权重SHA：e11322e39d2911ff8d2997f1ad4fc765cf87c0247303d15f0be1c188bfdf7c47。

report.json含完整指标与审计；history.json含每轮训练/验证指标；test_samples.json含1000条逐样本结果；split_audit.json含数据划分审计。此验证集来自原TRAIN，可能被原预训练见过，不宣称全新独立验证集。

完整best.pt/last.pt、test_outputs.pt、TRAIN/TEST原始CCD保留在师弟电脑 E:/code/guest/2026OpticsMoE/OpenMoji_Exp05_SHS_20260927/runs/decoder50_trainonly 及对应采集目录。设备已释放，未自动启动文生图。
