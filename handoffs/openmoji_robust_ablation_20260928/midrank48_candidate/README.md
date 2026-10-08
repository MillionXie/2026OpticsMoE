# OpenMoji 中等电子容量候选（待光路验证）

此候选是已有的、按 G5 配置完整训练结束的 rank-48 权重；本次核验并下载原 best，没有重复训练或用 TEST 重新选轮次。

- 权重：`g5_lowrank48_best.pt`
- SHA-256：`735ed73a900ac2b7b0984f65b719c98667198000de6defaa74b33554b8d370a3`
- 共享电子头：234,520 参数；rank-16 为 160,792，原完整头为 381,976。
- 训练：原 TRAIN 5000 固定拆为 FIT 4000 / VAL 1000；12 epoch × 100 step，VAL 选第 12 轮；原 TEST 1000 不参与梯度或轮次选择。
- 正常仿真 TEST1000 Changed-cell Accuracy：0.9000。此数值不预测实测结果。
- G5 增量措施：CCD 噪声、相干 DC 30%、训练内 17→8→17 可微光栅近似；保留相同相位/Router、17 µm/10 cm 光学几何和 `tanh(abs/.5)` 的仿真/BMP 共同振幅合同。BMP 只做 `round(255*a)`，不额外按峰值缩放。

建议在原 TEST 1000 上先做相同曝光/ROI/方向的少量六层信号与接口核验，通过后逐层全量采集；然后用**独立 TRAIN 实拍 CCD**微调末端电子头、VAL 选轮次，再对固定 TEST CCD 重放一次。rank-16 G5 的实测 0.5860→末端适配 0.6710，完整头 G5 已实测 0.8740；这两端不能推断中间头的实测值。不得混用不同权重的 CCD。

源码：`C:\Users\Xml12\OneDrive\2026OpticsMoE\.worktrees\t04_openmoji_robust_20260928\LightGenV2\tasks\t04_openmoji_robust_ablation`。随附 `report.json` 与 `protocol.json` 为服务器原始报告副本。
