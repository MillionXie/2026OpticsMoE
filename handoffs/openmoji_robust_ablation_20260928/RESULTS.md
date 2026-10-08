# OpenMoji 有界输入：四份训练、五个仿真条件

本次基于用户指定的新布局 OpenMoji 版本代码建立独立工程。四组均从同一个旧共享读出权重 `a69ddcee…` 续训，不是从随机初始化开始；它曾看过原 TRAIN，历史上也受 TEST 选模影响，因此这里的原 TEST 数字不应称为全新盲测。1000 条原 TEST 不参与这四组的梯度或本轮选 epoch；从 5000 条 TRAIN 固定分出 4000 拟合、1000 验证，仅由验证 Changed-cell Accuracy 选 best。

共同修复：保零、保相位的 `tanh(|field|/0.5)` 有界振幅直接进入仿真传播；BMP 仅做同一振幅的 `round(255a)`，不再另按 batch 峰值压缩。四组共用原有相位 dropout 0.08。CCD 扰动为训练期增益/偏置/读出噪声；DC30 为相干未调制光的**强度**比例。G5 的 17→8→17 光栅插值只是可微近似，不是 8 μm 精确传播或实拍。

| 显示条件 | 训练权重 | TEST 条件 | VAL 最佳轮 | VAL | 原 TEST Changed-cell |
| --- | --- | --- | ---: | ---: | ---: |
| G1 理想 17 μm | r0_base | 理想传播 | 2 | 0.9930 | 0.9545 |
| G2 基础设备 | r0_base | 光栅近似 | 2 | 0.9930 | 0.9550 |
| G3 +CCD 噪声 | r1_ccd | 同一光栅近似 | 3 | 0.9900 | 0.9490 |
| G4 +相干 DC30 | r2_ccd_dc30 | 同一光栅近似 | 2 | 0.9900 | 0.9500 |
| G5 +训练内插值 | r3_ccd_dc30_grid | 同一光栅近似 | 3 | 0.9925 | 0.9545 |

四份权重在各自原生仿真 TEST 下分别为 0.9545、0.9495、0.9475、0.9540；上表 G2–G5 是预先声明的统一光栅条件，所以与各自原生评估略不同。用户原 `.8890` 参考是另一份权重，不能把本轮数字差额全部解释为有界编码的增益。后期验证值低于第 2–3 轮峰值，故均保留验证选出的早期 best；没有用 TEST 挑轮次。

五组目前**全部是仿真**，不能拿 G1→G2 的近零差距说明真实光路已对齐，也未达到老师要求的“先有至少 10 个百分点实拍落差、消融后小于 5 个百分点”的实验证据。下一步要先核对六阶段仿真/BMP/相位桥等价、硬件 SNR，再以同一 1000 TEST 身份整层采集五条件，完整报告每条件的实拍 Changed-cell、重复性及和对应仿真的百分点差。

逐组 `report.json`、`history.json`、`split.json`、`protocol.json` 和 `five_conditions.json` 在本目录。best/last 权重位于服务器 `/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t04_openmoji_robust_ablation/runs/20260928/<group>/`。源码为服务器独立 worktree `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t04_openmoji_robust_20260928`，并保留本地隔离 worktree `.worktrees/t04_openmoji_robust_20260928`。
