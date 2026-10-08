# OpenMoji 五组消融：四份权重

以下均为同一原始完整电子头架构、同一 1000 条正常仿真 TEST 的准确率；不是光路实测。G1 理想组与 G2 直接部署组共用一份权重，所以五组只有四份 PT。

| 组别 | 权重 | 正常仿真 TEST Accuracy | SHA-256 |
| --- | --- | ---: | --- |
| G1 理想 / G2 直接部署 | `g1_g2.pt` | 0.8880 | `579b19e78befe93cd258ce02308235f1a9176316f37bc42fc2e1b6ff06d38e9f` |
| G3 探测器噪声训练 | `g3_ccd.pt` | 0.9455 | `ccd6a1767d6e1a88aa2f4896f1dda372d0f4fb6a88716893afc5744a720b873b` |
| G4 再加相干 DC 30% | `g4_ccd_dc30.pt` | 0.9440 | `d80b04f3081099cb05c24e13fa1da4ae047d8525789777cb22b6b144001726a5` |
| G5 再加训练内光栅近似 | `g5_ccd_dc30_grid.pt` | 0.9435 | `8ba828c8168db6cc67b2721c65fa6db57efadc6d091e3a7204b79a49a17b3e52` |

共同光学合同：17 µm 输入、10 cm 传播；保零有界振幅 `tanh(abs/.5)` 在仿真传播和 BMP 输出前一致应用；BMP 为 `round(255*a)`，不再按每批峰值二次缩放。G3—G5 各从 5000 条 TRAIN 划分 4000 拟合/1000 验证，验证集选 epoch，TEST 不用于梯度或选模。光路实测需另行完成。

源码：`C:\Users\Xml12\OneDrive\2026OpticsMoE\.worktrees\t04_openmoji_robust_20260928\LightGenV2\tasks\t04_openmoji_robust_ablation`。
