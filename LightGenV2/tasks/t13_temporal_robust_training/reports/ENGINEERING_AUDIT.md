# 导师交付与四工程审计

本报告记录schema=2的工程核查。当前schema=3（tanh/0.5、DC30、k3072/read10）以BOUNDED_AMPLITUDE_FIX.md和configs/study.json为准；下文旧参数不作为本轮实际配置。

## 已核对的原导师包

实际ZIP为 `LightGenPublic/tasks/t06_lgvq_temporal_consistency/releases/LGVQ_Temporal_08044_teacher_final_v2_20260923.zip`。
77个清单文件与解压目录逐字节一致，ZIP哈希见 teacher_zip_audit.json。
目录有 runtime/configs/weights/inputs/phases/assets/tests；best_checkpoint.pt为81,776,084字节。
导师拿到完整包可直接做固定权重推理；不需要训练缓存或原始视频。

原配置启用相干直流（训练0.20–0.35，eval0.20）；CCD噪声总开关关闭；router logit noise不是CCD噪声。
原 `_detector` 在每次forward重采样复场，17微米逻辑网格映射到8微米设备网格，再传播、探测、area返回逻辑网格。
因此“原来没做训练中插值”不准确。新G5是恢复这一处理；G2–G4训练暂按17微米传播，但评价都按8微米。
原包还带位移/dropout等处理；新四组统一关闭这些非研究因素，避免把它们混入递进结果。

## CCD模型：旧版与本次推荐版本

旧版：每场mean(I)乘有偏截断Gaussian，之后按场均值定Poisson计数。
这是经验相对扰动，Gaussian偏置混入shot，不应称已标定读噪声。

新schema=2：用统一强度到电子转换k；d为曝光内平均暗电子；sigma_e为读噪声RMS。

```text
X ~ Poisson(k * max(I, 0) + d)
R ~ Normal(0, sigma_e^2)
Y = max((X + R - d) / k, 0)
```

未裁零时：E[Y]=I，Var[Y]=(k*I+d+sigma_e^2)/k²。
shot随局部信号强度变化；read noise独立于信号；暗电流的均值扣除后仍有shot variance。
裁零导致低信号偏差，测试需考虑它；不得在零强度下声称仍严格无偏。
训练采用straight-through梯度，这是梯度近似，不是Poisson精确重参数化。

当前k=4096、sigma_e=8、d=0只是pilot，不是相机标定。所有组使用同一profile。
此算子在逻辑CCD边界（518网格）作用，未建立真实相机像素面积积分/ROI；未实现PRNU/DSNU、饱和和ADC。
正式标定需要固定曝光/gain、重复暗帧、多亮度平场、方差-均值拟合，以及仿真强度到电子尺度的匹配。
不能仅把calibrated字段改true就当成已完成实验标定；须保留标定数据及参数拟合证据。

## ACCEL的借鉴边界

ACCEL的噪声建模与我们的CCD不是同一硬件，不能移植其电压噪声常数。
Methods的Modelling of low-light conditions明确使用Poisson shot和零均值、固定方差Gaussian输出噪声；Gaussian量级用实测平均SNR定标。本次按这一物理拆分思路实现CCD算子，但参数与注入边界是本项目的，不能声称逐项复刻其OAC/EAC模型。
其自适应训练使用光学输出微调电子部分，区别于这里的噪声感知mask端到端训练。
应借鉴的是物理噪声来源、实测定标与统一误差条件下的对照展示，不把四组实验称为复现ACCEL适配算法。
来源：[论文正文与Methods](https://www.nature.com/articles/s41586-023-06558-8)、[作者代码说明](https://github.com/ytchen17/ACCEL)。

## 四份工程合同

01_baseline_post: 无CCD训练/无直流训练/训练后映射。
02_ccd_post: 只加CCD训练。
03_ccd_dc_post: 保留CCD，再加相干直流。
04_ccd_dc_intrain: 保留CCD/直流，将设备映射放进训练。

逻辑17微米、设备8微米，保持物理孔径和可学习mask参数数目；17→17时是identity，不能检验pitch mismatch。
映射幅值使用bilinear，相位单位复数使用nearest；传播使用设备pitch，探测area回到逻辑网格；该链保留梯度。
相干泄漏后的复场重采样与真实独立幅相面板播放尚需核验；当前不能称严格hardware twin。

四份工程由一份源码生成，各自复制runtime，不导入父目录。生成目录不手工维护。
teacher_reference里完整旧PT和输入用于reference推理；各组weights只允许真正训练后的同组checkpoint。
不会把同一旧PT复制改名而伪造四组结果。当前可计划、合成smoke、导师参考推理；正式训练受缓存预检限制。
