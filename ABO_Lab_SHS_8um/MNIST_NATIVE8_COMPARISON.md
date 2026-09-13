# MNIST：旧相位重采样与原生8μm训练

## 为什么做，不代表必须重训

导出中心(980,590)改为当前(960,600)只是放置坐标变化，本身不要求重训。
真正需要验证的是：原17μm离散相位映射到8μm面板后，对实际输入和传播的适配。
重训不是相位LUT、偏振、振幅LUT或像素配准问题的替代品。

本次采用用户确认的两组；不是ABO重训，也不改变ABO固定权重。

|组|训练|仿真/导出|
|---|---|---|
|A|固定post_robust_best epoch12，不更新|旧478×478相位按物理坐标重采样，在原生8μm传播器评估|
|B|原生1016×1016相位从raw_phase=0开始，即相位π|原生8μm网格训练，再在相同传播器和同一test集评估|

A没有可训练参数；其原参数量228484。B可训练参数1032256。两组容量、训练历史
不同，所以结果只能回答“这两种部署方案哪个好”，不能当成只改变像素间距的严格因果消融。

## 物理与读出合同

- 532nm、传播10cm；逻辑输入保持原MNIST的bicubic336、两侧pad32到400，再pad39到478。
- 原物理有效宽8.126mm。两组都用同一个1016×1016的8μm原生栅格（8.128mm）。
  栅格取整每边多1μm；不是扩大ROI来获得性能优势。两组完全相同。
- 数值传播网格2176×2176，8μm，宽17.408mm；与旧1024×1024、17μm的数值窗口宽相同。
- 保留实际继承配置中的角谱截止1.10°。不是旧README概述中的0.65°，本轮不重新调k空间。
- 探测框采用原59×59、17μm区域的物理坐标，原生网格边界像素以覆盖面积计权。
  因此探测区位置/面积不因整数像素取整而移动或放大。
- CCD只做光电探测后的四区域积分和argmax；无电子分类头、归一化、log或背景扣除。
- 相位FP32正常优化，不做训练期8bit直通量化；只在导出BMP时量化。
- 相位BMP1920×1200，中心(960,600)，当前实验室上下+左右翻转及整幅255−g。
  师姐的正常LUT面板不能直接使用这个反灰度文件。

特别注意：原A文件是在478逻辑相位上先翻转、再物理栅格化；恰好落在逻辑像素
边界的native像素采用原程序的半开区间规则。因此“栅格化再翻转”不总是等价。
A仿真已按原BMP路径精确还原；导出的A BMP与本轮60%实测所用文件逐像素完全相同。
两者只有像素数据前调色板的保留字节不同（255处），解码像素全部一致；文件SHA不同
不代表SLM接收的灰度不同。A文件SHA：
`395d2c60abb72ae9735d47e1fc7ec303c370662127f7571f5dbd24ba45fb9c0b`。

## 数据与训练口径

沿用原MNIST-4划分：train=22279、validation=2475、官方test=4157（以protocol内实际counts为准）。
没有按光路40张结果筛训练样本，也没有把这40张当全集。
为了与原任务划分一致，本对照保留原validation：B每epoch以validation准确率选best，
同分比较MSE；A在test评一次，B训练结束后best在test评一次。

60 epoch，Adam lr0.01，microbatch10、累积到effective batch100。前8epoch不加位置扰动，
之后输入/相位/CCD前分别以0.5概率作上下左右2native像素扰动（16μm；旧1像素=17μm，
差1μm是整数栅格近似；每microbatch抽样）。没有新增直流/噪声或教师loss。
loss是原全平面MSE×100；边界按0/1物理目标覆盖面积正确积分。
只保存best_checkpoint.pt和last_checkpoint.pt，不每5epoch保存一份。

## 已完成与运行位置

GPU检查和单步回归已通过：退回17μm时与原模型CCD最大绝对误差0；原生相位梯度
RMS约2.05135e-5，单步相位最大变化0.015707rad，确认相位真的更新。

A的完整原生8μm仿真评估已完成：validation **85.0505%**，test **86.5528%**（4157张）。
这不是实测准确率；之前40张实测60%、40张旧网格仿真95%是另外的口径。
B正在训练，不能用早期validation冒充最终test结果。

服务器：guest3@202.120.62.181:24096，仅GPU1（RTX4090）。启动PID1253612。
run：`/DATA/DATA1/guest3/2026OpticsMoE/ABO_Lab_SHS_8um/runs/simulation/mnist_native8_20260913/`。
启动日志是同级`mnist_native8_20260913.launcher.log`。

- protocol.json：源代码hash、配置、数据量、几何和设备。
- baseline_A.json：A的validation/test以及BMP哈希。
- history.json、best.json：B的逐epoch损失、准确率、梯度、相位变化。
- best_checkpoint.pt / last_checkpoint.pt：字典中的model_state_dict.raw_phase是1016×1016 FP32。
- result.json：**仅在全部60epoch完成、best重新test后才产生**。
- A_old_fixed_native8_xy_inverse.bmp / B_native8_best_xy_inverse.bmp：后者训练完成后生成。
- 对应`*_phase_rad.npy`保存未翻转的逻辑物理相位（弧度），不要再把BMP反推为训练相位。

## 可复现命令

源码commit：`4dc0c4cf9362aba9015895f582c2a01eee7a9a40`。
不修改服务器脏工作树，通过已fetch的确定commit执行新增独立脚本，原MNIST依赖文件只读并记录SHA。
在服务器项目根运行，使用未存在的新输出目录；先检查GPU空闲，不能覆盖正在运行的任务。

```bash
git show 4dc0c4cf9362aba9015895f582c2a01eee7a9a40:ABO_Lab_SHS_8um/mnist_native8.py | \
CUDA_VISIBLE_DEVICES=1 /home/guest3/miniconda3/envs/xml/bin/python -u - \
  --mode train --dataset-root data/mnist \
  --out ABO_Lab_SHS_8um/runs/simulation/mnist_native8_repeat01 \
  --source-commit 4dc0c4cf9362aba9015895f582c2a01eee7a9a40 \
  --epochs 60 --microbatch 10 --effective-batch 100
```

同一脚本`--mode smoke`可运行原模型一致性、物理探测区面积和梯度检查。
训练完成后先看完整test，再对A/B使用同一固定输入、曝光、ROI和时序作成对实测；
不能只凭B仿真更高就宣称已经修复实验光路。
