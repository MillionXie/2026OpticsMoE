# 弱光场修正与训练合同（schema=3）

## 核对来源

参考当前主工作树 `LightGenV2/tasks/t12_text_to_image/lab_shs8um/bounded_amplitude.py`，
以及 `handoffs/t12_lab_small_20260926/full_test_bounded/README.md`。
该实测版本采用保零tanh映射（scale=0.5），不是rational版本，不是简单删除所有RMS。
旧逐图峰值缩放可能让大部分显示像素很暗，并且若只在BMP导出做，会造成仿真/实测合同不同。

```text
A_physical = tanh(A_nonnegative / 0.5)
E = A_physical * (sqrt(1-eta)*exp(i*phase) + sqrt(eta))
BMP = round(255 * A_physical)
```

保零、单调、值域[0,1]；梯度在非饱和区保持。不要将这条tanh作用在已含直流的复场上，
否则会改变干涉模型。也不要BMP导出时再除以本帧最大值。

## 本项目适配位置

六层：视觉router/expert/global、语言router/expert/global全部在相位/DC调制前插入。
expert先应用原路由权重，再进行有界编码；不改变router决策、RMS电子融合、readout或参数量。
保留原逻辑输入RMS归一化及CCD归一化，改变的是实际SLM显示的幅度合同。
`amplitude.py` 对哈希锁定原源码的AST在六个指定方法各插入一次，运行结束恢复；原导师21文件不修改。
新run/train/infer/部署暂存使用此图。旧teacher_reference保持旧图用于历史复查，不能冒充新版已修正权重。
旧PT不可直接作为schema=3结果，四组必须重新训练。

## 损失是否改

本轮不新增功率loss，保持原时间一致性回归、ranking、correlation及原辅助loss。
T12的图像MSE/L1和光功率operating penalty不能直接搬到MOS任务；否则与原消融同时改变额外因素。
先保证显示振幅有界且仿真/BMP一致；这不保证实际CCD SNR或每层功率一定达标。
需要实测检查曝光、饱和和ROI功率后，再决定另立profile增加功率约束，不能把仿真测试当光路验证。

## 验证范围

六方法插入与恢复；保零/范围/梯度；8bit振幅量化误差<=0.5/255；六层回填与仿真prediction一致；
G5合成前向反向相位梯度finite。上述都是仿真/接口测试，不是新光路采集。
设备幅相独立重采样顺序、LUT及camera ROI尚有原先记录的核验边界。

## 本轮四组设置

四组统一tanh/0.5；无pixel平移/phase dropout/router logit noise。
G2/G3训练DC=0；G4/G5训练DC固定0.30；四组统一设备评估DC=0.30。
CCD模型只有一份探测端Gaussian读噪声，另外有Poisson shot；不模拟额外EAC噪声。
k从4096降到3072，sigma_e从8升到10，暗电子仍0，属于增强pilot而非标定数据。
训练cache重建使用原LGVQ manifest、原始视频与本地冻结Qwen3-VL-2B，4帧/7x7/quality14。
从原train留出固定validation选模；原test不参与选模，最终同时报告train/validation/test回归指标。
四张GPU是用户本轮明确授权，单组单卡；supervisor只清理自己的PID并记录退出后的GPU PID检查。
