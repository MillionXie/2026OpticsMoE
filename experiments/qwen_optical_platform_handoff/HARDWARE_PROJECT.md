# 实验室硬件与实测微调工程

## 职责边界

本工程只负责当前设备驱动、几何/亮度标定、SLM payload 重建、CCD 采集、仿真—实测
一致性以及使用实测 CCD 的逐层微调。正式训练配置和初始 checkpoint 来自仿真工程。

代码包本身不等于可运行任务。首次运行某个新任务前，必须把仿真工程产生的 task payload
放入独立任务目录；若实验室断网，还必须把完整 Qwen snapshot 放在
`models/Qwen3-VL-Embedding-2B`。禁止因为缺文件而在线静默下载或回退到 Caltech checkpoint。

新实验电脑必须重新测量：

- Meadowlark LUT、工作温度和曝光；
- 相位 SLM 中心及必要翻转；
- CCD 四个逻辑角点、hardware ROI 和 homography；
- settle delay、丢帧数和饱和率；
- 实际方向、零级分量、PCC/SSIM/能量比。

旧电脑的四点坐标、LUT 和曝光只能作为示例，不能作为新平台默认值。

## 标准执行顺序

1. 安装厂商驱动，再安装 Python 环境；先做 camera/SLM smoke。
2. 标振幅 LUT；标定曝光和时序。
3. 生成并播放双 SLM 对齐、菲涅尔和不规则形状图案。
4. 填四个逻辑角点，生成 478×478 canonical homography 合同。
5. 用人工形状和任务样本同时评价 PCC、SSIM、NMAE、质心、能量比和饱和率。
6. 核对仿真 payload manifest、checkpoint SHA、stage 与 phase SHA。
7. 按顺序采集：vision expert → vision global → language expert → language global。
8. 每采一层，只微调该层下游；已经实测的上游必须冻结。
9. development 选 checkpoint，sealed test 最终一次；保存完整报告。

## CCD 数据合同

正式 `ccd_captured` 是 hardware ROI 的原始整数强度经过 homography、固定 bit-depth
映射后的 canonical 图像。禁止逐帧 min-max、自动背景扣除或为了提高 PCC 自动选翻转。
网络域允许均值归一化、相对截断和 log1p，但仿真与实测必须使用同一个函数。

## 当前包内设备范围

- 1024×1024、17 μm Meadowlark PCIe 振幅 SLM；
- 1920×1200、8 μm 相位 SLM（手动或 Meadowlark）；
- TUCam/Mosaic CCD；
- 532 nm、10 cm、478×478 canonical active field 的参考实现。

Holoeye、DVP legacy 和历史标定大图不进入交接包。若未来换设备，应新增 driver，不能
把新设备伪装成现有 driver 后绕开分辨率和 bit-depth 校验。
