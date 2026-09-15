# Language Block 2 实物光路桥接

## 光路边界与尺寸

Language Block 1 的专家相位、传播、专家 CCD、本层 readout 和第一次门控融合均在
服务器正常前向。融合结果 `F1` 被重新编码并按 router 写回 2×2 区域；导出的振幅
正是这个 Language Block 2 global phase 输入的 `478×478` 逻辑有效区。按 8 μm
实物像素播放时最近邻扩展 2 倍，得到
`956×956` 物理有效区。全局 phase mask 同样由 `478×478` 扩展到 `956×956` 后居中
放入 SLM 画布。仿真仍以 16 μm logical sampling 在 `518×518` canvas 上传播。

CCD 原始照片可比它大，也可以不是正方形。先在独立的 `hardware_sdk` 配置中指定你
截取到的近似 2×2 专家有效区 ROI；处理器会缩放完整 ROI 到 `956×956`，不会只裁
中央 `224×224`。输出固定为 8-bit 灰度 PNG，且不翻转。项目随后按
`hardware.ccd.flip_vertical/flip_horizontal` 翻转，再做严格 `2×2` block mean，得到
模型使用的 `478×478` 强度。CCD 已经表示光强，不能再次平方。

## 归一化与扰动

独立预处理默认保持已有 8-bit 的固定 `0→255` 标度，不逐图拉伸，也不估计背景。
如果原图确实是 16-bit，必须根据相机标定显式填写统一 fixed range。
没有独立暗场文件，因此模型明确不执行背景扣除，也不会从当前 CCD 图像猜测背景。
仿真和实测只共同执行每帧均值尺度归一化、相对强度裁剪及 `log1p`。该操作抵消乘性
光强/曝光变化，但不会假装消除加性背景。训练仍加入全局增益、偏置、读出噪声和
空间错位，让后续网络学习承受这些误差。若以后确实采集 dark frame，应在独立
hardware_sdk 流程中显式扣除并记录，而不是在模型中估计。

## 数据和微调

`manifest.csv` 固定 gallery、train、test 的播放次序与 basename。处理后的 PNG 必须按
相同 basename 放进 session 的 `ccd_captured/`。`register_ccd` 会拒绝非 uint8、非
`956×956` 文件，并把翻转、块平均及统计写进 `ccd_registered/*.json`。

实测 CCD 是不可微边界。微调只更新 Block 2 CCD readout/output adapter、Block 2 gate、
Language output norm 与 64D retrieval readout；相位、router、光前编码器、Vision、
Language Block 1 和电子 Block 2 都冻结。
