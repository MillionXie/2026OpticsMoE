# Language Block 2 实物光路桥接

## 光路边界与尺寸

保留旧 MoE4/router。专家相位、专家传播和 OEO 先在服务器仿真；导出的振幅是全局
相位板之前的 `478×478` 逻辑有效区。按 8 μm 实物像素播放时最近邻扩展 2 倍，得到
`956×956` 物理有效区。全局 phase mask 同样由 `478×478` 扩展到 `956×956` 后居中
放入 SLM 画布。仿真仍以 16 μm logical sampling 在 `518×518` canvas 上传播。

CCD 原始照片可比它大，也可以不是正方形。先在独立的 `hardware_sdk` 配置中指定你
截取到的近似 2×2 专家有效区 ROI；处理器会缩放完整 ROI 到 `956×956`，不会只裁
中央 `224×224`。输出固定为 8-bit 灰度 PNG，且不翻转。项目随后按
`hardware.ccd.flip_vertical/flip_horizontal` 翻转，再做严格 `2×2` block mean，得到
模型使用的 `478×478` 强度。CCD 已经表示光强，不能再次平方。

## 归一化与扰动

独立预处理用全文件夹共享的强度范围转成 8-bit，避免逐图拉伸破坏相对光强。若相机
黑电平和饱和值已标定，优先使用 `fixed_range`；否则默认用全数据共同的百分位范围。
进入模型后，仿真和实测共同执行低分位背景扣除、每帧均值除法、相对强度裁剪及
`log1p`。训练还分别扰动输入/专家、全局相位输入以及 CCD 偏移，并加入全局增益、
偏置和读出噪声。

## 数据和微调

`manifest.csv` 固定 gallery、train、test 的播放次序与 basename。处理后的 PNG 必须按
相同 basename 放进 session 的 `ccd_captured/`。`register_ccd` 会拒绝非 uint8、非
`956×956` 文件，并把翻转、块平均及统计写进 `ccd_registered/*.json`。

实测 CCD 是不可微边界。微调只更新 MoE CCD readout/output adapter、融合 gate、
Language output norm 与 64D retrieval readout；相位、router、光前编码器、Vision、
Language Block 1 和电子 Block 2 都冻结。
