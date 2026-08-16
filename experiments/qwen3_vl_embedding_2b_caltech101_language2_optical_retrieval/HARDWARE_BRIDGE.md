# Language Block 2 实物光路桥接

## 数据边界

`manifest.csv` 固定了 gallery、train、test 的播放顺序和 basename。训练 CCD 只用于
下游微调；test CCD 只用于报告性能，不参与参数更新。实物文件必须保持 manifest
中的 basename，避免电子缓存和 CCD 错配。

## 光路定义

导出的 `224 x 224` 振幅在 8 um 实物 SLM 上按最近邻扩展成 `448 x 448`，相位
也采用同一物理尺寸。仿真使用 16 um logical sampling 和 518 padding canvas。
CCD 推荐直接采集对齐的 `448 x 448` ROI，程序严格执行 2x2 block mean，还原为
`224 x 224` logical intensity。CCD 文件已经是强度，禁止再次平方。

## 归一化

仿真与实测共同执行：低分位背景扣除、每帧平均光强除法、相对强度裁剪、log1p
压缩和逐 token LayerNorm。前两步主要抵消暗电平、激光功率、曝光和模拟/实物的
全局增益差；它不会修复空间错位，因此训练中另行加入输入错位和读出噪声。
输入相对 phase mask 的平移与 CCD ROI 的平移是两次独立随机扰动，默认各为
正负 8 logical pixels，即当前 2x 实物缩放下约正负 16 个物理像素。

## 微调范围

实测 CCD 被视为不可微边界。只更新 CCD normalizer 的 affine、光学 decoder、
融合 gate、Language output norm 和 64D retrieval readout。相位、光前编码器、
Vision、Language Block 1 以及电子 Block 2 均冻结。
