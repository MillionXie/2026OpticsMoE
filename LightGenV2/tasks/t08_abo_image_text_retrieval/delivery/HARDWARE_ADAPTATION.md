# 实验室硬件适配边界

本交付包的 checkpoint 保存了可训练相位、光 Router、电子残差和读出头，但它仍是
**仿真最终版**。不能把网络输入张量当作单张 BMP，也不能把相位图用普通 resize
后直接宣称完成硬件部署。

## 不允许改变的模型合同

- 15 cm 自由空间传播。
- Vision 两层、Language 两层，各 4 个专家，光 Router Top-2。
- 专家逻辑尺寸 224×224，按 2×2 放在 478×478 有效场；全局层为 478×478。
- 相位由 checkpoint 的 raw phase 通过 `2*pi*sigmoid(raw)` 得到。
- 64D 最终 embedding 和 192→96→192 电子残差结构。
- 同尺度融合以及正式 alpha；不要在 CCD 强度归一化后再偷偷改变 alpha。

## 需要实现的硬件桥

1. 将每一层实际输入编码成振幅 SLM BMP。
2. 按相位 SLM 的 8 µm 像素和既有标定合同导出相位 BMP；不得使用图像软件插值。
3. 每次换图后等待 SLM write-complete、settle delay，再抓取 CCD；丢弃旧帧。
4. 用 detector homography 把 CCD 原始 ROI 变换到 478×478 canonical model xy。
5. 从 CCD 的四个 Router 探测区读能量并选择 Top-2；语言侧保留共享 E0 规则。
6. 逐层保存 raw CCD、canonical CCD、归一化张量和 SHA256，才能定位仿真/实测差异。

公共硬件工作流位于 `experiments/hardware_sdk/`。它提供设备、ROI、时序和重建基础，
但本模型专用的四层采集循环仍需由师姐的 AI 按上述合同接入。完成接入前，不应把此包
称为“插上设备即可复现 0.88”；0.88 是仿真 TEST Hit@1。
