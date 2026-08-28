# Qwen 光路实验完整包

把 ZIP 解压到 `E:\code\guest\2026OpticsMoE`，所有实验入口都位于短路径
`experiments\lab_qwen`。实验人员只编辑 `LAB_CONFIG.yaml`；运行一次
`python -m experiments.lab_qwen.prepare_lab` 后，程序会自动计算合法硬件 ROI、生成
四点透视 contract、固定 SHA，并生成正式硬件配置。实验顺序只看 `COMMANDS.md`。

包内包含：双 SLM 配准 BMP、全白振幅的 Fresnel 点/十字标定、32 灰度×3 帧曝光
标定、sim-to-real agreement 理论帧与采集清单、quick210 末层离线微调数据、四层逐层
采集的首阶段数据、正式 EMA checkpoint、训练证据和绘图代码。

四个光学角点使用 CCD 全传感器坐标，不要求是 4 的倍数；只有程序自动外扩得到的
TUCam 硬件 ROI 必须按 4 对齐。实验室电脑负责 SLM/CCD 播放采集、末层离线微调和
画图；四层逐层流程每完成一层后仍需把该层 CCD 传回服务器生成下一层输入。ZIP 不复制
数 GB 的 Qwen 基座权重，服务器沿用完整仓库和已有 Qwen cache。
