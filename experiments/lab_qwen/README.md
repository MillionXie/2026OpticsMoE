# Qwen 光路实验完整包

把 ZIP 解压到 `E:\code\guest\2026OpticsMoE`，所有实验入口都位于短路径
`experiments\lab_qwen`。实验顺序、需要修改的配置和命令只看同目录
`COMMANDS.md`，不要再混用历史 Fresnel/ROI 命令。

包内包含：双 SLM 配准 BMP、全白振幅的 Fresnel 点/十字标定、32 灰度×3 帧曝光
标定、sim-to-real agreement 理论帧与采集清单、quick210 末层离线微调数据、四层逐层
采集的首阶段数据、正式 EMA checkpoint、训练证据和绘图代码。

实验室电脑负责 SLM/CCD 播放采集、末层离线微调和画图；四层逐层流程每完成一层后
仍需把该层 CCD 传回服务器生成下一层输入。ZIP 不复制数 GB 的 Qwen 基座权重，服务器
沿用完整仓库和已有 Qwen cache。
