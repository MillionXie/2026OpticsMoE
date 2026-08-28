# Qwen + MNIST-4 新实验室完整包

把 ZIP 解压到 `E:\code\guest\2026OpticsMoE`。实验人员只编辑
`experiments\lab_qwen\LAB_CONFIG.yaml`，全部操作从 `COMMANDS.md` 第 0 步顺序执行。

包内只面向本次设备：1024×1024、17 µm Meadowlark 高速振幅 SLM；1920×1200、
8 µm 手动相位 SLM；2048×2048 TUCam CCD。程序根据四个任意像素坐标自动生成合法
硬件 ROI、478×478 透视校正、contract 和 SHA，不要求角点坐标是 4 的倍数。

内容包括双 SLM 对齐、Fresnel P1/P4/P9、32 灰度×3 帧曝光标定、Qwen 仿真—实测
一致性、Qwen 最后一层和四层逐层流程，以及 MNIST-4 的 4 个已训练 mask、quick40、
formal400、原始四 ROI 分类、PCC/SSIM/NRMSE/余弦/能量/质心等一致性分析和 Arial
7 pt 的 PDF/SVG/600 dpi PNG 图表。
