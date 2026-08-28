# Qwen + MNIST-4 新实验室完整包

把 ZIP 解压到独立目录 `E:\code\guest\qwen_mnist4_full_lab`。实验人员只编辑
`experiments\lab_qwen\LAB_CONFIG.yaml`，全部操作从 `COMMANDS.md` 第 0 步顺序执行。

包内只面向本次设备：1024×1024、17 µm Meadowlark 高速振幅 SLM；1920×1200、
8 µm 手动相位 SLM；2048×2048 TUCam CCD。程序根据四个任意像素坐标自动生成合法
硬件 ROI、478×478 透视校正、contract 和 SHA，不要求角点坐标是 4 的倍数。硬件
ROI 会自动按当前 TUCam 约束生成：左/上/高按 4 像素、宽按 8 像素向外对齐。

内容包括双 SLM 对齐、Fresnel P1/P4/P9、32 灰度×3 帧曝光标定、Qwen 仿真—实测
一致性、Qwen 最后一层和四层逐层流程，以及 MNIST-4 的 4 个已训练 mask、quick40、
formal400、原始四 ROI 分类、PCC/SSIM/NRMSE/余弦/能量/质心等一致性分析和 Arial
7 pt 的 PDF/SVG/600 dpi PNG 图表。

另外提供 `shape_agreement.py` 的模型无关几何基准：6 个非对称振幅形状与 6 个
几何相位 mask 形成 36 组 10 cm 传播实验，同时导出理想连续相位和实际 8-bit
传输量化仿真。评估严格复用正式四点 homography，输出逐样本及逐 mask 的 PCC、
signal-PCC、SSIM、shape-NRMSE、余弦、质心、能量与饱和指标；具体命令见
`COMMANDS.md` 第 6.1 节。
