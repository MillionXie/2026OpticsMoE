# Qwen + MNIST-4 新实验室完整包

把 ZIP 覆盖解压到统一仓库目录 `E:\code\guest\2026OpticsMoE`。实验人员只编辑
`experiments\lab_qwen\LAB_CONFIG.yaml`，全部操作从 `COMMANDS.md` 第 0 步顺序执行。

Qwen 主模型是强噪声续训版本：截断偏置高斯 CCD 噪声为干净单帧均值的
`mean=6%、std=5%、clip=[-4%,16%]`，四个光电融合门的配置下限为 1%。该模型
固定测试 Top-1 为 82.0%，检查点由训练损失选择，不使用测试集挑选。

包内只面向本次设备：1024×1024、17 µm Meadowlark 高速振幅 SLM；1920×1200、
8 µm 手动相位 SLM；2048×2048 TUCam CCD。程序根据四个任意像素坐标自动生成合法
硬件 ROI、478×478 透视校正、contract 和 SHA，不要求角点坐标是 4 的倍数。硬件
ROI 会自动按当前 TUCam 约束生成：左/上/高按 4 像素、宽按 8 像素向外对齐。

包内保留当前设备的 `slm7930_at532-70c-pixel-2.lut`，并提供 64 灰度×3 帧的全局
LUT 重标定：自动寻找实测暗态，选择动态范围较大的单调支路，做保序拟合与反向线性
插值，生成新的 256 项 LUT，再自动重扫验证。旧 LUT 永不覆盖，具体见命令文档第 4.1 节。

内容包括双 SLM 对齐、Fresnel P1/P4/P9、32 灰度×3 帧曝光标定、Qwen 仿真—实测
一致性、Qwen 最后一层和四层逐层流程，以及 MNIST-4 的 4 个已训练 mask、quick40、
formal400、原始四 ROI 分类、PCC/SSIM/NRMSE/余弦/能量/质心等一致性分析和 Arial
7 pt 的 PDF/SVG/600 dpi PNG 图表。

MNIST quick40 还带有独立的 20 帧时序诊断：5 档 SLM 完成后等待时间 × 4 个数字，
逐帧记录 SLM 写入、实际等待、CCD 丢帧/采集/透视/保存耗时，并输出 478×478 CCD
四个判别区与规范方向叠加图。具体命令见 `COMMANDS.md` 第 5.2 节。

另外提供 `shape_agreement.py` 的模型无关几何基准：6 个非对称振幅形状与 6 个
几何相位 mask 形成 36 组 10 cm 传播实验，同时导出理想连续相位和实际 8-bit
传输量化仿真。评估严格复用正式四点 homography，输出逐样本及逐 mask 的 PCC、
signal-PCC、SSIM、shape-NRMSE、余弦、质心、能量与饱和指标；具体命令见
`COMMANDS.md` 第 6.1 节。
