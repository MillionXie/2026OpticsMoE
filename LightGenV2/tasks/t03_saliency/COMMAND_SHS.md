# SALICON CC 0.8625：实验室三阶段

这是独立复评 CC=0.8624925081777596 的固定权重，不是重新训练的模型。
权重 SHA256：`036bc8caedcde4d6dabce276b1a2e4af15d960d88620098b19e1c198849b8bfe`。

## 架构与输入

冻结 Qwen 的图像预处理、patch embedding 和位置编码已逐图离线缓存，保持原 dtype。
缓存位于第一个替换块之前，不缓存电子残差或光分支输出。实验电脑执行原模型：
电子残差 E1 + 光 Router→Top2 专家→CCD → 同尺度融合 →
电子残差 E2 + Global→CCD → 同尺度融合 → 原显著性读出头 → 224×224 密度图。
只有三次物理传播，**没有语言阶段，也不是六次传播**。

此版本本身为图像显著性任务，无新增文本输入或 Transformer。
正式测试为官方 val2014 全部 5000 图；没有独立验证集，历史权重按公开 test 选模。
四图 pilot 的 CC 只能标为流程诊断，不能当作完整测试指标。

## 硬件合同

- 原仿真：532nm、17µm、478×478 有效区、10cm。
- 8µm 面板：有效区按物理尺寸重采样为1016×1016；振幅1920×1080、相位1920×1200。
- 振幅使用双线性重采样；相位最近邻。相位翻转和 `255-g` 仅按 LAB 配置在导出时做一次，SDK原样显示BMP。
- 最新 ROI/方向来自本次 LAB JSON，不从相机自动猜测。不允许采集中途改配置；改变曝光/ROI必须新建会话。
- CCD仅透视到478×478并保持原DN，PNG不做逐图对比度拉伸；原模型的mean-only归一化保留。
- 理论 CCD 与显著性预览的PNG仅用于观看，数值保存在NPY；预览不能拿来代替测量输入。
- 标准评估按训练工程原规则关闭随机DC/扰动；实际CCD已经包含物理零级分量，不额外注入第二份。

## 师弟电脑执行

已有 `ABO_Lab_SHS_8um` 的相机/振幅驱动和GPU环境，关闭相机和振幅GUI。
相位由另一台电脑SDK持续保持。本工程不会接管/修改硬件电压或LUT。

```powershell
Set-Location E:\code\guest\2026OpticsMoE\SALICON_Lab_SHS_8um
$py = '..\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe'
& $py run.py init --session pilot01 --fields 4
& $py run.py prepare --session pilot01 --stage vision_router --device cuda
# 本地SDK已加载 sessions/pilot01/phase/01_vision_router.bmp 并保持后：
& $py run.py capture --session pilot01 --stage vision_router --phase-ready
& $py run.py prepare --session pilot01 --stage vision_expert --device cuda
# 换 02_vision_expert.bmp 并保持后：
& $py run.py capture --session pilot01 --stage vision_expert --phase-ready
& $py run.py prepare --session pilot01 --stage vision_global --device cuda
# 换 03_vision_global.bmp 并保持后：
& $py run.py capture --session pilot01 --stage vision_global --phase-ready
& $py run.py evaluate --session pilot01 --device cuda
```

全量使用新会话 `init --session full01 --fields 0`，同样按三个阶段执行。
采集断线后可重跑同一个capture：只跳过身份和SHA验证通过的已采PNG，不跳过准备/实测阶段。
本地协调器支持 `lab_manual_stage --task salicon`；每次只保持一层，采集并准备下一层后等待监督程序换层。
自动化使用相位存活租约，失联停止本次采集，不能静默以仿真CCD补齐。

## 看哪些文件

- `release.json`：固定权重、原测试身份、源码commit、全量仿真复现和缓存/三CCD回放误差。
- `sessions/<会话>/status.json`：采集进度；`logs/`：命令日志。
- `sessions/<会话>/ccd/<阶段>/*.png`：真实ROI输出及旁边的哈希/曝光记录；不保存TIFF。
- `sessions/<会话>/saliency_previews/`：前16张预测和真值；不挑样本。
- `sessions/<会话>/results.json`：逐图CC及整体CC/KLD/SIM/NSS/AUC等。指标是CC，不是LGVQ的SRCC。

不要删除原会话或覆盖权重来“修复”结果。任何后续微调都应另建训练run，保留本次原始固定权重基准。
