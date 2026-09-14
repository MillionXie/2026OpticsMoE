# Temporal 0.8044 固定权重：SHS六层实测

这是一次固定权重硬件前向评估，不是重新训练、微调、挑选样本或纯光学模型的准确率。
六层实测CCD进入原始光电计算图；电子残差及读出头保留。

## 身份与结果

- Run：`runs/hardware/temporal_full_20260914`；统一入口为其中`00_查看这里.md`。
- 模型：`multivideo16x4_rank_s163`；Temporal，16个独立视频×各4帧，四专家Top-2光路由。
- checkpoint SHA256：`5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`。
- 推理包源码commit：`8e869473787f4ffceb2a6a77f4430b94c206f459`；本地单层协调器commit：
  `992ae70fa7fa8d424e5e23fc5d8ad8ab51340e89`。协调器更新不改变推理模型。
- 完整原test划分558个视频；35幅场/层，每幅16槽位，最后2个padding槽位不计指标。
- 210张正式CCD；6层全部采集，所有视频保留。另有第五层2张诊断复拍，不替换正式数据。
- 原模型按训练时固定prompt/Qwen前端缓存推理，未联网重提特征，未加载新的大模型后端。

| 指标 | 完整固定权重仿真 | 六层硬件实测 |
| --- | --- | --- |
| SRCC | 0.8043868643075132 | 0.7977138739203681 |
| KRCC | 0.5968244683622915 | 0.5901795637723111 |
| PLCC | 0.8180329373662566 | 0.8091420695614706 |
| RMSE | 7.99107551574707 | 8.208910942077637 |
| MAE | 5.992058753967285 | 6.1252875328063965 |

逐视频原始分数和标签在`results.json`。固定558个唯一视频，无后验筛选；局部复拍没有参与指标。
独立SciPy复算证据在`independent_metric_check.json`。权重、缓存原manifest身份见部署包release.json，
完整CCD/上游依赖验证见`measured_data/measurement_audit.json`及measurement_SHA256.json。
实测归档SHA256：`694869d4c26d9305eec52a54a9a101e9d9415acbc03f82c35f26af07dd32e378`。

## 硬件与命令

相机SHS-202-M，Mono8，1920×1080全帧；400μs、Gain_X4、相机100fps。
输入Holoeye 1920×1080、8μm；相位Meadowlark HDMI 1920×1200、8μm，linearVoltage LUT，
相位BMP已做既定上下/左右变换及255-g反相。17μm模型478有效区按物理尺寸映射到约1016像素。
传播10cm；换输入后持续排空240ms再取新帧。推理使用师弟电脑既有CUDA Python/RTX4060。
相位由本地SDK主线程保持，换层前先确认采集完成，旧SDK退出后才创建下一层所有者。

新ROI来自`E:/code/guest/20260914.txt`：相机TL[656,153]、TR[1474,156]、
BL[656,967]、BR[1471,970]；沿用此前左右镜像映射到逻辑坐标。
硬件配置SHA256：`0c06f647a52b8767b4d092253b4aafb8f5d434fba599bacf4f8e61c1420258a4`。
输出478×478透视ROI PNG，不逐张min-max拉伸；CCD固定DN/255进入原网络既有读出归一化。

在师弟电脑的`E:/code/guest/2026OpticsMoE/LGVQ_Temporal_Lab_SHS_8um`：

```powershell
$py = '..\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe'
# 复现实采必须创建新的会话名，不覆盖此run。
& $py run.py init --session new_measurement --fields 0 --config LAB.20260914.json
& $py run.py prepare --session new_measurement --stage vision_router --device cuda --config LAB.20260914.json
# 确认对应相位已加载保持，然后逐层capture并prepare下一层；所有层都指定同一config。
& $py run.py capture --session new_measurement --stage vision_router --phase-ready --config LAB.20260914.json
& $py run.py prepare --session new_measurement --stage vision_expert --device cuda --config LAB.20260914.json
# 按vision_expert、vision_global、language_router、language_expert、language_global依次执行。
# 完成全部六层后评估；复算本轮结果则无需再次capture：
& $py run.py evaluate --session temporal_full_20260914 --device cuda --config LAB.20260914.json
```

本轮实际协调器的完整命令和PID保存在各层`*_launch.json`；每层日志及相位SDK回执保存在对应目录。
依赖为已有Windows设备SDK、CUDA torch、numpy、Pillow、opencv-python、PyYAML；本地协调器另用paramiko和pywin32。

## 限制

第五层p99仅16~18，信号较弱；两次相同设置复拍相对原图PCC为0.96317、0.96646，保留原数据。
第一层前三幅实测与仿真强度PCC约0.30，因此不能把最终SRCC较接近视为光场完全匹配。
未进行本轮去光消融或相位光学标定认证，不据此量化光学贡献。SDK返回不等于独立的光学正确性证明。
未测本轮全系统能耗，也不将曝光或240ms等待当作整个任务端到端耗时。
