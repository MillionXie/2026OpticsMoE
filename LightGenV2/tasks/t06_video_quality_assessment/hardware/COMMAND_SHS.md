# LGVQ 当前实验台：逐层操作

维护者在已同步Git commit的训练服务器构建：

```bash
python -m LightGenV2.tasks.t06_video_quality_assessment.build_lab_package \
  --bench shs --target temporal --source-root /path/to/source_repo \
  --output /path/to/T06/releases/20260914_shs_temporal08044
# Spatial把target改为spatial，输出另一个目录；默认包含558条完整test。
```

此包只用于其 `release.json` 指定的一个任务，禁止互换权重或CCD。
Temporal固定SRCC参考0.8044：16个视频×4帧，一幅场输出16个独立时间MOS；Spatial固定参考0.6710：一个视频×4帧，一幅场输出一个空间MOS。两者各有6张相位，不能用单视频36帧模型替代Temporal。

## 设备与计算分工

师弟电脑：Holoeye振幅SLM 1920×1080、8μm；SHS高速相机；RTX4060负责缓存输入→下一层BMP及最后分数。
本地电脑：Meadowlark HDMI相位SLM 1920×1200、8μm，既定linearVoltage LUT。只按用户通知换层。
相位BMP依照配置进行既定方向变换和255-g反相，生成后不要再反相。

新工程复用同级 `ABO_Lab_SHS_8um` 中已部署的SDK/驱动与GPU Python，**不修改ABO会话、不启动ABO采集**。Python依赖torch(CUDA)、numpy、Pillow、PyYAML、opencv-python；控制相位另需本地已部署Blink SDK。

## 一次初始化（师弟电脑登录桌面终端）

进入解压后的LGVQ工程目录。复制这台电脑最近使用的、已经确认过几何的硬件配置为 `LAB.local.json`。迁移工具会完成这一步；不要直接拿旧模板config.json代替。

```powershell
$py = '..\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe'
# 首轮先4幅场：Temporal=64条视频，Spatial=4条视频。全部558条用 --fields 0。
& $py run.py init --session pilot01 --fields 4
& $py run.py prepare --session pilot01 --stage vision_router --device cuda
```

模型只逐幅加载缓存，不需要联网或加载完整Qwen。缓存包含对应Spatial/Temporal prompt的冻结Qwen前端结果，并未删除文本分支。当前是固定权重采集/评估包，不会自动重训或更换模型。

## 每层：必须等用户通知换层

1. 本地加载 `sessions\pilot01\phase\01_vision_router.bmp` 并保持。六张相位可以从师弟电脑复制至本地，完全一致，不要额外翻转。
2. 用户确认本层相位已加载，才在师弟电脑执行：

```powershell
& $py run.py capture --session pilot01 --stage vision_router --phase-ready
& $py run.py prepare --session pilot01 --stage vision_expert --device cuda
```

这里prepare只是读取上游实测CCD、计算下一层输入；**不会切相位，也不会开始下一层采集**。采集或prepare结束后进程退出，不偷偷运行下一层。没有用户指令不得运行下一个capture。

六层顺序和相位文件：

| 次序 | --stage | 相位文件 |
| --- | --- | --- |
| 1 | vision_router | 01_vision_router.bmp |
| 2 | vision_expert | 02_vision_expert.bmp |
| 3 | vision_global | 03_vision_global.bmp |
| 4 | language_router | 04_language_router.bmp |
| 5 | language_expert | 05_language_expert.bmp |
| 6 | language_global | 06_language_global.bmp |

每层使用同样capture、prepare下一层两条命令。第六层采完，不再prepare，而是：

```powershell
& $py run.py evaluate --session pilot01 --device cuda
```

本地若用SDK保持相位（不触碰相机和振幅），在包目录设置 `$env:PYTHONPATH=(Resolve-Path .\runtime)` 后执行：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.lab_phase `
  --bmp .\phase\01_vision_router.bmp `
  --bench-root ..\ABO_Lab_SHS_8um `
  --link-config ..\ABO_Lab_SHS_8um\results\20260913_400us240ms\link_full.json
```

保持进程运行；下一次用户要求换层时Ctrl+C退出旧保持，再执行新文件。SDK返回/SHA匹配不等于光学响应认证。相位GUI不能同时占用。

## 查看结果与恢复

对于明确授权自动换层的任务，本地协调器为`lab_supervise`。若它中断而当前单层已完成，
先确认旧相位持有器的`report.json`是`next_inputs_ready_wait_for_user`且远端采集进程结束，
再在该持有器目录创建`RELEASE`，等其退出。用新的监督输出目录执行原supervisor命令并加
`--resume-completed`：它重新审计已有层的PNG/相位/输入/上游SHA链，只从第一个未采集层继续，
不清空CCD、不改曝光、不重建session。新的监督状态在新目录的`status.json`。
JSON读取遇到临时Windows/OneDrive共享访问拒绝会有限重试，持续拒绝仍明确报错，不伪装成功。

- 每个会话只有一个入口：`sessions\pilot01\status.json`。
- 本层输入：`sessions\pilot01\play\<stage>\*.bmp`。
- 正式CCD：`sessions\pilot01\ccd\<stage>\field_*.png`，478×478、ROI透视校正，固定0–255，无逐张对比度拉伸。只保存PNG及小型记录，不保存TIFF。
- 理论场：`sessions\pilot01\theoretical_ccd\<stage>`，NPY为数值，PNG为p99.5展示，**展示缩放不用于推理**。后续理论场由已采集的上游CCD驱动，不是完全独立的纯仿真链。
- 最后 `sessions\pilot01\results.json`；只有all SIX measured之后才能evaluate，没有缺层仿真兜底。
- 中断后重复同层capture，已校验SHA的有效PNG会跳过；近暗底、饱和图留在rejected，不会混入有效数据。
- 相位、输入、上游CCD、硬件配置和权重均封存身份；改曝光/ROI/LUT/朝向后必须新session，不混采。

## 不应混淆的数值合同

训练模型是17μm像素、478有效区域、518传播画布、532nm、10cm；输入和相位导出到8μm硬件时维持物理尺寸，约1016×1016有效区域，不重新训练、不缩成478硬件像素。
训练含20%名义未调制光；这里不在实测CCD上再人为叠加20%直流，也不把复数调制的模乘进振幅第二次。
CCD先固定转换DN/255，再由原网络执行其训练时的归一化/对数读出。可在新硬件会话中显式设置 `detector_intensity_scale` 的六层增益用于光度标定，不能拿每张理论结果拟合其增益。相机响应、LUT、对齐和物理直流差异意味着**仿真参考SRCC不保证实测达到同值**。
目前400μs+240ms是ABO条件下的起始设置，不代表已通过LGVQ曝光扫描。首轮应先核对各层亮度，不盲目跑全部。

## 下游层曝光不足时

`camera_exposure_us_by_stage`可显式指定某一层的曝光；未指定层仍用`camera.exposure_us`。
先做独立曝光诊断，保持增益和等待不变。改变曝光后新建配置、新建会话，不能直接改正在采集的旧配置。
若第四层从400μs改1600μs，显式设`camera_exposure_us_by_stage.language_router=1600`，
同时设`detector_intensity_scale.language_router=1/(255*4)`作曝光倍率补偿。此线性补偿不是暗场标定，不能恢复弱信号SNR。

已有前三层可用`lab_exposure_session`在新会话中继承：它必须逐层验证有效曝光、ROI、LUT、方向、读出尺度等完全相同，
验证原CCD、输入、相位和上游SHA。保留原会话；新记录附原记录SHA及原采集硬件身份，不能宣称重新采过继承层。

```powershell
# 在含该模块的已校验Git版本/更新包runtime上运行；参数仅为示例。
$env:PYTHONPATH=(Resolve-Path .\runtime)
& $py -m LightGenV2.tasks.t06_video_quality_assessment.lab_exposure_session `
  --source-session spatial_full_20260914 --new-session spatial_lr1600_20260914 --config LAB.lr1600.json
& $py run.py prepare --session spatial_lr1600_20260914 --stage language_router --device cuda --config LAB.lr1600.json
```
