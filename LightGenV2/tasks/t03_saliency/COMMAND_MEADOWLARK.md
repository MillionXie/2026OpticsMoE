# SALICON 0.8625：17µm Meadowlark + 8µm手动相位 + TUCam

这是三次传播：Router → Expert → Global。不要运行旧的 `run.py capture`（那是SHS设备）。
解压到短路径，例如 `E:\lab\salicon8625`。本文件所有命令从解压根目录运行。
Python环境沿用已经能驱动Meadowlark/TUCam的环境。本机需安装厂商驱动；SDK/LUT沿用原LGVQ硬件包，
不要从Linux服务器安装/打开设备。包内包含Python SDK控制代码，不重复附送厂商安装包。

## 1. 完整性与仿真

```powershell
Set-Location E:\lab\salicon8625
conda activate xml
python handoff.py verify
python handoff.py simulate --fields 4 --device cpu
```

4图分数约0.893689，仅用于回放检查；5000图正式仿真CC是0.8624925。
需要全量复现时：`python handoff.py simulate --fields 0 --device cuda --output sim5000`。
输出目录必须未使用过；不覆盖旧结果。

## 2. 本机标定与绑定（唯一需要按本机改的部分）

先使用你原来的LGVQ实验硬件包，确认两个SLM对齐、四角CCD几何、LUT、曝光、等待/丢帧已标定。
这个新任务不需要重新标LUT，但不可以借用他人机器的ROI/翻转配置。
在原工程运行其 `prepare_lab` 后，选用**本机已验证的** `generated/formal_hardware.yaml`。
保持该文件在原位置，SDK/LUT的相对路径才不会失效。
以下交互会让你选择这个已有文件，随后只需修改相位中心/方向/灰度编码为本机标定值。

```powershell
$hardware = Read-Host '粘贴本机已标定 formal_hardware.yaml 的完整路径，不带引号'
Test-Path -LiteralPath $hardware
python meadowlark.py bind --hardware-config "$hardware" --phase-center 960 600 --phase-orientation none --phase-gray-encoding normal
```

**`960 600 / none / normal`仅是面板居中示例，不保证就是你的平台设置。**
`--phase-center X Y` 是BMP连续边界坐标中心；面板1920×1200居中为960、600。
若旧配置使用像素中心坐标，先加0.5转换，不要凭感觉平移。
方向可选 `none/h/v/hv`；灰度编码可选 `normal/inverted_255_minus_g`。
振幅默认不翻转，必要时显式传 `--amplitude-orientation h` 等。
CCD方向由四角逻辑标签的homography解决，之后不会再做翻转。

绑定生成 `LAB_Meadowlark.json`，记录原YAML、LUT和几何文件的SHA。
不会打开设备，也不会替换LUT。以后若改曝光、LUT、ROI、等待或方向，
使用新绑定文件（`bind --config LAB_new.json ...`）和新session，并在所有命令显式传 `--config LAB_new.json`。
原YAML应使用：`max_files: null`、PNG、478×478、8位固定映射、已启用的几何合同、
`require_phase_mask: true`、`confirm_before_start: true`、固定曝光；不接受自动逐图拉伸。

## 3. 先做4图完整闭环

```powershell
python meadowlark.py init --session pilot4 --fields 4
python meadowlark.py prepare --session pilot4 --stage vision_router --device cuda
python meadowlark.py check --session pilot4 --stage vision_router
python meadowlark.py capture --session pilot4 --stage vision_router
python meadowlark.py audit --session pilot4 --stage vision_router
python meadowlark.py prepare --session pilot4 --stage vision_expert --device cuda
python meadowlark.py check --session pilot4 --stage vision_expert
python meadowlark.py capture --session pilot4 --stage vision_expert
python meadowlark.py audit --session pilot4 --stage vision_expert
python meadowlark.py prepare --session pilot4 --stage vision_global --device cuda
python meadowlark.py check --session pilot4 --stage vision_global
python meadowlark.py capture --session pilot4 --stage vision_global
python meadowlark.py audit --session pilot4 --stage vision_global
python meadowlark.py evaluate --session pilot4 --device cuda
```

`init`产生 `sessions/pilot4/phase/01_vision_router.bmp`、`02_vision_expert.bmp`、`03_vision_global.bmp`。
每次capture之前，**手动在相位SLM显示对应的BMP**。命令会显示文件名及SHA并要求输入y确认；
phase为纯黑只适用于标定，不适用于这三个正式阶段。
振幅BMP由程序自动逐张加载；`check`只检查SDK/LUT/几何/BMP，不打开设备。
首次检查通过不代表光路已对齐，先观察四图、饱和比例与输出方向，再做全量。

后两次prepare严格依赖前面实测CCD，不使用参考BMP冒充闭环；每张相位、振幅、CCD有SHA记录。
采集调用原生 `acquire_folder`，先ImageWriteComplete，再原配置的settle等待，然后CCD捕获。
曝光、warmup、丢帧来自原YAML；读取日志 `sessions/pilot4/sdk_log/<stage>/resolved_devices.json` 核对实际设备。
CCD PNG是SDK几何校正+固定8位映射，不是传感器原始图；无log/gamma/逐图minmax，不再次warp。
网络读取除255并沿用训练中的mean-only处理。理论CCD的PNG仅作显示，NPY才是数值场。

若完整采集已写好capture_manifest.csv，但导入时中断，可在核查后运行：
`python meadowlark.py import-captures --session pilot4 --stage vision_router`（阶段按实际更改）。
不完整采集请保留原session日志并新建session重做；不要使用clear-output覆盖证据。

## 4. 正式5000图

使用 `python meadowlark.py init --session test5000 --fields 0`，然后照第3节顺序执行，
所有 `--session pilot4` 改为 `--session test5000`，总共15000次CCD捕获。
先确认磁盘空间：1024²振幅BMP三阶段约15GB，另有CCD/日志；建议留至少25GB。
结果在 `sessions/test5000/results.json`，包含实测CC、相同子集仿真CC、逐图身份和指标。
这台设备尚未实采验证此适配器，不能保证复现另一台SHS设备的实测分数。

## 5. 后续AI微调边界

本包有完整模型/训练Python源码、best权重和原配置，但5000图缓存是test，不是train。
当前没有已验证的一键实测微调CLI。不要拿这些test图当训练集然后继续声称独立测试。
需要追加train2014输入/标签/前端缓存，先固定光学和上游输入映射，采集train子集，
再适配电子尾部；上游映射变了就要重采。详见 `00_START_HERE.md`。
本次不附周期快照，只附选定best；不更换模型或降低alpha来凑实测分数。
