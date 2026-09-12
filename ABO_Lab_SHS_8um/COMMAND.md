# 操作顺序（师弟 Windows 高速相机电脑）

MNIST v2 已训练相位的独立反灰度导出：在工程根运行
`python export_mnist_phase.py`，读取 `assets/mnist_v2_original` 中原 BMP/清单，
写入新的 `generated/phase_inverted/mnist_v2`。只做 `255-g`，保留原中心(980,590)
及原有空间方向，输出1920×1200。该目录存在时拒绝覆盖；无需连接设备。
配套说明见输出目录README，不要与ABO的输入和探测ROI混用。

2026-09-12 联调状态见 [JOINT_RESULTS.md](JOINT_RESULTS.md)。GPU 模型准备已通过；
联合控制已验证，旧帧问题已修复。当前 `LAB.local.json` 使用150 μs曝光、100 fps、
200 ms持续取帧等待；20次交替输入全部对应正确。**新CCD四角还未标定，不能开始六阶段正式采集。**
这一结果不等于已经复现 ABO 实测准确率。

## 1. 进入新工程

关闭 Flex Io Viewer / FastStream 的采集窗口，SDK 需要独占采集卡。PowerShell：

```powershell
Set-Location E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$gpu = Join-Path (Get-Location) '.venv_gpu\Scripts\python.exe'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
```

这套 `.venv` 已安装并实测，不用激活 conda，不用修改 Luceda 环境。相机采集只需 Python x64、NumPy、Pillow；不需要 torch。换电脑先安装原厂采集卡驱动/SDK，用 Python 3.12 x64 创建 venv，再 `& $py -m pip install -r requirements.txt`。

## 2. 检查连接（不取图、不改参数）

```powershell
& $py probe.py --out "results\probe_$stamp"
```

结果：`probe.json` 和设备 XML。期望型号 SHS-202-M、Mono8、1920×1080。若提示 already open，关闭 Viewer 后重试；不要强杀未知进程。每次重试重新生成 `$stamp`，输出目录不能覆盖。

## 3. 拍图、曝光和吞吐诊断

先用配置里的 100 fps、1000 μs 拍 3 张：

```powershell
& $py capture.py --out "results\scene_$stamp" --frames 3
```

多曝光对照（单位 μs，先设 100 fps）：

```powershell
& $py capture.py --out "results\exposure_$stamp" --fps 100 --exposures-us 100 500 1000 3500 --frames 3
```

数字条纹检验（不是光学拍摄结果）：

```powershell
& $py capture.py --out "results\digital_$stamp" --fps 100 --exposures-us 1000 --test-pattern VStrip --frames 3
```

短时主机取帧吞吐（不写 PNG，不等于光电系统速度）：

```powershell
& $py benchmark.py --out "results\speed_$stamp" --fps 1000 --exposure-us 100 --frames 200
```

以上命令结束都会恢复运行前的相机曝光/增益/帧率/测试图设置，结果中检查 `complete=true` 与 `restored=true`。如恢复失败，读取 `restore_errors`，不要继续盲采。

首次开流默认预热 2 秒并持续丢弃帧（`camera.startup_warmup_s`），防止启动阶段全零
缓存被保存为真实图。`capture.json` 的 `startup_warmup` 记录耗时和丢弃帧数。
这是每次打开相机的一次性等待，不是 SLM 每换图都等 2 秒。不要为了省时直接关闭。

文件 `e00_f0000.png` 为原始强度，没有增强。全暗可能是暗场，也可能是启动异常，不能只凭回读成功判断。白天测试已观察到曝光响应，但自然光变化、无暗帧扣除，不能当作正式线性标定。

联合控制/ABO 只编辑 `LAB.local.json`，它不会被代码升级覆盖。
独立 `capture.py` 默认读 `config.json`；如需使用同一套设置，显式加
`--config LAB.local.json`。默认 Mono8 原样保留。特别注意 2250 fps **不能曝光 3500 μs**。

也可临时在 capture 命令后加 `--gain Gain_X1`（还支持 X2/X4/X8）；设置回读和恢复均写入 capture.json。

## 4. 离线生成标定 + 六层 ABO 相位（不连接设备）

```powershell
& $py calibrate.py --config LAB.local.json --phases
```

文件都在 `generated\phase_inverted`：

|目录/文件|用途|
|---|---|
|`cal\A_WHITE.bmp`、`A_BLACK.bmp`|振幅全白/全黑|
|`cal\A_ACTIVE.bmp`|1016 px 左右的有效振幅区域|
|`cal\A_L.bmp`、`A_TL/TR/BR/BL.bmp`、`A_DIGIT_*.bmp`|判断镜像/方向；仅用于对齐，不是识别任务结果|
|`dual\01_check64\A.bmp` + `P.bmp`|配套棋盘格和局部交替方向光栅，先用这组调位移台|
|`dual\04_check16`|更细的配套棋盘格；A/P 必须同一子目录|
|`dual\02_blocks_x` / `03_blocks_y`|大块振幅 + 局部 X/Y 光栅|
|`cal\P_ZERO.bmp`|本实验室零相位是灰度 255，不是黑图|
|`cal\P_F4.bmp`|10 cm 四个相连菲涅尔子区，中心在有效 ROI 四角|
|`P\01_vision_router.bmp` … `P\06_language_global.bmp`|六层正式 ABO 相位，1920×1200 原生显示|

`P_V.bmp` 是额外的上下翻转对照，不要与灰度反向混淆。图像查看器缩放仅供查看，实际 SLM 必须原生像素 1:1 显示，禁用窗口边框/自动缩放。

## 5. 后续接入振幅 SLM：先单张联调

本机 SDK 路径已填写在 `LAB.local.json`，先确认 1920×1080/8 μm 面板是振幅面板。
下面命令必须在**该电脑已经登录的桌面 PowerShell**运行，不能直接在无桌面 SSH 会话中显示 SLM。
确认相位面板手动加载正确的 P.bmp。

```powershell
& $py slm_camera.py --config LAB.local.json --bmp generated\phase_inverted\dual\01_check64\A.bmp --out "results\paired_$stamp"
```

相位不会自动切换。程序顺序：振幅 Visible → 等待期间**持续取帧丢弃** → 额外丢弃6帧 → 取新帧 → raw.png。
不能只 sleep 后丢6帧，这一旧方法已被实测证伪。200 ms下 Visible 至取图约276 ms，不含BMP加载及磁盘保存。
相机本身的速率不是 Holoeye 60 Hz 的播放速率。测试结束 SDK 会关闭显示窗口，不能假定退出后仍保持测试图。

确认 CCD 确实随输入变化后，先在全黑相位下做短测：

```powershell
& $py joint_diagnostic.py --mode checker --exposures-us 100 150 200 --out "results\exposure_$stamp"
```

这只采 9 帧。已测150 μs无饱和，200 μs有少量饱和；当前无需再用旧3500 μs诊断值。
换光强/相位后重新检查，再测时序：

```powershell
& $py joint_diagnostic.py --mode timing --timing-exposure-us 150 --out "results\timing_$stamp"
```

先采 4 张独立参考检查左右图能否区分，再采 24 张交替图；参考差异不足会中止，
防止静止错误画面也得到高 PCC。最长等待参考不等于物理真值，不能只看其自身误差为零。
当前短测0/20/50/100 ms有错帧，200/400 ms本轮正确；200 ms追加20次交替全部正确。
这不是长期零错帧保证。复核当前推荐值：

```powershell
& $py joint_diagnostic.py --mode timing --timing-exposure-us 150 --delays-ms 200 --repeats 10 --out "results\repeat_$stamp"
```

## 6. 四点标定填哪里

CCD 硬件保持全幅；把四个逻辑角的**新 SHS 全传感器坐标**填入 `LAB.local.json` 的 `logical_corners_full_sensor_xy`。这些坐标不要求 4 的倍数。相位 ROI 中心：

```text
TL=(451.625,91.625)     TR=(1467.375,91.625)
BL=(451.625,1107.375)   BR=(1467.375,1107.375)
```

这是 SLM 侧的中心，**不能直接抄成 CCD 四点**。逻辑标签用单角/非对称输入确认，不按相机屏幕左右排序。确认后设 `geometry_confirmed=true`；Mono8 的 `capture_input_range` 保持 `[0,255]`。新相机不用旧 DVP 的 ROI，也不要重新逐图拉伸。

## 7. 正式 ABO：等接入光路、导入模型后再做

师弟电脑现已导入固定 checkpoint、原始 2400 查询数据及模型前端参数，
不再需要复制旧工程。`INFERENCE_ASSET_MANIFEST.json` 记录逐文件 SHA。
GPU 环境 torch 2.8.0+cu126 / transformers 4.57.3 已通过模型测试。
换机部署的已测试依赖及 CUDA 安装顺序见 `requirements-gpu-tested.txt`。
相机 `.venv` 不含 torch 是有意分离，不是安装了 CPU 版 torch。
如换一台电脑且尚未导入资源，才使用旧工程导入方式：

```powershell
& $py import_abo.py --source ..\ABO_Lab_8um
```

只复制模型、runtime、checkpoint、原始 test_dataset；不复制旧 CCD/旧设备参数。
本机可先做不占用光路的检查，输出 4 张 router 振幅 BMP：

```powershell
& $gpu model_check.py --out "results\cuda_check_$stamp"
```

四角标定、曝光/时序验证完成后，在登录桌面设置 `python` 为该 GPU 环境
（或将下面每个 `python` 换成 `& $gpu`），先 4 查询小样本，新建 session：

```powershell
python run.py init --session shs_pilot01 --limit 4
```

每层先手动加载 `P` 目录中对应编号的相位，再执行该条；自动准备+采集，不要一次粘贴六条让相位来不及切换：

```powershell
python run.py stage --session shs_pilot01 --stage vision_router --device cuda
python run.py stage --session shs_pilot01 --stage vision_expert --device cuda
python run.py stage --session shs_pilot01 --stage vision_global --device cuda
python run.py stage --session shs_pilot01 --stage language_router --device cuda
python run.py stage --session shs_pilot01 --stage language_expert --device cuda
python run.py stage --session shs_pilot01 --stage language_global --device cuda
python run.py evaluate --session shs_pilot01 --device cuda
```

语言阶段是 100 候选标题+4图像=104 张；不是重复采集。六层共 324 次曝光。全量 2400 查询共 14700 次。跳过确认用显式 `--yes`，但仍必须手动加载相位；首次联调建议保留确认。

默认 `save_raw_frames=false`：正式采集每张只留 canonical 478×478 PNG 和必要 JSON/记录，
不另存全幅 raw PNG/TIFF；PNG 压缩级别 1，不做逐图拉伸、log/gamma。
诊断短测仍保留少量全幅 PNG 供定位问题。需要排查原图才显式改为 true。
目前只验证 GPU 模型和第一阶段准备，**未完成新光路六阶段实测准确率**。
勿复用旧 session 名称/record。改曝光或几何后重新建 session，不能混用旧 CCD。
