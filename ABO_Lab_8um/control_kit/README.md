# Holoeye 振幅 SLM + DVP 相机控制包

这是当前 ABO 8 μm 实验光路的**独立设备控制包**，不是模型/微调包。先读本文，再按顺序执行。
不需要原 2026OpticsMoE 仓库、Qwen、数据集、权重、PyTorch 或 CUDA Python 包。

## 1. 对应设备与已验证边界

| 设备 | 当前实验台 | 本包接口 |
|---|---|---|
| 振幅 SLM | HOLOEYE，1920×1080，8 μm，60 Hz | SLM Display SDK，3.2.2 安装包，API version=5 |
| 相机 | camera_dvp_legacy，当前实机 MONO8，5480×3648 | DVP，CPython 3.6 x64 子进程 |
| 相位 SLM | 1920×1200，8 μm | **人工加载，本包绝不控制** |

这不是之前高速 Meadowlark 17 μm + TUCam/Mosaic 的包；不要拿错设备或套用 Meadowlark LUT。
2026-09-10 已通过 SSH 只读核对：包中 `device_driver.py`、`legacy/dvp_capture_worker.py`、
`dvp.pyd`、`DVPCamera64.dll` 与当前实验电脑运行版本的 SHA256 完全一致。
这套底层驱动已用于真实采集；新增独立调用壳做了离线测试，**此次没有重新开光路实测**。
`LAB_VERIFIED_SOURCE.json` 保留核验值，`MANIFEST.json` 记录本次代码 commit 与每个文件哈希。

## 2. 环境安装（Windows 64 位）

建议解压到短路径 `D:\SLM_CCD`，进入能看到 `control.py` 的那一层。
必须在连接 SLM 的本地登录桌面运行，不要直接从 SSH 的非交互桌面打开 Holoeye；
如果通过远程桌面操作，须确认显示器枚举/扩展桌面没有改变。两块 SLM 同时连接时，
先在厂商配置中确认 SDK 所选的是**振幅屏**，不要让 SDK 选中手动相位屏。

先安装厂商软件：

1. HOLOEYE SLM Display SDK（当前实机 v3.2.2），含 Python wrapper、win64 DLL 和必要授权。
   本包不包含其商业安装器、授权文件；请使用自己设备随附的安装介质。
2. DVP 相机厂商驱动/查看软件及其运行库（Windows USB/GigE 驱动和 DLL 依赖须正常）。
   包含现有 SDK 的 `dvp.pyd` 和 `DVPCamera64.dll`，**这两个文件不是完整硬件驱动安装器**。
   厂商文件仅供已获授权设备的实验协作使用，不要上传到公开代码仓库。
3. 正常可用的显卡驱动及扩展显示屏。SLM 显示会使用 GPU，但**无需安装 torch 的 GPU 或 CPU 版**。

在 Anaconda PowerShell 中创建主环境（不要修改已有训练环境）：

```powershell
Set-Location D:\SLM_CCD
conda create -n slm_control312 python=3.12 -y
conda activate slm_control312
python -m pip install -r requirements.txt
```

相机单独用旧 ABI 环境。已有能调用该 DVP 相机的 Python 3.6 x64（例如 RFL）时直接复用；
否则创建如下环境。Python 3.6 已停止维护，只用于本地厂商驱动，不运行训练/联网服务。
旧包能否下载安装依赖所用 Conda 源；本 ZIP 不含完整离线 Python 环境。

```powershell
conda create -n dvp36 python=3.6 numpy=1.19.5 -y
conda run -n dvp36 python -c "import sys,struct,numpy; print(sys.executable); print(sys.version); print(struct.calcsize('P')*8); print(numpy.__version__)"
```

把上面打印的 **python.exe 完整路径** 填到 `config.json` → `camera.python_executable`。
例如 `C:/Users/YourName/.conda/envs/dvp36/python.exe`，JSON 中推荐 `/`。
或者保留 `%DVP_PYTHON%`，每次在当前 PowerShell 设置：

```powershell
$env:DVP_PYTHON = 'C:\Users\YourName\.conda\envs\dvp36\python.exe'
```

后续命令全部在 **slm_control312** 主环境运行。不要激活 dvp36 来运行 control.py。
主进程 Python 3.12 ⇄ JSONL/无损 NPY ⇄ Python 3.6 相机进程。
程序自动把 worker 与两份 DVP 二进制放到 `dvp_runtime/` 同目录，并加入旧环境 DLL 搜索路径。

## 3. 只改 config.json

| 配置 | 含义/需要做什么 |
|---|---|
| amplitude_slm.sdk_path | Holoeye Python 模块所在目录（包含 slmdisplaysdk 或 holoeye 包） |
| amplitude_slm.binary_folder | 包含 holoeye_slmdisplaysdk.dll 的 win64 目录 |
| expected_resolution_wh | 当前为 [1920,1080]；必须匹配振幅屏，不能填相位屏尺寸 |
| sdk_api_version | 当前 SDK 3.2.2 的实际 API 为 5；不要把安装版本号直接填这里 |
| preload / wait_until_visible | 保持 true；每次仅预加载一张，等待 SDK Visible，再开始软件等待 |
| camera.python_executable | 独立 CPython 3.6 x64 的 python.exe，不是主程序 Python |
| camera.sdk_path | 默认相对路径 vendor/dvp_py36_x64，通常不用改 |
| camera_index | 默认 0；多台相机时确认索引 |
| exposure_us | **微秒**。5000 μs=5 ms。示例用1000μs起步，须重新测亮度/饱和度，不保证适合新台架 |
| analog_gain / auto_exposure | 默认 1 / false；固定曝光采集，避免自动亮度干扰比较 |
| device_roi_xywh | null 表示不主动设置硬件 ROI，**不保证相机没有保留之前的 ROI**。看启动/记录中的实际 ROI；需要全帧时在厂商软件恢复，或填写该相机真实的 [0,0,width,height] |
| warmup_frames | 相机刚启动丢弃3帧，只执行一次 |
| discard_frames_after_display | 每次 capture 请求先丢弃2帧，包括同一 BMP 的重复采集 |
| settle_delay_ms | SLM 报告 Visible 后再等200ms，之后请求相机。不是曝光时间 |
| timeout_ms | 单次 GetFrame 最长等候时间，不是曝光时间 |

新的实验台必须自己确认 SLM 灰度/偏振响应、光路方向和曝光。这里不替换 LUT，不做相位标定。
原台架最近曝光5000μs，不应直接作为别人的默认标定结果。分辨率错误直接停止，不会偷偷缩放 BMP。

## 4. 从检查到采集（无 y 确认）

运行前关闭抢占设备的相机查看软件和振幅 SLM 播放程序。手动相位显示程序可以保持，
但须确认它与振幅 SDK 对应不同屏。**全白图可能使相机饱和，先弱光/短曝光测试。**

```powershell
conda activate slm_control312
Set-Location D:\SLM_CCD

# 离线代码测试；不打开设备、不要求安装厂商 SDK
python -m unittest test_control -v

# 检查路径、相机 Python 版本及位数；不打开设备
python control.py check

# 生成全屏黑、灰128、白、64像素棋盘格。仅诊断图，不是相位配准图
python control.py patterns --out patterns

# 仅打开相机，不碰SLM。输出目录必须不存在
python control.py camera --out captures\camera01 --frames 3

# 仅显示振幅，停留10秒，再关闭SDK。不碰相机和相位屏
python control.py slm --bmp patterns\checker64.bmp --seconds 10

# 显示一张BMP → 等待Visible → settle → 相机丢帧 → 连续采3帧
python control.py capture --bmp patterns\checker64.bmp --out captures\pair01 --frames 3
```

重复运行要换新的 `--out`（例如 pair02），不允许覆盖旧图。Ctrl+C 会释放上下文管理的设备，
已写出的数据保留；若进程强杀或厂商 DLL 卡死，必要时重新连接厂商设备。关闭 SLM SDK 后
显示是否保持取决于厂商状态，本包**不保证退出后自动黑屏**；需要遮光请使用物理遮挡或厂商操作。

数据：`0000.npy` 和 `0000.tif` 是相同原始像素，后者无损压缩方便直接看；`capture.json`
记录配置、实际曝光/ROI、输入 BMP SHA、shape、dtype、亮度统计、耗时、输出SHA。
没有逐图归一化、log、gamma、CLAHE、阈值、翻转、裁剪、单应变换或缩放。
软件查看器可能自己拉伸显示，视觉亮暗不等于保存码值不同。厂家若输出左移12-bit uint16，
“dtype_max_fraction”不等于真正传感器饱和比例，本包统计不作为曝光标定验收。
每帧保留 NPY+TIFF 两份，20MP图像注意磁盘空间。

## 5. 接入师姐自己的推理/微调程序

参照 `example_api.py`，最小调用：

```python
from control import Controller

with Controller('config.json') as hw:
    hw.display('input.bmp')  # 必须是本屏原生分辨率的8-bit L灰度BMP
    raw = hw.capture('captures/new_sample', frames=1)
    # raw: [H,W] uint8或uint16，尚未做网络需要的ROI/方向/归一化
```

把后续的 ROI、方向、线性标度及网络输入转换留在自己的项目中，且记录其配置；不要一边采集
一边静默改变图像含义。若逐张播放不同输入，在同一个 with 里循环 display/capture，
每个样本使用不同输出目录。这不是6层网络执行器，也不包含模型权重。

## 6. 常见故障与交接 AI 注意事项

- Holoeye 卡在 open：检查登录桌面、屏幕分配、SDK是否已被另一个程序打开；不是安装 CUDA 就能解决。
- DVP import/DLL load failed：确认 **3.6 x64**、厂商相机驱动、VC运行库及两个二进制完整；
  不能把 dvp.pyd 装到3.12，也不要用 pip 的同名包替换它。
- DVP 找不到相机：确认供电/连接/厂商查看器可识别，关闭占用相机的软件。
- 图全黑/太亮：看原始 min/max/mean 和实际 Exposure；先核对显示屏、曝光、偏振和光路，
  不用归一化伪造亮度，不禁用硬件错误去假装采集成功。
- 原始底层 devices.py 内保留其他历史设备分支以保持字节一致；本包入口明确只开放 Holoeye/DVP，
  未携带其他驱动，**不要通过修改driver字符串把它当高速SLM/TUCam包**。
- 不携带作者个人账户路径、密码、CCD四角、模型或原始数据。相机固有枚举与新电脑路径必须重查。
- 源码修改以 Git 为准；离线转交ZIP用 MANIFEST+SHA校验。不要修改原厂二进制。

开发者在仓库内重建：`python ABO_Lab_8um/control_kit/build_lab_package.py`。
生成文件位于 `ABO_Lab_8um/releases/`；builder只取指定Git commit源码和明确列出的两份厂商文件。
