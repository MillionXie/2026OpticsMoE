# 从这里开始：ABO / Holoeye 8µm / DVP

不要进入旧的Meadowlark/TUCam目录。本包从下面目录执行，使用独立Python，不用conda activate：

```powershell
Set-Location D:\code\guest\2026OpticsMoE\ABO_Lab_8um
$py = "$PWD\.venv\Scripts\python.exe"
```

## 0. 检查文件和环境

安装尚未完成的电脑才执行：

```powershell
& D:\anaconda\python.exe -m venv .venv
& $py -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
& $py verify_release.py
& $py test_lab.py
& $py simulate.py --device cpu --limit 4 --batch-size 1 --output local_smoke
```

仿真仍有真实数值精度差异，CPU小样本不是2400张80.583%的复现证据。服务器完整复核见证据目录和README。

## 1. 唯一配置文件

```powershell
Copy-Item lab.json LAB.local.json   # 仅首次；已有LAB.local.json不要覆盖
notepad LAB.local.json
```

先核对：Holoeye分辨率1920×1080、两屏中心、SDK路径。相位固定1920×1200。DVP保留旧Python3.6子进程，不要把DVP的pyd装进新Python。

必须实测填写：`capture_input_range`（相机真实码值范围，不是本张图最大值）、`logical_corners_full_sensor_xy`四个逻辑角、`geometry_confirmed`。先保持false。相机全幅采集，不要求四个角是4的倍数，也不照搬TUCam的硬件ROI整除约束。

这台DVP已实测为Mono8、5480×3648，初始值域已填[0,255]，不要改成之前TUCam的[0,65535]。若在相机软件中改变像素格式，必须重新核对。

曝光`camera.exposure_us`单位μs，起始3500=3.5ms。`settle_delay_ms`是SLM已经显示后的额外等待，初始200ms。`discard_frames_after_display`丢旧相机帧；`warmup_frames`只在打开相机时执行。Holoeye约60Hz，不是Meadowlark1.4kHz。

## 2. 生成并核对标定BMP

### 现在对齐：最短操作

文件已生成在 `D:\code\guest\2026OpticsMoE\ABO_Lab_8um\generated\cal`。
振幅为1920×1080 BMP，相位为1920×1200 BMP；两者按8μm像素设计。
478×17μm=8.126mm，对应1015.75（栅格约1016）个设备像素，不能把478直接作为显示宽度。

关闭其他占用振幅屏的播放器后，在**实验电脑桌面**运行；此命令不打开CCD，可同时使用CCD软件观察：

```powershell
& $py align.py --bmp generated/cal/A_CHECK_32.bmp
# 上一个命令按Enter结束后，再换下一张：
& $py align.py --bmp generated/cal/A_DIGIT_3.bmp
# 手动加载相位P_F4.bmp，再持续播放全白振幅：
& $py align.py --bmp generated/cal/A_WHITE.bmp
```

`A_ACTIVE.bmp`只照亮1016×1016有效范围，`A_WHITE.bmp`是整屏白；菲涅尔标定使用整屏白。
`A_DIGIT_0/1/2/3.bmp`来自旧实验的真实MNIST输入，只判断方向，不代表本项目执行了MNIST识别。
四角单独菲涅尔`P_F_TL/TR/BR/BL.bmp`可逐张确定焦点身份。四焦点共同出现时不能凭画面左右猜逻辑标签。
相位物理四角中心（x,y）为(451.625,91.625)、(1467.375,91.625)、
(1467.375,1107.375)、(451.625,1107.375)，间距1015.75像素；不是专家中心。

标定后，在`LAB.local.json`原有以下对象内填CCD**全传感器坐标**，然后将原有`geometry_confirmed`改true。
不要添加第二份同名字段；`null`必须替换为实际`[x,y]`，不是下面示例值。

```json
"logical_corners_full_sensor_xy": {
  "top_left": null,
  "top_right": null,
  "bottom_right": null,
  "bottom_left": null
},
"geometry_confirmed": false
```

相位上下翻转由`phase_slm.flip_vertical`控制。当前仍false，尚未实测确认；若确认应翻转，改true后运行`patterns.py`，
正式相位始终取`generated/P`。`generated/P_opposite_vertical`提供相反上下方向的对照版，勿混着采集。
CCD方向由上述逻辑四角单应变换处理，不再另加翻转。设置变化后必须新建会话，不能混用旧CCD。

### 完整标定检查

```powershell
& $py patterns.py
& $py run.py probe
```

`results/probe/raw.tif`是未翻转、未拉伸的相机原图。先关闭占用相机的CCD软件/旧SDK程序；相位继续保持全黑。

按顺序：

1. 相位手动放`generated\cal\P_ZERO.bmp`。播放非对称L和单角标记，记录新光路的方向；不能只看对称棋盘格猜翻转。
2. 棋盘格`A_CHECK_32.bmp/A_CHECK_64.bmp`配`P_GRAT_X.bmp/P_GRAT_Y.bmp`，确认两SLM物理范围和横纵光栅方向。
3. 振幅`A_WHITE.bmp`配相位`P_F1.bmp/P_F4.bmp/P_F9.bmp`。四焦点对应ROI四角，九焦点额外检查中点。角坐标按逻辑TL/TR/BR/BL填，不按CCD画面的左右排序。
4. 确认振幅/相位的相对方向，再设置两者各自的flip标志并重新`patterns.py`。四角单应变换已经负责CCD→模型方向，不能再额外翻一次CCD。

播放并采一张的命令示例（相位必须由你手动加载）：

```powershell
& $py run.py probe --bmp generated/cal/A_L.bmp
& $py run.py probe --bmp generated/cal/A_CHECK_32.bmp
& $py run.py probe --bmp generated/cal/A_WHITE.bmp
```

每次probe会更新同一个快速预览，仅正式会话的原始CCD按样本长期保存。连续采集前要退出这些独占设备的脚本。若SSH会话无法打开显示屏，在实验电脑已登录的桌面PowerShell运行同一命令。

## 3. 亮度检查

填写四角与传感器值域后，将geometry_confirmed改true。相位切回P_ZERO。

```powershell
& $py run.py exposure
```

32灰度×3帧。结果在`results/exposure/<时间>/response.png`。不应有大范围饱和，响应应适合振幅编码。如果出现U形/反向/漏光，不要沿用Meadowlark LUT；Holoeye的偏振与灰度响应需要单独标定。可配置256项`gray_lut_file`，但必须实测验证才用于正式采集。

## 4. 新建小样本六阶段会话

```powershell
& $py run.py init --session pilot01 --limit 4
```

4张测试图 + 完整100标题候选。标题也需3次Language采集，总计324帧；不是只采4×6。正式完整2400图另建会话`--session full01 --limit 0`，总14700帧。不要第一步就采全量，原始CCD会占用大量磁盘。

此DVP全幅一帧未压缩约20MB：全量仅原始帧可达294GB，另有播放BMP等；无损TIFF可减少占用但压缩比不保证。现D盘空间不足以按未压缩上界保存全量，请先小样本、确认ROI后适当设置硬件ROI或准备大容量存储。程序在剩余空间不足2GiB时会停止并保留已有帧。

## 5. 每层先生成输入、手动切相位、再采集

每次capture都会显示相位完整路径和SHA，只有你手动加载后输入y才采。后层依赖前层实测，禁止提前全生成或用仿真替代。

```powershell
& $py run.py prepare --session pilot01 --stage vision_router --device cpu
& $py run.py capture --session pilot01 --stage vision_router

& $py run.py prepare --session pilot01 --stage vision_expert --device cpu
& $py run.py capture --session pilot01 --stage vision_expert

& $py run.py prepare --session pilot01 --stage vision_global --device cpu
& $py run.py capture --session pilot01 --stage vision_global

& $py run.py prepare --session pilot01 --stage language_router --device cpu
& $py run.py capture --session pilot01 --stage language_router

& $py run.py prepare --session pilot01 --stage language_expert --device cpu
& $py run.py capture --session pilot01 --stage language_expert

& $py run.py prepare --session pilot01 --stage language_global --device cpu
& $py run.py capture --session pilot01 --stage language_global
```

六张相位在`generated/P/01_vision_router.bmp`至`06_language_global.bmp`。振幅每层在`sessions/pilot01/play/<stage>/00000.bmp`等。每个router真实CCD四区域积分→标准化能量→温度2 softmax→Top2→power-L2幅值权重，再拼专家输入。不是Top2强度直接相加。

重复同一capture命令会校验并跳过已完成帧，失败帧不会被标记完成。不要加clear-output、不要删除原始采集。

## 6. 电子处理和结果

```powershell
& $py run.py evaluate --session pilot01 --device cpu
```

结果`sessions/pilot01/results/metrics.json`含R@1/5/10、MRR、逐图预测；`embeddings.npz`可继续分析。任何一层真实CCD缺失都会停止，绝不回退仿真。

换曝光/ROI/方向/灰度LUT后重新生成BMP并新建会话，如pilot02。不要复用不一致的前层CCD。
