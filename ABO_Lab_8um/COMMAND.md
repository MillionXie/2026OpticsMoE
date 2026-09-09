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
& $py -m pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 --index-url https://download.pytorch.org/whl/cu126
& $py -m pip install -r requirements.txt
& $py verify_release.py
& $py test_lab.py
& $py simulate.py --device cuda --limit 4 --batch-size 1 --output cuda_smoke
```

本实验电脑GPU环境已安装并通过小样本验证，无需重新安装。小样本不是2400张80.583%的复现证据；服务器完整复核见证据目录和README。

### GPU与内存

本机GTX1060 3GB采用CUDA12.6版PyTorch。`--device cuda`明确要求GPU，不可用时直接报错；
省略参数则auto检测。旧相机RFL/Python3.6环境保持不动，不需要给相机环境安装torch。

```powershell
& $py -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

本机使用CUDA FP32（不是服务器BF16），batch=1。小于等于4GB的显卡自动把最大的冻结词表留在CPU，
仅查表后把选出的token送GPU；图像前端、光学计算和电子残差/读出仍在CUDA。数值公式和权重未改变。
词表保留原文件的存储精度，只把查出的行转FP32，不把整张词表展开成FP32，进一步减少系统内存；没有重新量化权重。
不加载已被光电网络替换的大模型Transformer权重。

“一层结束释放一层”针对**内存/显存**，不是删除硬盘权重：每条prepare命令是独立进程，退出即释放模型；
每个样本完成或到达待采CCD边界后清理临时张量和CUDA空闲缓存。残差和Language第一阶段状态在被后续使用前不能提前删。
权重、相位、原始CCD和重放必需的前层采集都保留。各层`play/<stage>/compute_memory.json`记录实际设备与显存峰值。

改配置并重新生成BMP以后，`verify_release.py`全量校验会报告generated文件与出厂版不同。
这时用`& $py verify_release.py --immutable-only`核验源码/权重/数据；该模式明确跳过配置生成物，
不会把它们称为已校验，也不会覆盖你的标定。正式采集仍检查会话配置身份和实际播放文件SHA。

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

**双SLM像素配准请用`generated/dual`内的配套图，不要把普通棋盘格与整屏P_GRAT拼在一起。**
每组固定`A.bmp`（振幅）配同目录`P.bmp`（相位，只有白格对应区域有0/π光栅）。
`P_V.bmp`是与P相反上下方向的相位备选；只能二选一，不要播放器额外翻转。

| 配套目录 | 图案 |
|---|---|
| `generated/dual/01_check64` | 旧版c64棋盘格；可见白格交替横纵光栅 |
| `generated/dual/02_blocks_x` | 非对称大块，块内X方向光栅 |
| `generated/dual/03_blocks_y` | 同样大块，块内Y方向光栅 |
| `generated/dual/04_check16` | 更小的c16棋盘格，作进一步位置核对 |

两屏都是8μm，k=1；图案共用8.126mm物理范围（约1016像素）。c64的设备格宽136像素；
c16为34像素；光栅周期17个设备像素，二值条纹宽度8/9交替。振幅纯0/255、相位纯0/128，
标定采用一致最近邻栅格，不影响正式网络BMP原有双线性振幅插值。

先固定一对，调平移；用非对称大块确认方向；比较中心与边缘是否同时重合。
若中心对齐而边缘持续偏离，检查倍率/旋转，不能靠平移消除。这里不自动扫倍率、不改模型ROI。
`preview.png`只是配对布局示意，**不是CCD衍射仿真**；实际应观察光栅衍射分布与亮区边界的对应变化，
不承诺相位条纹会直接按预览灰度成像。

```powershell
# 已生成；只有改了中心/flip后才需重新生成这组配准图（不改其他标定图）：
& $py dual_patterns.py
# 手动加载输出提示的相位P.bmp，程序仅持续播放配套振幅、CCD软件可正常使用：
& $py align.py --pair generated/dual/01_check64
# 上一条按Enter结束后再换下一组，切勿同时启动多个播放器：
& $py align.py --pair generated/dual/02_blocks_x
& $py align.py --pair generated/dual/03_blocks_y
# 若确认光路需要与当前相反的上下方向，改选提示的P_V.bmp：
& $py align.py --pair generated/dual/02_blocks_x --opposite-vertical
```

确定方向后将`LAB.local.json`的phase_slm.flip_vertical设成对应值（具体值见该组pair.json），
再重新生成和建立正式会话。程序不替你切相位，也不修改现有曝光/ROI/LUT。

文件已生成在 `D:\code\guest\2026OpticsMoE\ABO_Lab_8um\generated\cal`。
振幅为1920×1080 BMP，相位为1920×1200 BMP；两者按8μm像素设计。
478×17μm=8.126mm，对应1015.75（栅格约1016）个设备像素，不能把478直接作为显示宽度。

关闭其他占用振幅屏的播放器后，在**实验电脑桌面**运行；此命令不打开CCD，可同时使用CCD软件观察：

```powershell
& $py align.py --pair generated/dual/01_check64
# 上一个命令按Enter结束后，再换下一张：
& $py align.py --bmp generated/cal/A_DIGIT_3.bmp
# 手动加载相位P_F4.bmp，再持续播放全白振幅：
& $py align.py --bmp generated/cal/A_WHITE.bmp
```

`A_ACTIVE.bmp`只照亮1016×1016有效范围，`A_WHITE.bmp`是整屏白；菲涅尔标定使用整屏白。
新版四角菲涅尔已按用户提供的15/20/40cm BMP改为**四个方形透镜区域直接相接**，不再有176px小窗口之间的空白。
若只需生成10cm新版、不想动正式网络相位或其他标定，运行：

```powershell
& $py fresnel.py
```

手动加载`generated/cal/Phase_BMP/P_F4_10cm.bmp`（1920×1200、8μm、532nm、10cm），
振幅配该目录`A_WHITE.bmp`（1920×1080）。`P_F4_10cm_preview.png`带ROI/镜心标记，**只看图，不要加载到SLM**。
`P_F4_10cm.geometry.json`记录中心、编码、采样限制和原始参考图逐像素核验（参考图存在时）。
常规`patterns.py`生成的`generated/cal/P_F4.bmp`也使用同一个新版算法；该命令会生成其他标定和正式网络相位，和只生成菲涅尔的命令不同。

参考BMP的四镜心间距只有400px，不能直接作为当前1015.75px的ROI四角标定。
按当前ROI扩展后，完整阵列本应2031.5×2031.5，超过1920×1200屏，因此只在屏幕外缘截断；
镜心坐标和内部拼接边界不缩小。10cm时最密处条纹约1.64px/周期，低于2px采样条件，可能出现额外衍射；
本文件保证几何和公式一致，**不保证无杂散焦点**，也不能通过缩略图判断实际焦点数量。
新版按三张参考BMP的正二次灰度方向和`floor(255*mod(turns,1))`编码；这不代表已标定SLM实际灰度→相位正负号。

`A_DIGIT_0/1/2/3.bmp`来自旧实验的真实MNIST输入，只判断方向，不代表本项目执行了MNIST识别。
四角单独菲涅尔`P_F_TL/TR/BR/BL.bmp`可逐张确定焦点身份。四焦点共同出现时不能凭画面左右猜逻辑标签。
单角菲涅尔标签随配置的相位flip映射；修改flip后需重新生成标定图，再核对逻辑身份。
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
2. 像素配准用`generated/dual`每个子目录内的`A.bmp+P.bmp`。`P_GRAT_X/Y.bmp`整屏光栅仅保留为独立衍射方向诊断，不是配套棋盘格相位。
3. 振幅`A_WHITE.bmp`配新版`Phase_BMP/P_F4_10cm.bmp`或重新生成的`P_F4.bmp`，四焦点用于ROI四角。角坐标按逻辑TL/TR/BR/BL填，不按CCD画面的左右排序。`P_F1/P_F9`仍是旧的小窗诊断，此次不用于新版四角标定。
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
& $py run.py prepare --session pilot01 --stage vision_router --device cuda
& $py run.py capture --session pilot01 --stage vision_router

& $py run.py prepare --session pilot01 --stage vision_expert --device cuda
& $py run.py capture --session pilot01 --stage vision_expert

& $py run.py prepare --session pilot01 --stage vision_global --device cuda
& $py run.py capture --session pilot01 --stage vision_global

& $py run.py prepare --session pilot01 --stage language_router --device cuda
& $py run.py capture --session pilot01 --stage language_router

& $py run.py prepare --session pilot01 --stage language_expert --device cuda
& $py run.py capture --session pilot01 --stage language_expert

& $py run.py prepare --session pilot01 --stage language_global --device cuda
& $py run.py capture --session pilot01 --stage language_global
```

六张相位在`generated/P/01_vision_router.bmp`至`06_language_global.bmp`。振幅每层在`sessions/pilot01/play/<stage>/00000.bmp`等。每个router真实CCD四区域积分→标准化能量→温度2 softmax→Top2→power-L2幅值权重，再拼专家输入。不是Top2强度直接相加。

重复同一capture命令会校验并跳过已完成帧，失败帧不会被标记完成。不要加clear-output、不要删除原始采集。

### Router概率差不再中断采集

`ambiguous_top2_margin`已改为**仅告警**：仍按实测四区域积分得到Top-2，不替换专家、概率或权重。
不需要改配置、降低曝光门槛或新建会话。亮度不足、饱和、无有效能量等原有检查继续保留。
每张Router的`.quality.json`记录概率差与告警；正式`.record.json`记录所用策略。

若旧版因这个限制中断，但`.tif/.png/.json`已经保存，可以不打开任何硬件，直接核验并恢复：

```powershell
& $py recover_router.py --session pilot01 --stage language_router --accept-legacy-saved
& $py run.py capture --session pilot01 --stage language_router
```

第一条检查会话身份、原始TIFF与校正PNG是否一致及其他质量要求，写入有效采集记录，不修改原始像素；
第二条跳过所有已完成帧，只采剩余帧。不要重新init/prepare，更不要删除前面三层。
旧版失败帧没有采集时的相位/振幅哈希，所以`--accept-legacy-saved`代表实验人员明确确认它属于当前准备的阶段；
恢复记录会注明这个来源限制，不伪称存在旧的采集时哈希。新版已在质量检查前保存`.capture.json`，以后恢复不需该参数。

## 6. 电子处理和结果

```powershell
& $py run.py evaluate --session pilot01 --device cuda
```

结果`sessions/pilot01/results/metrics.json`含R@1/5/10、MRR、逐图预测；`embeddings.npz`可继续分析。任何一层真实CCD缺失都会停止，绝不回退仿真。

换曝光/ROI/方向/灰度LUT后重新生成BMP并新建会话，如pilot02。不要复用不一致的前层CCD。
