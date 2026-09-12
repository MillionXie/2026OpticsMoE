# ABO + SHS 高速相机 / 8 μm SLM

自动六层流程先读 [START_HERE.md](START_HERE.md)：本地SDK换相位，师弟电脑振幅/SHS/GPU；相机参考验证、有限重试和失败批次隔离。首次仍需人工确认方向、四角ROI和六层参考图，不能把代码就绪当成实测验收通过。
旧单设备/手动流程见 [COMMAND.md](COMMAND.md)。本工程保留 ABO 六阶段的模型/几何约定，**不改变光路**。
最新联合测试与 RTX4060 推理证据见 [JOINT_RESULTS.md](JOINT_RESULTS.md)。
给老师的周期分解、数字切换复测、相机独立吞吐和两种SLM理论边界见 [TIMING_REPORT.md](TIMING_REPORT.md)。不要混用100 fps、200 ms等待、Visible和完整任务推理时间。
本地HDMI相位与远端振幅/相机的单命令入口见 [DUAL_CONTROL.md](DUAL_CONTROL.md)，尚需完成联合方向/相位响应验收，不能当成已验证的六层实测结果。

实測证据和吞吐边界见 [BRINGUP_RESULTS.md](BRINGUP_RESULTS.md)；2250 fps 的 Python 直取短测有跳帧，不宣称已实现满速无丢帧。

白天复测发现首次开流短时间内可能返回“完整但全零”的图。新增
`camera.startup_warmup_s=2.0`：每次打开相机后，第一次开流持续取帧并丢弃 2 秒，
再进入正常采集；同一连接后续换图不重复这 2 秒。它不根据亮度自动重试，
不改变曝光，也不等于每张 SLM 的 settle delay。诊断记录保留在
`results/daylight_20260912_084824`，初始全零数据不能当作曝光/增益标定结果。

## 当前已经验证与尚未验证

2026-09-12 在师弟电脑实测：SHS-202-M，Magewell Flex I/O Quad CXP-12 Enhanced，Windows x64，CEasyCapS + GenTL 1.1.4.22。

- 不启动 FastStream/Viewer、不插 GUI 加密狗，SDK 已成功连接和采集。
- 1920×1080 Mono8；曝光 100/500/1000/3500 μs，每档 3 帧，设置回读一致，完整帧检查通过，PNG 保存逐像素一致。
- 数字 Black/White/VStrip 各 2 帧，分别为全 0、全 255、0～224 条纹；结束恢复 Normal。
- 真实场景为夜间未开灯房间，均值约 3.4～3.65/255。**尚不能据此认证曝光线性、光学信噪比或像素响应均匀性。**
- 联合控制已验证：原问题是 sleep 后固定丢6帧仍取得旧光场，不是光路无响应。改为等待期间持续取帧丢弃后，棋盘格及反相图能正确切换。150 μs、200 ms等待、100 fps连续流下，20次左右交替全部正确，最低同输入 PCC=0.997965、无饱和。尚未完成新CCD四角标定和 ABO 六阶段实测评估。

## 为什么选择这套 SDK

|组件|用途|本工程处理|
|---|---|---|
|SHS FastStream|相机厂商 GUI /录像/参数工具|不依赖 GUI，不绕过加密狗|
|Flex Io Viewer|采集卡显示与诊断工具|与 SDK 互斥，运行代码前关闭|
|cxplink_gentl.cti|底层 GenTL producer / CXP 传输|使用已经安装的匹配版本|
|CEasyCapS.dll|厂商推荐的较简单 C API|直接取帧；显式 ctypes x64 ABI|

依据用户提供的 SDK 手册“SDK 层次关系”、已安装 CEasyCapS.h、GenTL.h、PFNC.h 和源代码。不是普通 HDMI/SDI 的 MWCapture SDK；不要混用 FastStream 目录的旧 DLL。

SDK 根目录在 `config.json` 的 `camera.sdk_root`。必须同时保留 `demo/base_dll/bin/CEasyCapS.dll` 与 `cti/x86_64/cxplink_gentl.cti` 的配套安装。原厂目录不移动、不覆盖、不自动加载 demo 中针对另一款 Adimec 相机的配置。

厂商 PDF 已整理到本机 `vendor/manuals`，不提交 GitHub；四份分别解释相机硬件、FastStream、FlexIO 快速入门、SDK。所有代码在 Git；生成图/实测/SDK 在各自的忽略目录。

## 必须区分的几个参数

1. `camera.exposure_us` 是**相机 ExposureTime**，单位 μs，3500 μs=3.5 ms。不是采集卡 DeviceExposureTime。
2. `frame_rate_hz` 决定帧周期，也限制曝光上限。实测 2250 fps 上限 444.2 μs；100 fps 上限 9999.8 μs。想用 3500 μs，先降帧率，不能仍设 2250 fps。
3. `gain` 支持 `Gain_X1/X2/X4/X8`；null 保持当前增益，不能假定默认 1。首次连接发现是 X4。
4. `settle_delay_ms` 是 Holoeye 确认图像 Visible 后额外等待。相机快不代表 60 Hz Holoeye 能更快切换；不能把相机 2250 fps 当作整体吞吐量。
5. `buffer_count=4` 是可重复使用的 SDK 缓冲区数量，不是要保存 4 张。取到一帧即复制有效像素，finally 归还缓冲区。
6. 当前确认 `SyncMode=InternalSync`，使用连续流。Visible 后在整个等待窗口持续取帧归还缓存，再额外丢弃 buffer_count+2 帧取下一帧。**不能退回 sleep 后只丢6帧：实测会错帧。**目前推荐200 ms是此光路短测起点，不是硬触发保证；相机GUI、帧率或程序负载改变后复测。

## 相位 LUT 方向与文件约定

**新接入本地HDMI相位的验收注意：**下面反灰度是旧实验约定，不是本次已经证明的标定结论。用户指定 `19x12_8bit_linearVoltage`，它不保证线性相位；相位上下翻转、反灰度是否正确仍在联合诊断中，见DUAL_CONTROL.md。不得用“SDK加载成功”替代光学响应验收。

本实验室 `phase_slm.gray_encoding=inverted_255_minus_g`，最终 BMP 每个像素 `g_out=255-g`。先做原有相位量化和空间方向处理，最后才反灰度。零相位背景也为 255；这不等于遮光，也不是把训练相位乘 -1。

师姐正常 LUT 选择 `normal`。垂直/水平翻转独立配置，不因本次 LUT 反向自动改变。振幅、CCD 均不反灰度。旧工程原始文件和会话不覆盖；新文件统一放 `generated/phase_inverted`。切换 LUT 配置后使用**新 session**。

保持旧 ABO 模型 478×478、17 μm 的物理宽度 8.126 mm，映射到 8 μm 面板是 1015.75 px（栅格支持约 1016 px），不是把物理 ROI 缩为 478 个 8 μm 像素。振幅 1920×1080，相位 1920×1200。四菲涅尔子区相连，中心在物理 ROI 四角；10 cm 外圈可能欠采样，正确几何不等于没有衍射混叠。

## ABO 兼容范围

`run.py` 复用随包 `compat_abo` 的六阶段准备/评估逻辑，仅替换硬件入口、phase 文件路径与 raw PNG 存储。六个阶段依然是 vision_router/expert/global、language_router/expert/global。保留固定 80.583% 历史参考 checkpoint 的 SHA，不承诺新光路得到相同准确率。

源码小包与大资源包分开。师弟电脑现已导入原始 checkpoint、2400 查询数据和固定前端参数，
记录在 `INFERENCE_ASSET_MANIFEST.json`；参数原字节复制，不包含未使用的 Transformer 层。
新 `.venv_gpu` 已通过 RTX4060 图像/文本仿真特征前向与四张 router BMP 准备测试。
这不是完整测试集准确率，也不是硬件六阶段验证。相机独立 `.venv` 不需要 torch。

正式默认仅保存 `ccd/*.png`：单次 homography 到 478×478，固定 [0,255]，无逐图归一化/log/gamma。
只有 `save_raw_frames=true` 才额外保存全幅 `*.raw.png`。兼容记录中的 raw 文件引用在最简模式
指向同一张 canonical PNG，不意味着保留了原图；元数据 `raw_frame_saved=false` 明确区分。
实际相机四角必须重新标定，不能复用旧 DVP ROI。四点标签代表逻辑光场方向。

## 文件管理与后续 AI

- 本机配置只改忽略入 Git 的 `LAB.local.json`；通用模板是 `config.json`。实测结果 `results/`；ABO 会话 `sessions/`；生成 BMP `generated/`；原厂资料 `vendor/`；发布包 `releases/`。
- 提交代码并推 GitHub，再以带 commit + 每文件 SHA256 的包部署，不能只改远端一份代码。
- 不自动启动/关闭别人的 Viewer；不要修改未连接的 SLM。原 `run.py stage`仍由人工加载相位；新增 `dual_run.py`仅在联合标定验收后自动控制用户指定的本地HDMI相位SLM。
- 新相机标定/方向/LUT/曝光策略变动后新建 session，不重写旧 record 或图片。
- LSP 等新任务另建工程，复用相机接口；不要在本工程强塞另一套模型。
