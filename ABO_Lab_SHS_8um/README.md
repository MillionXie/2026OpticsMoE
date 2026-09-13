# ABO + SHS 高速相机 / 8 μm SLM

## 日常只看一个文件夹

当前最新：**300μs+220ms未通过**。80次数字切换79次正确，`test65_d1`拍到上一张0
（对0 PCC=0.998968，对目标1=0.206798）。完整125帧保留，六层配置未改、全量未启动。
本轮记录`results/20260913_candidate300`，下方三组是此前对照，不是最新推荐。

本地：`C:\Users\Xml12\OneDrive\2026OpticsMoE\ABO_Lab_SHS_8um\reports\00_current`。
师弟电脑：`E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um\reports\00_current`。
打开`01_summary.html`看汇总和图片，`00_READ_ME.md`看文字；不需要逐个翻results。
每次完成工作必须更新此入口并告知绝对目录及打开命令；只发单张图片链接不算完成交接。

新增同批实验放在一个`results/<日期_任务>/`父目录，各参数组放子目录，父目录保存suite.json和日志。
旧结果因配置/脚本/文档有引用，未经依赖检查不移动；保留真实失败帧作为排障证据。
只删除可再生成且确认不再被使用的传输ZIP/构建包/无采集的临时BMP，并记录精确路径和释放大小。
模型、ROI、LUT、正式会话及原始CCD不在清理范围。目录分类与清理清单放固定入口，不再散建说明。

当前三组测试的单命令（在本地项目根目录运行；需要新输出目录）：

```powershell
python trial_suite.py --link results/smoke_configs/link_smoke_abo_newroi_20260913_161905.json --source-config results/smoke_configs/smoke_abo_newroi_20260913_161905.json --out results/20260913_timing_matrix
```

自定义候选：同一命令使用新`--out`，加`--trial 300 220 --switch-count 80`。
曝光单位μs、等待单位ms；最多80次换图以保持诊断总量≤128帧。汇总还包含周期分解和首个错帧对照。

顺序400μs/250ms、350μs/200ms、350μs/250ms；每组85帧，不并行争用设备。
各组.log在同一父目录，随时可以`Get-Content <日志路径> -Tail 20 -Wait`。图案测试失败仍记录并完成其余组，
硬件异常则停止，不伪造通过或偷偷修改参数。
若仅在两组之间更新汇总时中断，可在同一命令后加`--resume`：已完成组不重采，
仅继续尚无目录、尚无日志的未开始组。存在半采集目录时会拒绝自动覆盖，须先检查。
`--publish-only`只刷新固定入口，不操作设备。汇总JSON遇Windows短暂文件占用会有限重试。

最新三组已完成：400μs+250ms、350μs+200ms、350μs+250ms，均40/40图案对应正确，
最低PCC分别0.998797/0.995180/0.998034；灰度255最大饱和分别44.98%/20.56%/16.78%。
350μs+250ms仅为下一步网络输入复核候选，不是全量批准；高灰度饱和和跨轮亮度变化仍存在。
完整曲线和日志从固定入口查看，详见 [EXPOSURE_ANALYSIS.md](EXPOSURE_ANALYSIS.md)。

此前固定400μs+200ms均匀灰度扫描：
13档×3帧、原始ROI统计完成；灰度255饱和约46.46%，40次数字切换39次正确，
一次明确旧帧（错误参考PCC0.99939）。不能将该组合批准为全量可靠参数；14700帧尚未启动。
灰度响应曲线有重复异常，完整保留，不拟合新LUT。

六层曝光分析新入口见 [EXPOSURE_ANALYSIS.md](EXPOSURE_ANALYSIS.md)：原始ROI饱和/亮度统计、
固定输入两张快速相位检查、数字等待复测。2026-09-13恢复后已扫六层：450μs是光度共同候选，
部分细纹重复性未通过，暂不批准全量。200ms出现1次旧图；300/400ms各20次形状判别通过，
小样本MNIST已按400ms完成配对验证。完整限制与原始证据见该文档，不能把抽样检查当全量保证。

2026-09-13 新ROI下的 **ABO六层真实流程已完成**：4查询、100候选，324张网络实采，
R@1=3/4，R@5=4/4；不是完整数据集准确率，也不是四样本微调。
六层采前/采后96×96缩略指纹PCC均>0.9995（不是原始像素PCC）；相位LUT和精细配准仍未正式验收。
结果、CCD图和复现入口见 [ABO_SIX_SMOKE_20260913.md](ABO_SIX_SMOKE_20260913.md)。
MNIST旧mask重采样与原生8μm训练的对照见
[MNIST_NATIVE8_COMPARISON.md](MNIST_NATIVE8_COMPARISON.md)：60epoch已完成，完整test仿真
A=86.5528%、B=88.7659%；新鲜同40张实拍A=24/40、B=37/40（92.5%），
不是完整数据集实测准确率。两组重复PCC>0.995，但实拍与仿真光场仍有差距。

2026-09-13 下午相机移至10cm、用户新四角后，MNIST单层实测入口与完整证据见
[MNIST_VALIDATION.md](MNIST_VALIDATION.md)。新方向已用四标记和F验证：CCD左右镜像；
相位当前候选为上下+左右翻转后255−g。固定40张：仿真95%，实测60%，因此性能
**该旧mask性能尚未验收通过**；原生8μm新方案的配对结果见上文。MNIST单层结果与上面的ABO六层流程检查是两项不同实验；旧配置和会话未覆盖。

2026-09-13 更新：相位 owner 默认使用厂商示例的持久 RGBA 缓冲、同线程消息泵及每秒重发当前图；
灰度复制到RGB三通道、alpha=255，仅改变传输打包，不改原BMP。早先Mono8虽短测可用，
后续出现不响应；同连接对照中RGBA恢复透镜/专家响应，不能继续把Mono8当成可靠路径。
不进行红绿通道切换或黑图—透镜交替预热。`phase_display_align_top=true`
仅在 owner 存活期间将识别到的非主屏 FNR0002 顶部设为 Y=0，退出恢复原屏幕位置，
不改分辨率/刷新率/主屏/LUT/电压，不写注册表。该修正的依据是显示原点对照实测，
不是仅凭 SDK 返回值。需本地 `pywin32`，师弟电脑不负责相位显示。

六层少量实测排障入口为 `smoke_six.py`，操作和边界见 [SMOKE_SIX.md](SMOKE_SIX.md)。
其配置、相位 BMP、会话与正式流程隔离；保留全部100个候选标题；结果明确标为
`diagnostic_real_six_stage`，不等于正式 ROI/LUT 验收或完整数据集准确率。

自动六层流程先读 [START_HERE.md](START_HERE.md)：本地SDK换相位，师弟电脑振幅/SHS/GPU；相机参考验证、有限重试和失败批次隔离。首次仍需人工确认方向、四角ROI和六层参考图，不能把代码就绪当成实测验收通过。
旧单设备/手动流程见 [COMMAND.md](COMMAND.md)。本工程保留 ABO 六阶段的模型/几何约定，**不改变光路**。
最新联合测试与 RTX4060 推理证据见 [JOINT_RESULTS.md](JOINT_RESULTS.md)。
给老师的周期分解、数字切换复测、相机独立吞吐和两种SLM理论边界见 [TIMING_REPORT.md](TIMING_REPORT.md)。不要混用100 fps、200 ms等待、Visible和完整任务推理时间。
本地HDMI相位与远端振幅/相机的正式单命令入口见 [DUAL_CONTROL.md](DUAL_CONTROL.md)，
尚需正式方向/相位LUT验收；上面的smoke小样本通过不能替代正式参考库验收。

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
- 联合控制已验证：原问题是 sleep 后固定丢6帧仍取得旧光场，不是光路无响应。改为等待期间持续取帧丢弃后，棋盘格及反相图能正确切换。150 μs、200 ms等待、100 fps连续流下，20次左右交替全部正确，最低同输入 PCC=0.997965、无饱和。9月13日已补新CCD四角诊断和ABO六阶段小样本实测，见文首；不等于正式全量验收。

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
6. 当前确认 `SyncMode=InternalSync`，使用连续流。Visible 后在整个等待窗口持续取帧归还缓存，再额外丢弃 buffer_count+2 帧取下一帧。**不能退回 sleep 后只丢6帧：实测会错帧。**9月13日复测200ms出现旧图，不再作为当前可靠推荐；300/400ms小样本通过，本次MNIST采用400ms。没有硬触发保证，相机GUI、曝光/帧率或程序负载改变后复测。

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
