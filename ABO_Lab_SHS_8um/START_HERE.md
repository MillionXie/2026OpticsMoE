# 从这里开始：本地相位 + 师弟电脑振幅/SHS/GPU

## 先说当前能做什么

代码已提供完整的六层自动调度、相机参考图验证、有限重试、失败批次隔离和断点续跑。
2026-09-13新ROI下，4查询、100候选的六层小样本实测已完成（324张网络采集）；
这不等于全量验收。新ROI和方向的诊断证据见`README.md`，相位LUT与精细配准仍未正式验收。
最新曝光/时序结果统一看`reports/00_current/01_summary.html`，不要用旧排障目录判断当前进度。
200ms等待出现过旧图；重试是容错，不代表底层问题已经消失。不能把验证开关直接改成true来绕过标定。

两台电脑分工：

|电脑|工程|职责|
|---|---|---|
|本地，相位SLM连接在这里|`C:\Users\Xml12\OneDrive\2026OpticsMoE\ABO_Lab_SHS_8um`|保持相位SDK、换层、判定参考图、通过SSH调度|
|师弟电脑|`E:\code\guest\2026OpticsMoE\ABO_Lab_SHS_8um`|RTX4060生成本层振幅、Holoeye播放、SHS采集、GPU电子计算及最后评估|

师弟电脑运行的是已验证的 `.venv_gpu` CUDA 环境；本地不加载Qwen/torch。
两台电脑都要保持登录、不休眠。关闭Blink GUI、Holoeye播放器、FastStream/Viewer，避免抢占。
**只有一个协调程序可以运行；不要同时启动手动采集或另一个相位SDK。**

## 你需要做的事，只有四项

1. 对齐振幅与相位，确认相位空间翻转及灰度编码。
2. 在新SHS原图中标出四个逻辑ROI角点，填入配置。
3. 在光路不变、曝光固定时采一套六层参考图，检查后只批准一次。
4. 启动小样本；通过后再启用全量。之后不需要逐层确认。

## 1. 本地准备命令

以下除特别注明外，都在**本地电脑** PowerShell 执行：

```powershell
Set-Location C:\Users\Xml12\OneDrive\2026OpticsMoE\ABO_Lab_SHS_8um
$py = 'C:\ProgramData\anaconda3\python.exe'
if (-not (Test-Path .\dual.local.json)) { Copy-Item .\dual.example.json .\dual.local.json }
```

`dual.local.json` 中的连接参数为师弟电脑 `PS@1.tcp.vip.cpolar.top:12705`。
密码由脚本交互询问，不写进JSON。若多次询问不方便，可在当前终端设置
`SHS_SSH_PASSWORD` 环境变量；不要把密码写进共享md或提交Git。
首次SSH主机密钥必须事先核对、接受；脚本拒绝未知密钥。

本地只编辑 `dual.local.json` 控制连接/重试。硬件参数以**师弟电脑的 `LAB.local.json`**为准；
不要把本地旧的 `LAB.local.json` 直接覆盖过去。用下面命令下载当前配置来编辑：

```powershell
& $py sync_lab_config.py pull --file LAB.review.json
```

已有同名文件时会拒绝覆盖，避免丢失编辑。需要重新下载就用 `LAB.review2.json` 等新名字。
`.source.json` 是并发编辑检查，不要删除或手改。

## 2. 生成并拍摄标定图

先让脚本拍一组全幅原图：

```powershell
& $py guarded_workflow.py alignment --out results\alignment01
```

它在师弟电脑上生成最新标定BMP和六层BMP，再由本地SDK加载相位并拍摄。
结果在本地 `results\alignment01`，有原始1920×1080 PNG、回执、`review.png`。
**这是未验收的标定照片，不会自动设置任何方向/ROI通过标记。**
SDK回退时这些照片也可能错误；应能明显区分四点、单点、L形、单角，否则先不要填坐标。

标定BMP在师弟电脑：

```text
generated\phase_inverted\cal\A_WHITE.bmp
generated\phase_inverted\cal\A_L.bmp
generated\phase_inverted\cal\A_TL.bmp / A_TR.bmp / A_BR.bmp / A_BL.bmp
generated\phase_inverted\cal\P_ZERO.bmp
generated\phase_inverted\cal\P_F4.bmp
generated\phase_inverted\cal\P_F_TL.bmp / P_F_TR.bmp / P_F_BR.bmp / P_F_BL.bmp
generated\phase_inverted\dual\01_check64\A.bmp 和 P.bmp
```

`P_F4`是10 cm、四个相连子区的菲涅尔阵列，不是四个隔开的孤立小透镜。
单角菲涅尔帮助给四个光点命名。配套棋盘格/局部光栅用于调整位移台像素对齐，
不能用一个不配套的整幅光栅替代。需要手动调位移台时可使用原有 `calibrate.py` / `slm_camera.py` 流程；
不要与自动协调器同时操作同一SLM。

本实验室当前约定最终相位编码为 `255-g`，文件已做反灰度，SDK不再反第二次。
上下/左右翻转由 `phase_slm.flip_vertical` / `flip_horizontal` 独立控制。
**不能仅根据CCD左右镜像就认定相位也要镜像；必须用配套局部光栅/非对称标记确认。**
`19x12_8bit_linearVoltage.lut`是线性电压表，不是已验证的线性0～2π相位表。
因此此处自动流程不替代相位LUT定量标定，也不保证自动复现原仿真精度。

## 3. 四个ROI角怎么填写

编辑本地下载的 `LAB.review.json`：

```json
"logical_corners_full_sensor_xy": {
  "top_left": [这里填x, 这里填y],
  "top_right": [这里填x, 这里填y],
  "bottom_right": [这里填x, 这里填y],
  "bottom_left": [这里填x, 这里填y]
}
```

上面是解释占位，**必须换成真实数字才是合法JSON**。
用原始1920×1080 SHS照片，x向右、y向下；不能用GUI缩放截图坐标、旧DVP坐标或478×478输出坐标。
标签按输入光场的逻辑角命名，而不是按相机画面的左上右下排序。
例如输入TL如果成像在相机右上，那么这个右上光点仍填写到 `top_left`。
先用单角振幅/L形确认逻辑方向，再通过四点菲涅尔确定对应的有效范围边界。
光点被裁切/不清楚时不要猜坐标；先调整成像范围/光路。

填好后先预览（替换成实际原图文件名）：

```powershell
& $py roi_preview.py --config LAB.review.json --image results\alignment01\实际原图名.png --out results\roi_overlay01.png
```

这会画出四角标签和连线，**不改像素、不自动确认配置**。
确认四角顺序正确、范围覆盖目标后，才在 `LAB.review.json` 设置：

```json
"geometry_confirmed": true,
"capture_input_range": [0, 255]
```

同时确认下面两项（在各自已有对象里面添加/修改，不是另建重复对象）：

```json
"camera": {
  "exposure_us": 150.0,
  "gain": "Gain_X4",
  "frame_rate_hz": 100
}
```

保留camera对象其余SDK字段，不要把整个对象缩减为这三项。
150 μs / Gain_X4 是当前诊断起点，不是任何图案都合适。改变光强后要检查饱和。
正式自动采集要求增益明确，不能用 `null` 继承GUI残留设置。
相位对象增加当前LUT的SHA：

```json
"lut_sha256": "6a968b970fc81b035788534fc6ab7236a59e25e1a73068de393d90720a52efb6"
```

确认后上传，脚本会备份旧配置并检查没有被别人同时修改：

```powershell
& $py sync_lab_config.py push --file LAB.review.json
```

最后在本地 `dual.local.json` 设置 `orientation_and_phase_response_verified=true`，
仅表示你已完成这次方向/编码检查，不代替下面每次换图的实时验证。
改ROI、LUT、曝光、增益、中心、翻转以后，必须重新采参考图并使用新session。

## 4. 一次性建立六层相机参考图

```powershell
& $py guarded_workflow.py enroll --out results\phase_bank01
```

共8种相位：六层实际训练相位、均匀相位、诊断透镜；使用固定棋盘格振幅，
做3轮不同顺序的独立采集，共24张诊断图。保存全幅原图和 `review.png`，无逐图增强。
每张与“其余轮次”的参考比较，不把它自己作为参考；不同相位分不清或初次换图失败会拒绝建库。
**不能通过删掉坏图、降低阈值到旧图也能通过来完成建库。**

通过后仍需你或协助标定的同学看一次 `review.png` / 原图，确认各相位对应正确、
不是旧画面、没有过曝，方向与前一步一致。稳定重复不等于物理相位一定正确。
确认后只批准一次：

```powershell
& $py guarded_workflow.py approve --bank results\phase_bank01\bank.json --confirm-reference-images
```

没有通过重复性/区分性检查的库不能批准。若同层不稳定，先恢复SDK/稳定光路后新建目录重采；
若不同层本身光场极相似，当前相机指纹不足以辨识相位，需要设计更有区分度的探测输入或可靠的控制器回读，
本版会停，不会假装已经加载正确。

## 5. 一条命令跑六层 + 最后评估

先4个图像查询；100个候选标题仍要走语言三层：

```powershell
& $py dual_run.py --session shs_auto01 --limit 4 --bank results\phase_bank01\bank.json
```

顺序为 `vision_router → vision_expert → vision_global → language_router → language_expert → language_global → evaluate`。
每层准备进程结束后才打开振幅/相机，避免模型和采集进程同时挤占GPU。
阶段之间相位SDK保持同一个连接；不需要你逐层换相位或输入y。
先小样本核对结果和光场。全量2400图像查询使用新session：

```powershell
& $py dual_run.py --session shs_full01 --limit 0 --bank results\phase_bank01\bank.json
```

4查询正式样本曝光总计324次，全量正式样本14700次，**不含额外相位验证探测帧**。
验证会增加时间；这些时间不得计成纯光传播/9ms理论时间。

## 6. 验证/重试具体做什么

1. 用固定振幅先拍一个不同的诊断相位，确认不是一直停在目标旧图。
2. 下发目标相位，拍固定振幅，与该层参考比PCC、亮度比例、饱和度，同时要求比其他层更相似。
3. 失败做有限的通道往返/预热后再试；默认最多3次，耗尽则报错退出，不无限循环。
4. 通过后采一小批正式输入。批后不重新写相位，直接再拍探测图，检查相位是否仍对应目标。
5. 批后失败：该批次文件移动到独立quarantine目录保留，再重试该批。不会使用这些图片生成下一层输入。

`dual.local.json` 关键参数：

|参数|默认|含义|
|---|---:|---|
|`phase_startup_cycles`|30|SDK启动时黑/透镜交替写30次并切回红通道，约1分钟，一次连接做一次；不是已证明的底层修复|
|`phase_settle_s`|1|每次相位写入后的等待；另有通道往返等待|
|`phase_retry_cycles`|4|验证失败后的有限恢复换图次数|
|`phase_max_attempts`|3|相位验证/批次重采各自的最大次数|
|`capture_batch_size`|32|一次复查间隔的正式样本数；不是模型训练batch|
|`phase_thresholds.minimum_pcc`|0.97|与目标参考的最低PCC|
|`phase_thresholds.minimum_margin`|0.01|目标相似度至少领先其他相位的幅度|

**批前后验证不是每一瞬间的硬件反馈。**若中途短暂错误后又恢复，仍可能漏检。
最保守可把 `capture_batch_size` 设为1（明显更慢）；即使如此也不是硬触发逐帧相位认证。
要严格保证每帧对应，需要最终修复SDK/HDMI同步问题或增加可靠的硬件反馈。
此限制不能在论文或实验记录中省略。

## 7. 中断、续跑和文件去哪里

同配置、同参考库、同session重跑原命令即可。已通过验证的批次不会重采。
远端session还会绑定参考库ID；旧手动/无验证session中的CCD不会被当成已验证数据直接复用，第一次请用新名字。
中断时尚未获得批后验证的批次会隔离，再采；不要手动删除journal/phase_batches。
若某阶段准备很慢，等该阶段结束即可；电子计算和GPU都在师弟电脑。

```text
本地 results/guarded_runs/<session>/
  journal.json             每个批次是否通过；中断恢复依赖它
  *.verification.json      每次换图/批后验证，包含失败原因和各候选PCC
  *.png                    少量全幅诊断图，非每个正式样本的raw副本
  metrics.json             最终实测结果（全部六层完成后才出现）

师弟电脑 sessions/<session>/
  play/<stage>/            GPU生成的输入BMP与清单
  ccd/<sample>/            正式478×478 PNG与必要记录，不存TIFF
  phase_batches/           不可混用的批次身份、待验证/通过/隔离状态
  quarantine/<batch_id>/   失败批次，保留可审计，不自动删除
  results/metrics.json     最终结果
```

正式PNG只做ROI几何变换、固定Mono8范围保存，没有逐图min/max、log、gamma或CLAHE。
参考指纹的缩小/PCC只用于验图，不注入模型，不替换真实CCD，不按准确率挑样本。
如果重试仍失败，直接保留日志停下，不要关闭安全检查或继续后续层。

## 8. 版本与验收边界

源码通过Git提交并推送；实验室部署包由 `build_lab_package.py --code-only` 从提交生成，
附 `CODE_MANIFEST.json` 和ZIP SHA256，不复制工作区未提交源码，不覆盖LAB.local.json、模型、旧CCD。
小型软件回归测试覆盖旧图拒绝、参考库歧义、重试上限和隔离；
**这些软件测试及诊断相位测试不等于完成六层实测验收**。最终仍以新ROI标定后的真实小样本为准。

2026-09-13构建记录：30项软件测试通过（含六阶段调度/失败批次重采的纯软件夹具测试、旧未验证session拒绝复用）。
新验图代码实机诊断记录在本地 `results/guard_code_live_20260913_042029`：
目标为透镜时，相机图与黑相位参考PCC约0.9993、与透镜约0.604，程序两次判定旧图并重试，
没有放行正式采集。随后SSH连接中断，测试退出；这验证了旧图拒绝路径，**不是成功完成六层的证据**。
# 新相机位置的最新实测（2026-09-13 下午）

先看 [MNIST_VALIDATION.md](MNIST_VALIDATION.md)：用户新ROI已用独立标记验证相机左右镜像，
MNIST固定40张仿真95%、实际60%。相位候选已生成，但精细配准和光学响应仍需改进。
下面的ABO六层流程说明仍不等于已通过新光路的六层性能验收。
