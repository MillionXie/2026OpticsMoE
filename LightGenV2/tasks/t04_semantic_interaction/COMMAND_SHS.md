# OpenMoji 单电脑六层光路

仅用于 routerfill_shared 原共享头 epoch40：仿真修改格准确率87.15%，整场景69%；不是旧98%版本。
模型、数据与词嵌入来自已核验独立仿真包，不需要联网下载Qwen、不执行语言Transformer。
这次只迁移硬件，不训练权重。原数据标签只供评价；source_grid参与最后保留区域合成，协议与仿真一致。

## 硬件与方向

振幅 Holoeye PLUTO 1920×1080，8μm；相位 Meadowlark HDMI 1920×1200，8μm；SHS-202-M相机。
两块SLM现在同接师弟电脑；相位必须是FNR0002、非主屏、1920×1200/60Hz且Windows屏幕顶边y=0。
相位通过SDK常驻线程、保留RGBA缓冲、消息泵与周期重写，不用GUI，不改VCom/固件/时序电压。
只使用19x12_8bit_linearVoltage LUT（不是已验证的线性相位LUT）。
沿用实测相位上下+左右翻转、255-g编码；导出执行一次，SDK不再重复变换。
模型478×478@17μm按物理宽度映射1016×1016@8μm；这不是原生8μm重新训练。
相位中心960,600；振幅中心960,540。ROI沿用20260914四点和水平镜像，若光路移动必须新标定、新session。

## 部署与命令

安装器 `install_shs.py --base ... --overlay ... --overlay-sha256 ... --output 新目录` 校验两包及文件SHA，拒绝覆盖旧工程。
`LAB.local.json`保存ROI、曝光与振幅配置；`PHASE.local.json`保存SDK/LUT路径、相位等待和显示设置。
SDK/相机驱动使用已安装环境及相邻ABO控制工程，不把厂商二进制提交Git。

```powershell
Set-Location E:\code\guest\2026OpticsMoE\OpenMoji_Lab_SHS_8um
$py='..\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe'
# 不接硬件：全1000test仿真复评、前4样本六个CCD边界回放、导出相位
& $py run.py export --device cuda
# 登录桌面终端执行；关闭三种GUI，不通过SSH Session0直接启动SDK
& $py run.py probe --session device_probe01
& $py run.py auto --session pilot01 --fields 4 --device cuda
# 少量六层实采通过后才运行1000条全部测试，6000次捕获
& $py run.py auto --session test1000_01 --fields 0 --device cuda
```

也可用Windows计划任务InteractiveToken在当前登录桌面启动pythonw.exe，同样run.py参数。
不要对SDK宿主设置SW_HIDE。pythonw仅去掉控制台，SLM显示窗口必须正常显示。
每次后台启动日志在 `logs/job_时间.log`。不需要本地电脑保持相位，也不依赖远程桌面在线。

顺序为 language_router → language_expert → language_global → vision_router → vision_expert → vision_global。
每个阶段自动prepare、保持相位、采集、审计，下一阶段输入由上游真实CCD生成；不以仿真代替实采。
每50张以及阶段前后做固定全白输入下平相位/目标/目标对照，检查变化和重复PCC。抽查不等于每帧绝对保证。
任何暗场/饱和/相位检查失败停止，不标为该层完成，不覆盖旧CCD；先检查证据再决定新session或恢复。
现有PNG配套SHA记录后断点续采；STOP文件放在对应session下，一张结束后停止。恢复前须人工审查失败批次。
运行中不打开相机/SLM GUI，也尽量不连接会改变显示拓扑的远程桌面软件。

## 看哪里

`sessions/会话/status.json`：当前阶段、数量、失败原因；`results.json`：六层全部完成后的实测指标。
`ccd/阶段/*.png`：478×478经固定四点透视后的8bit DN，不逐图拉伸、不log；不保存全传感器TIFF。
`play/阶段/`：原尺寸振幅BMP及输入/上游SHA；`phase/`：六张最终相位BMP。
`checks/`：少量相位变化/重复性证据；`theoretical_ccd/`：每层前16张npy与仅供显示的拉伸PNG。
`predictions/`：前16条源图、目标与预测，加指令。理论显示PNG不能作为网络输入。
原模型在电子处理中按 `I/mean(I)`统一强度尺度；弱信号与饱和的物理信息损失不会被这个操作修复。
阶段曝光改变必须新session并记录固定曝光补偿；首次正式曝光须先通过实际样本检查。

部署适配仅指硬件/文件路径。若以后微调，必须保留原best，另建训练run、明确训练/测试身份，不把测试适配成绩冒充独立测试。
