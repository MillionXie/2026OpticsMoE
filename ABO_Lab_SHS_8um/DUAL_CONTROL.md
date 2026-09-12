# 本地相位 + 师弟电脑振幅/相机/GPU

## 设计与当前状态

**2026-09-13暂停相位SDK测试：**只读二进制审计发现，厂商 `HdmiDisplay.dll` 的1920×1200初始化路径会调用颜色通道与VCom设置实现。我们没有显式调用 `Set_SLMVCom`，但不能因此声称 `Create_SDK` 不会重新下发控制器状态。此前“不改VCom”只能解释为“包装代码不显式修改”，不是厂商初始化无副作用的保证。当前GUI与SDK图案均未测出明确相位响应，原因未定位；未经原工作状态核对，不重启SDK或试改电压。用户要求先软件诊断，不自动重置控制器。

审计对象 SHA256 `8781f730cc3113064a6f5a6dd16e0bfe0b872893cd62c011a1079c72ae2cd532`：构造函数 RVA `0x1e461` 调用初始化 `0x137d0`；后者 `0x13837` 调用 `0x15b70`（与导出的Set_channel相同实现），`0x13846` 调用 `0x15280`（与导出的SetVcom相同实现）。这证明有设置路径，不证明此前具体值改变，也不证明失效因果。当前磁盘Preferences中VCom=2.5，历史资料副本也是2.5，但没有初始化前后的控制器读数，不能据此认定运行时一直不变。

一条命令在**连接相位SLM的本地电脑**启动：本地保持HDMI SDK连接，通过SSH调度师弟电脑登录桌面上的GPU准备和相机采集。无需另开公网相位控制端口；SSH密码从提示或环境变量读取，不写进配置。

流程：检查新硬件配置 → 初始化会话 → GPU准备本层振幅 → 下载本层最终相位BMP并核对SHA → 本地SDK加载相位、等1秒 → 师弟电脑采完整层 → 下一层 → 六层结束后GPU评估、将metrics下载到本地。

当前是**待联合方向/相位响应验收的入口**，不是已经完成新光路准确率复现。必须保留 `orientation_and_phase_response_verified=false`，直到方向和响应真正验证；不得为了跳过检查而直接改成true。新SHS四角仍不能沿用DVP坐标。

2026-09-12首轮联合诊断：本地检测到1920×1200、8-bit，SLMFound/COMFound均为1，LUT加载返回成功；振幅向右192px导致CCD质心向左157.675px，振幅向下192px导致CCD向下153.620px（有少量倾斜/尺度差），支持“CCD相对振幅左右镜像、上下同向”。这还不是四角高精度标定。中心正负横纵光栅没有观察到明确位移，**相位实际换图/空间方向/灰度响应均未验收**；不要自动写入flip_vertical、gray_encoding或geometry_confirmed。

该版Blink_C_wrapper.dll（SHA `0d3cc283165bb62ed60a4c8b1c1a256af9441e6342654511fd1f80dbe46ce225`）的Write_image函数固定返回0：函数入口将返回局部字节置0，调用void LoadImg后未经赋值便返回它。因此这一个已核对版本的0既不能作为成功回执，也不能单独据此认定失败。驱动保留原返回值，并仅对该SHA标记特殊语义；未知版本仍检查失败返回。不能将“SDK调用结束”当“光场已经验证”。

原始记录：本地 `results/phase_joint_20260912_05`，远端 `results/phase_joint_20260912_233021`。最后一次远端采集实际退出后，写状态文件遇到Windows共享锁，旧状态仍显示running；已保存原stale状态并发布真实失败记录使本地安全退出。后续代码改为一次性result.json终态回执及文件替换重试。没有删除CCD。

## 依赖与命令

### 手动GUI对照补测（2026-09-12）

用户在Blink GUI分别保持同一32px周期横向光栅和纯黑，两次均不启动本地相位SDK；远端显示同一中心振幅块，150 μs、Gain_X4、100 fps、200 ms等待。光斑质心差0.0440 px，固定ROI `[760,470,1000,710]` 的原始强度PCC为0.997624，没有看到明确偏转。此结果不能单独归因于SDK，也不能据此认定相位SLM损坏或完全无相位调制；下一步核对实际显示输出、光束覆盖、偏振和相位LUT。保持所有方向/相位验收标记为false。

证据见 `reports/phase_manual_20260912/comparison.json` 与同目录 `comparison.png`（共同0～255显示范围，不做逐图拉伸）。原图保留在本地与师弟电脑 `results/manual_phase_gx_20260912_01`、`results/manual_phase_flat_20260912_01`。两次桌面作业均正常完成并释放相机，独立result.json终态回执已实测通过。

更强图案的补测入口：`python generate_phase_response_patterns.py --out generated/phase_response_<唯一编号>`。输出1920×1200原生8 μm面板的4/8px周期光栅及反向码、标称532 nm/10 cm全孔径透镜及反向码、纯黑。当前本地输出为 `generated/phase_response_strong_20260913`，先测试 `P_gx_p8.bmp`。这些是定性诊断码，不是已标定相位：linearVoltage非线性、透镜外区欠采样均需注意。GUI原尺寸加载，不自动翻转，也不要对inverse文件再次反灰度；不要替换正式训练mask。

本地Python需要numpy、Pillow、paramiko；测试环境 `C:\ProgramData\anaconda3\python.exe`。相位使用原厂Blink 1920 HDMI SDK，不是高速PCIe SDK，不改VCom或pre/post ramp。

```powershell
Set-Location C:\Users\Xml12\OneDrive\2026OpticsMoE\ABO_Lab_SHS_8um
$py='C:\ProgramData\anaconda3\python.exe'
# 首次导入厂商SDK（本地已导入，不用重复）：
& $py import_phase_sdk.py --source 'C:\Users\Xml12\OneDrive\2026OpticsModel\实验设备\00-SLM-Meadowlark Optics\Blink 1920 HDMI'
```

SDK依赖整理至忽略Git的 `vendor/phase_hdmi`；其中ImageGen还依赖厂商python38.dll，不能漏掉。原安装目录不移动。LUT为用户指定的 `19x12_8bit_linearVoltage.lut`，SHA256：`6a968b970fc81b035788534fc6ab7236a59e25e1a73068de393d90720a52efb6`。

标定验收后再使用：

```powershell
# 首次复制模板；不要覆盖已有本机配置。
Copy-Item dual.example.json dual.local.json
# 填入已确认的主机路径/标定状态；将同一LUT SHA写入远端phase_slm.lut_sha256。
# SSH主机密钥需要先通过正常ssh连接核对并接受，脚本拒绝未知主机密钥。
& $py dual_run.py --session shs_auto01 --limit 4
```

`--limit 4`仍需100个候选标题，并非六层各只有4张；初始化后不随意修改limit/设备配置。同名会话支持跳过已完成的合法记录，不删除旧CCD。`--limit 0`表示2400个查询，第一次不要直接全量跑。

本地和远端桌面须保持登录，不休眠；关闭Blink、Holoeye和相机GUI，避免其他软件覆盖显示。自动换层会等待1秒，这只发生约6次，不是每张输入加1秒。最终BMP已含导出时的翻转/反灰度，SDK**不再重复翻转或反灰度**。

任务日志在师弟电脑 `results/dual_jobs`；`status.json`记录启动，`result.json`单独记录完成/失败，避免替换正在被读取的状态文件。本地相位回执/下载结果在 `results/dual_runs/<session>`。每次capture前核对相位回执SHA与prepare清单。当前SDK回执不是光学传感器回读，其他GUI仍可能破坏显示，必须保持独占。

连接断开时，远端作业超过20秒收不到心跳会终止自己的子进程树；本地异常退出路径尽量保持相位30秒以等待采集停止。操作系统崩溃/断电不属于软件保证。中断后检查未完成记录，再用同一有效配置续跑。不要手动删除ACTIVE.lock，除非确认对应作业已结束。

## 方向与相位响应：三个独立问题

1. 振幅像素(x,y)怎样映射到相机(x,y)：用中心、向右、向下的小块，确认实际坐标方向；后续用四角homography统一到模型方向，不再叠加一个未知相机镜像。
2. 相位像素怎样对齐振幅：用四个振幅小块与TL/TR/BL/BR局部光栅，辨认哪一块实际被调制。不能因相机看起来左右反就推断相位要上下反。
3. 灰度怎样对应物理相位：正负光栅的衍射方向用于确认方向，但 `255-g` 本身不是线性相位校准。`linearVoltage`是线性电压，不是线性0～2π相位；原厂示例明确说明这一点。

不能拿全0和全255均匀图的亮度判断相位符号：均匀全局相位可能不改变强度。也不能仅凭正负光栅移动就确认所有灰阶的相位线性度。若需要定量恢复模型相位，需适合532 nm/工作温度的相位LUT，或另做相位响应标定。

联合诊断入口（只在两台GUI关闭、设备独占时运行）：

```powershell
& $py phase_joint_probe.py --out results\phase_joint_<唯一编号>
```

它记录原始CCD及最终下发相位SHA，不自动覆盖正式翻转、ROI或LUT配置。相位测试结束先写纯黑，再关闭SDK；SDK关闭后不保证面板仍保持最后图，需要时重新打开GUI加载纯黑。
