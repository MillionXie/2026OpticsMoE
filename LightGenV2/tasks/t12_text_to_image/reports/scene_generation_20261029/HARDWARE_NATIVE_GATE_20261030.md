# F9原生接入核验通过；等待现场相机/标定确认（2026-10-30）

本页更新后续状态，不覆盖CPU准备和早期只读快照。原线程已正式交接独立500评价，
本地逐一核report/quality_gates/GPU_release/frozen manifest四项SHA与交接一致。
完整500电子与500LG生成/评分通过用户10%可比门，但**不是EXP–SIM5%已通过**。
handoff.json SHA为e5ea8ba5dd115d453014b8a3c4e7d50565a550fb02892ce55095b5a18097e0d8。
未把这500用于训练、超参或选模；以下原生自检仅两条固定TRAIN。

## 新入口实际原生核验

构造真实冻结Qwen/F6，恢复F9原head（joint_image_context=true）。对固定lamp与pillow
TRAIN输入，原始原生前向记录的六段理想CCD逐位一致地经新NativeRunner回填：
两请求的text/image/pooled三个条件向量最大误差均0。
接着用新Ledger按六层整组暂停/接受显式synthetic_fixture/前缀重算，六帧齐全后
完整推理，条件向量再次均误差0。不是只用假模型的账本单测。
统计20次新adapter完整/部分前向尝试，其中8次完整完成；另有原生基准记录前向，
不能把整个测试说成每请求只调用一次Qwen。正常六帧齐备infer仍一次完整Qwen。

源码`680c26395959d49b37104248a747d61a64dbeb34`。执行前按UUID检查
GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55无计算进程、仅14MiB，才使用已交接的GPU6；
其他GPU/进程未停止。自有PID4094160返回0、已退出且不在GPU计算进程表，释放收据保存。
原评分的源码、PID与数据没有重启/更改。未导入SDK、未打开相机/SLM、未生成新的SDXL图。

私有证据：`runs/hardware/scene_F9_native_validation_20261030_r2/`，
`native_check/report.json` SHA256：
`eded7e4fa55d69d330598128a7ff2dc5b694e3b6f67ce3af920ed8c0b573dfd9`。
source_receipt绑定实际新源码Git blob/源码SHA；report里的source_commit是保留的运行HEAD
ae9e5c0ff47f75a94f4b1ac35bfddb532d110b40，**不是把它冒称新增接口的发布commit**。
服务器仅导入精确Git blob，不整仓切换；render/boundary/训练README SHA不变。

## 失败证据与敏感性没有隐去

原生接入前修正了bf16 Qwen→fp32 head的转换，严格跟随原renderer，发布dc93f8392。
首轮把连续复场重建CCD用作逐位一致接口参照，text最大差0.41760254导致严格门失败；
自有PID4090299返回1并释放GPU，原失败/日志/源码/释放收据保留并下载。

R2没有放宽接口误差门。逐位相同CCD仍要求<=3e-6并实际全为0；连续复场重建另作
数值敏感性诊断，不将它称设备结果。lamp相对条件RMSE：text0.00077227、image0.00053410、
pooled0.00373704；pillow分别0.00021987、0.00047982、0.00176736。
这些是条件张量误差，不是CLIP/DINO/IR或图片成功率，也不能据此预称硬件5%达标。
原48例局部8bit CCD传播检查（最大NRMSE0.00728716）仍独立保留，不替代真实生成闭环。

## 接口参考包R2

使用`releases/T12_F9_Interface_CPU_20261030_r2.zip`，133727824字节，
SHA256 `f46c2b8c0fd74ffd613d82f7c6167234071e77c8f8ebe7ce5138148e5f1d0d42`，
源码pin680c26395。1605成员全部SHA/ZIP完整性通过，独立临时解压6项CPU测试通过。
包含真实三PT及外部基础资产manifest，不含14.58GB副本、CCD、SDK、现场校准/凭据。
manifest仍physical_capture_ready=false；真实完整Qwen接入已在服务器原工程核验，
ZIP不是自带全部基础模型及现场硬件驱动的可直接采集包。

**旧`T12_F9_Interface_CPU_20261030.zip`只留作开发审计，不再用于原生运行参考**：
它的pin10e81050a早于fp32 head转换修复。没有删除旧证据或覆盖原包。
原17M默认打包行为/固定模型约束不变，新显式profile才选择F9接口。

## 现场尚缺项：已向用户集中询问，不猜测、不试抢

只读盘点未见Python采集进程或已知锁；PnP的Camera/Image/Media类仅列音频设备与打印机，
按Dhyana/TUCam/Mosaic/Camera等名称筛查也未发现可确认相机。
这不是对所有厂商/PCIe设备的穷尽枚举，**不能推断相机一定断开或命令用户盲插拔**。
现有T18旧记录为1999.8us，旧geometry_spots_20261023坐标、identity方向和linearVoltage LUT；
它们不是T12新会话认证，不自动沿用。

用户需确认当前相机型号、最新坐标TXT完整路径、振幅/相位SLM像素间距及532nm/10cm条件，
并确认设备已开机可作小样本校准；问题已在当前对话提出。
拿到信息后才能选择正确SDK与native映射，执行排他/信号/响应验证并创建新session，
之后才是两条固定TRAIN的真实pilot与按层全量采集。不得为赶进度把未校准模板设为approved。
5%目标仍须同冻结LG/同RGB-prompt-seed的SIM与EXP总体/分类成对指标及逐图失败证明。
