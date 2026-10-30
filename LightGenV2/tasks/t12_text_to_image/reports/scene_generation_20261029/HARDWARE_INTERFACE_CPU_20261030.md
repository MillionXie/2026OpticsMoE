# F9新光路接口：CPU准备验证，未采集（2026-10-30）

本页续接并保留`HARDWARE_READINESS_20261030.md`历史快照。主线程已补齐持久资产：
34份基础文件14,583,745,230字节，清单SHA
`de7655ad9ffc0fea0eed12dbceb9a8e4d01d0f7b7b11c396ff3751bcacf4dd2f`。
本地三PT和25份数据/JSON已由主线程逐SHA验收，不再称资产必须依赖/tmp。
完整模型清单含未被Qwen分支加载的原生CLIP资产，不计成实际部署参数。

**状态：新的离线接口及参考包已实现，CPU检查通过；真实设备provider和现场标定未放行。**
不操作实验台、不导入SDK、不创建定时任务、不抢占原独立500评分GPU6，不改T18。
原线程拥有训练、独立评分、README与最终放行；本线程单写新scene_lab_*及打包profile。

## 已实现

- `scene_lab_protocol.Ledger`：固定TRAIN cohort按六段全层门控；上一层所有请求收据齐全
  才能准备下一层。用原模型前缀回填真实同身份CCD重算后缀；首个未测边界抛NeedCapture暂停。
  已有有效帧跳过，不盲重拍；缺失、乱序、ideal/FFT回退、变更输入/会话/模型均拒绝。
- 账本绑定F9/F6/源码、RGB/prompt/seed、层、bounded振幅/总相位及逻辑8bit码、前缀收据链；
  接收时另核曝光/增益/等待/朝向、dark/gain阵列SHA、原帧与固定校准后帧SHA。保留原帧，不覆写。
- `scene_lab_entry`：从原TRAIN608/76参考按类别轮转取固定两条pilot，使用source-only seed；
  不读目标、掩码、特征PT或SKU caption，不把TEST500导入适配。
- `scene_lab_native.NativeRunner`：接已有正确加载/冻结的F6原Qwen与F9原head，保留原codec、
  电残差、光学identity delta/top2/alpha，使用既有scene_optical_boundary（未修改）。
  原始RGB和自然两句prompt→一次最终Qwen→原F9 conditions，返回给原SDXL12步路径。
  本接口不替换SDXL/LoRA/采样器，也不自行选GPU、下载模型或启动设备。
- `load_head`严格恢复checkpoint的joint_image_context：E662=false、F9=true；strict加载
  原state，绝不能给F9装默认E头。原scene_paired_render/scene_optical_boundary保持原SHA。
- 会话模板明确calibration_approved=false，无默认曝光/坐标/LUT，未审会话不能创建账本。
  数字编码round255幅度、floor256总相位，保零、不二次tanh、不逐帧峰值归一。

按层全量编排需要多次**部分Qwen前缀重算**，不同于正常完整部署一次Qwen；开销单列。
最终所有六层齐备后，单请求完整infer一次Qwen，无FFT fallback；不能拿采集编排计单次延迟。

## 实际验证（均无SDK和GPU分配）

| 检查 | 实际结果 | 解释范围 |
|---|---|---|
| 新账本本地CPU测试 | 6 passed | 全层门、错误帧/变更输入/变更会话/缺CCD拒绝、保留已采帧、量化；使用显式synthetic_fixture，不是真CCD |
| ZIP独立临时解压测试 | 同6项passed | 不依赖本机源码目录；不是Qwen/SDXL端到端测试 |
| 服务器恢复真实E662/F9 head | 两份state逐bit相等，flag分别false/true | 真实CPU加载，不是假模型flag断言 |
| 原48份边界例连续复场重建 | 最大CCD NRMSE 3.8251711241e-7 | 真实当前propagator、518pad/478crop，确认总相位复场重建对应原ideal CCD |
| 同48例amp+phase双8bit数字传播 | 最大CCD NRMSE 0.007287164756 | 约0.729%的**局部强度归一RMS误差**，不是生成图质量下降或硬件5%过门 |
| 实例化propagator传播距离 | 0.1m | 解决通用settings含多个距离字段的歧义；现场距离仍待校准证明 |
| 固定TRAIN pilot输入 | lamp/pillow各1，原RGB SHA通过 | 原608清单，76参考，现场不读取TEST500；本机仍未复制76张TRAIN全部RGB |

48例是原开发输入的仿真示例，没有把ideal CCD记成实拍。只验证局部量化传播误差，
还未做新NativeRunner的完整原Qwen/SDXL端到端量化闭环或真正硬件反馈。
实际器件LUT/相位响应、native像素映射/保零、方向、dark/gain单位、曝光和排他占用仍待核。

## 源码和接口包

- 初版源码发布`583843a6bfa1b5057b371ef181cc788219b9952c`；Windows无参数长度限制
  Git blob流式打包修复`10e81050a8f20c2611d4dff4fb4961f5b8009990`；原server边界清单兼容
  `0e50ef93e49d5b45654c50e3538141e2bcfd2cd6`。
- 源码通过Git bundle及精确Git blob同步训练服务器，保留其运行HEAD，不整仓checkout。
  只有新的scene_lab_*、测试及非训练用builder分流受控导入；README/render/boundary/
  conditioning源码SHA保持不变，原评分进程未重启。不能称整仓三端HEAD已统一。
- 历史17M默认打包逻辑不改；显式新`--profile scene-f9-interface`才使用新接口包，
  不走旧build_sealed/20M预算。新profile不在import时加载Torch或设备。
- 包：`releases/T12_F9_Interface_CPU_20261030.zip`，133,724,587字节，
  SHA256 `8b77f03b20446f4f8512e427b028957f48493e8989987af360625ba8c761e308`。
  源码pin10e81050a；1598份Git源码、三PT/外部持久manifest、说明和包清单，共1604成员。
  所有成员SHA和ZIP完整性通过，独立解压CPU测试通过。0e50只修服务器无download_receipt
  时的量化审计清单选择；不把它谎称已包含在pin10e81050a的包里。
- 不重复打包14.58GB基础模型，不附TRAIN/TEST图片、CCD、现场LUT、SDK二进制或凭据。
  源码依赖包含历史实现，但新入口不调用旧17M runner。
  包清单明确physical_capture_ready=false、frozen_TEST_gate_passed=false，不能当现场启动包。

私有验证入口：任务`runs/hardware/scene_F9_interface_CPU_20261030/`，有
source_receipt、CPU_validation、quantization、download_receipt及TRAIN_pilot_input_check。
CPU_validation.json SHA256：
`3e29c12a6a77fbd43de533c307ad77fda7f56d7cda0898dd735d3ec377aa994d`。

## 后续正式放行仍需

1. 原线程确认独立500/1000评分最终report、quality_gates和自身GPU/PID释放；本页不替其作结论。
2. 取得实验台排他、新相机及几何/方向/响应校准证据，完成native BMP/SDK适配和信号守卫。
   Ledger只能验证记录身份，不能认证“physical_linear_camera”声明就是实际设备事实。
3. 用明确TRAIN/校准DEV小pilot核响应和新NativeRunner/SDXL端到端，按层全量采集。
   同一模型同RGB/prompt/seed与SIM逐类报告CLIP-T/DINO-I/CLIP-I/IR compact变化；
   负IR按(SIM-EXP)/abs(SIM)，不得拿局部量化NRMSE替代5%质量门。
4. TEST500不得训练/选模；换权重后保留冻结原SIM与新权重匹配SIM、新SHA与全新依赖身份。
   真实未达标就诊断，不保证或强凑5%；完整保留CCD/会话/校准/逐样本与设备释放收据。
