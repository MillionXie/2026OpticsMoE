# F9光路接口与现场校准（尚非完整实测放行包）

## 最新：分进程恢复取帧，TRAIN2已完成视觉Router和专家两层

2026-10-31用户在同一SLM白场期间以Mosaic实际看到光场，并保存2048×2048 uint16原TIFF。
Mosaic帧mean7999.84/p9933204；相机改为单独进程、SLM保持显示后，两张新SDK原帧
mean6984.83/7007.71、p9929102/29726，读回仍1999.8us/gain0/auto0，均无饱和。
这支持采用分进程后端继续部署，不应再把此前暗帧直接归咎于光源/器件；
尚未唯一定位某个SDK内部冲突函数，不改固件或替换DLL。

`scene_lab_camera_process`让原TucamCamera在独立子进程中持续取帧，父进程只持有SLM。
通信仅传递原帧路径与设备元数据，子进程不初始化SLM SDK，无理想/FFT帧回退。
**现场命令必须加`--camera-process`**，保持已验证的工作模式。
`pilot_vision_router_20261031_01`及`pilot_vision_expert_20261031_01`均return0，
各2张真实模型CCD已按完整cohort签收、核SHA并回传服务器；共完成4/12张模型CCD、2/6阶段。
专家输入通过原冻结F6/F9模型、以已测Router重算得到，不复用仿真专家输入。
曝光、坐标派生映射、器件LUT、模型和TRAIN2请求均未变，无TEST使用。
完整Qwen重放及SDXL生成尚待余下四阶段，**不宣称物理SIM–EXP5%质量已通过**。

## 2026-10-31：用户批准配准后接入实际采集入口

`scene_lab_capture`现支持两帧暗白复核，以及已准备ledger的一整层TRAIN采集；
导入模块不接SDK，不加载Qwen，不改曝光、gain、ROI、坐标或器件LUT。
原始2048相机数组永久保留，固定warp到canonical478，幅度逆响应仅应用一次，
相位仍为原总相位编码；每层输入ticket、设备BMP、原帧及接受收据均核SHA。
整层本cohort采完后才prepare下一层，不复用其他上游的旧CCD；已接受的正确帧只验证不重拍。
每次打开设备先用当前暗白检查新信号，有效才采模型数据；失败保留证据并关闭SDK，
不自动重复、更换驱动、重设曝光或把历史空间验收冒称当前整条光路已验收。

```text
python -m LightGenV2.tasks.t12_text_to_image.scene_lab_capture --config <registered_config.json> --calibration <calibration_registered_01> --out <new_check_run> --camera-process
python -m LightGenV2.tasks.t12_text_to_image.scene_lab_capture --config <registered_config.json> --calibration <calibration_registered_01> --ledger <physical_ledger> --stage vision_router --out <new_stage_run> --camera-process
```

固定TRAIN2真实Qwen字段由`scene_lab_pilot --mode prepare`准备，六阶段逐一处理；
只有实际CCD收据完整才允许`--mode infer`，之后另用原SDXL生成核验。
接口测试本身不代表采到模型CCD；最新实际采集数量见本页顶部及run收据。
空间配准获用户批准；线性相位尚未认证、原linearVoltage LUT不变，须在实际pilot中如实评估误差。

2026-10-30实际相机重试已成功：TUCam枚举1、2张原生2048x2048 uint16连接诊断帧。
`scene_lab_hardware`另完成11张初始几何检查及40张派生会话校准；原20261029 TXT不改，
固定派生采样映射使用3个校准轴点，5个全新未拟合位置最大误差2.160px，幅度响应heldout通过。
这些是非模型校准帧，不是模型TRAIN采集；原linearVoltage LUT也仍不是已认证linearPhase。

追加相位诊断发现白场消失：同BMP、读回设置及LUT，完整原帧p99由28988降至876，
后者与暗场一致。有限白场canary两帧后停止并释放；未盲重拍或改变曝光/物理合同。
截至10月30日晚该步骤模型TRAIN CCD仍0；10月31日恢复结果见本页顶部。
不得据此前或当前部分采集，宣称物理5%或完整Qwen/SDXL实拍通过。
详见[现场接续证据](reports/scene_generation_20261029/HARDWARE_20261029_SESSION_20261030.md)。

模块`scene_lab_protocol`管理按层全量账本，`scene_lab_native`把已有冻结Qwen/F6和F9头
接到既有`scene_optical_boundary`。旧17M模型、旧包和旧runner均不改变。
尚未运行真实Qwen的TRAIN硬件重放或SDXL实拍生成，因此不得称现场完整部署已通过。

## 正确使用顺序

1. 主线程独立500评分/释放过门后，现场核对排他、设备与标定。当前模板无默认曝光/坐标；
   calibration_approved=false会拒绝创建账本，不能把模板当校准证书。
2. 使用`scene_lab_entry.train_requests`从原TRAIN608输入清单按类别轮转确定pilot（默认2）。
   只读取白底RGB、prompt和source-only种子，不读场景GT、掩码、特征PT或SKU caption。
   测试/DEV选择不混入此TRAIN入口；TEST500不允许作适配。
3. `Ledger(...,create=True)`将模型F9/F6/source、请求及所有会话校准身份冻结。
   设备坐标/方向/相位LUT/振幅逆响应/暗场/增益的SHA、曝光等必须显式填写。
   模板是待审字段，不含真正标定。固定dark与gain_map对应canonical478数组的array_sha。
4. 注入**已有正确加载的**原Qwen/F6、processor和`load_head(F9)`返回的冻结head到NativeRunner。
   原head恢复joint_image_context：E662=false，F9=true，strict=True；绝不新建默认E头。
   该adapter不负责基础模型下载、GPU选择、UNet/LoRA/SDXL组装，也不调用SDK。
5. 对vision_router调用`ledger.prepare(stage,runner)`，每个请求到首个缺帧边界即抛
   NeedCapture，保存逻辑幅度/总相位/8bit码与ticket。整组本层输入齐备再交由现场driver采集。
   ticket**不是native BMP**；设备重采样、保零/方向、响应逆映射与相位LUT仍需独立现场adapter。
6. 调用`ledger.accept`导入外部driver提供的canonical线性原帧和同一审核dark/gain；
   核ticket/session/请求/曝光增益等待/方向/UTC等。不接受理想帧fallback。
   此函数验证声明与文件身份，不会认证相机实物或自动证明provenance为真。
7. 前面本层**所有请求**收据齐全后，才允许prepare下一层。前缀通过原网络重算，回填
   同ticket已测CCD；前层输入或收据变化会拒绝复用，绝不沿用仿真后段输入。
8. 六段全齐后，`ledger.infer(request,runner)`完整Qwen前向一次返回F9条件，交还原
   SDXL12步、CFG7.5、IP.25、合并任务LoRA与source seed路径。该接口没有新增生成器。

原生Qwen完整顺序：vision_router→vision_expert→vision_global→language_router→
language_expert→language_global。原codec、电残差、optical identity delta、top2及alpha均保留。
按层采集必须重算前缀，故每请求有多次**部分**前向与CPU/GPU状态重建；这是采集编排开销，
不能冒称正常推理Qwen只运行一次或计作单次部署时延。最终完整重放是一次前向，无FFT回退。

## 编码、响应与不可逾越边界

- 输入已tanh有界，仅round255幅度，phase总相位模2pi后floor256，不二次tanh或峰值归一。
- 输入符号/复相位已由boundary并入总相位，设备端不得再叠加一次mask相位。
- raw必须线性、方向/裁切规范；校准为max((raw-dark)*gain_map,0)。gain_map应为同会话固定
  响应/单位映射，不是逐样本拟合理想CCD；坐标、方向和LUT在此处只绑定SHA，不代替driver实施。
- 该逻辑量化检查不等于数字量化Qwen/SDXL质量闭环或硬件SIM-EXP5%过门。
- 当前接口不做SDK排他锁、硬件预览或曝光选择；现场driver必须另核设备所有权及有效信号守卫。
- 确定的字节身份不保证相机物理状态不漂移，仍需校准DEV的响应与误差检查。

## 账本CLI与打包（下列命令本身不采集）

```text
python -m LightGenV2.tasks.t12_text_to_image.scene_lab_entry --mode init --out <新run> --session <已审session.json> --train-manifest <TRAIN_input_manifest.json> --reference-root <scene_two_sentence_pairs目录> --count 2 --source-commit <已发布commit>
python -m LightGenV2.tasks.t12_text_to_image.scene_lab_entry --mode inspect --out <run> --source-commit <commit>
python -m LightGenV2.tasks.t12_text_to_image.build_lab_package --profile scene-f9-interface --source-commit <已发布commit> --asset-manifest <已验asset_manifest.json> --asset-manifest-sha256 de7655ad9ffc0fea0eed12dbceb9a8e4d01d0f7b7b11c396ff3751bcacf4dd2f --output <本任务releases内新zip>
```

新profile仅导出Git源码、三PT和外部持久资产清单；不复制14.58GB基础模型、不含CCD、现场
标定/SDK/凭据/训练或TEST数据。manifest明确physical_capture_ready=false。旧17M不加profile
时走原逻辑，仍检查17026642模型与原权重，不接受F9伪装旧包。
