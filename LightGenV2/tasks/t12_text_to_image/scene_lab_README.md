# F9光路接口与现场校准（尚非完整实测放行包）

2026-10-30实际相机重试已成功：TUCam枚举1、2张原生2048x2048 uint16连接诊断帧。
`scene_lab_hardware`另完成11张初始几何检查及40张派生会话校准；原20261029 TXT不改，
固定派生采样映射使用3个校准轴点，5个全新未拟合位置最大误差2.160px，幅度响应heldout通过。
这些是非模型校准帧，不是模型TRAIN采集；原linearVoltage LUT也仍不是已认证linearPhase。

追加相位诊断发现白场消失：同BMP、读回设置及LUT，完整原帧p99由28988降至876，
后者与暗场一致。有限白场canary两帧后停止并释放；未盲重拍或改变曝光/物理合同。
目前模型TRAIN CCD仍0，不得创建假审核会话、宣称物理5%或完整Qwen/SDXL实拍通过。
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

## CLI与打包（均不采集）

```text
python -m LightGenV2.tasks.t12_text_to_image.scene_lab_entry --mode init --out <新run> --session <已审session.json> --train-manifest <TRAIN_input_manifest.json> --reference-root <scene_two_sentence_pairs目录> --count 2 --source-commit <已发布commit>
python -m LightGenV2.tasks.t12_text_to_image.scene_lab_entry --mode inspect --out <run> --source-commit <commit>
python -m LightGenV2.tasks.t12_text_to_image.build_lab_package --profile scene-f9-interface --source-commit <已发布commit> --asset-manifest <已验asset_manifest.json> --asset-manifest-sha256 de7655ad9ffc0fea0eed12dbceb9a8e4d01d0f7b7b11c396ff3751bcacf4dd2f --output <本任务releases内新zip>
```

新profile仅导出Git源码、三PT和外部持久资产清单；不复制14.58GB基础模型、不含CCD、现场
标定/SDK/凭据/训练或TEST数据。manifest明确physical_capture_ready=false。旧17M不加profile
时走原逻辑，仍检查17026642模型与原权重，不接受F9伪装旧包。
