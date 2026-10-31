# F9光路接口与现场校准（尚非完整实测放行包）

## 2026-10-31 本轮完整收尾：数值可进5%，最终图像质量未全放行

本轮不再追加训练。完整结果见原run
`scene_F9_DEV128_registered_20261031/final_delivery_report.json`。
四轮均为完整DEV128、同题RGB/文字/种子/初始噪声、原12步/CFG7.5/IP.25，
原F9固定参照始终保留。下表为总体/台灯/靠垫×四项指标中**最差相对下降**，不是准确率：

| 原电子头适配 | 固定原F9同卡参照 | 同适配权重SIM参照 | 判定 |
|---|---:|---:|---|
| 首版 physical-only，lr3e-5 | 0.36823% | 7.39843% | 保留为较优图像候选；双门未全过 |
| LR对照 lr1e-4 | 1.57064% | 8.98353% | 未改善；原图有新灯罩缺失 |
| 双域w1 | 2.20934% | 4.38644% | 数值双门过；全128原图新增灯罩/灯杆消融，否决最终采用 |
| 双域w.1，实测优先 | 1.03340% | 9.01449% | 相比w1部分恢复，但双门未过且局部错误仍在，不替换首版 |

第三轮完整评分真实落盘为**08:40:52北京时间**；第四轮为**09:05:57**，
第四轮并未赶上09:00目标。四轮所有自身GPU作业已返回0并退出，采集SDK早已释放；
未停其他窗口、未新建自动化、未追加采集、未改变模型结构/曝光/坐标/LUT。
**没有任何一版同时满足双参照5%门与最终原图批准，不能宣布最终质量全部达标。**

推荐保留首版作图像质量回退，而非冒称双门最终版：best
`dc151164a4e7222a9183cf43e24ed0e832377f5f8e940300eb32a2a2283702dc`，
位于原TRAIN run的`adaptation_20261031/head_fit/best_checkpoint.pt`。
第四轮best
`e3c4798efc1e9a551691b2bfc52063afcbbb746536ecc392657d126d09e7d8a2`
仅保留候选证据，不覆盖原F9或首版。首/三/四版best/last及协议/历史/报告已在本地、服务器和
实验室核SHA，第二版在本地与服务器保留。所有128组PNG、逐图指标、LPIPS/PSNR、原图审查
和参考视图重采样的辅助区间均保留；代理均值通过不代表属性全正确或独立泛化。
独立TEST500始终封存。TRAIN608共3648张、DEV128共768张实际模型CCD均未删除，
raw仍在实验室、canonical数据与收据在计算端，不称原raw已三端镜像。

## 2026-10-31 08:43：双参照数值通过不覆盖原图否决

第三版`head_TRAIN_dual_domain_20261031`已完成同一608 TRAIN、1200步，best SHA
`12d387215db01fb71262247ea92dfbd1fe9085c4bc43906d8801c4d63a03a655`。
其`quality_adapt3`全部原始DEV128评分完整：相对固定原F9同卡SIM最差下降2.20934%，
同适配权重SIM–EXP最差下降4.38644%，两套12项均值门均通过；相对历史固定F9最差下降3.40473%。
这不是属性正确率或独立TEST结论，按参考视图重采样的辅助区间仍可跨5%。
32板全部128上下文的原图查看发现多处新增灯罩/灯杆消融、裸灯泡和漂浮亮块，
故`visual_review_EXP_adapt3/manual_visual_review.json`明确拒绝最终采用，不能凭数值通过掩盖。

首版仍保留为原F9参照下的较优实测候选（固定原F9最差0.36823%，同权重门7.39843%未过）。
第二版LR对照固定原F9最差1.57064%、同权重8.98353%，且有新灯罩缺失，未替换。
原版256张固定控制图在三次评分中逐图四项分数完全相同、指标资产SHA相同，未弱化参照。
第一/第三版best、last及协议/历史/报告均已在原实验室任务目录SHA验证；第三版仅是候选备份，
不是批准上线。所有原PT、有效/失败证据和CCD均保留。

CPU全608 TRAIN复核第三版两域条件误差：actual约.002833、nominal约.001335，原PT中
除head/source/新增适配元数据以外所有原状态逐项/逐Tensor不变，原7,791,616参数严格重载。
为避免仅压低nominal误差而损伤实测图像，显式追加**同方法唯一一个权重对照**：
`head_TRAIN_dual_domain_w01_20261031`，唯一配置变化w1→w.1，其余同初始F9、608 TRAIN、
seed、1200步、batch4、lr3e-5及所有光路/推理合同保持。仍只用TRAIN，DEV仅分析/评估，TEST500封存。
这是一项记录在`physical_priority_control_launch.json`的有限协议追加，不是无限候选扫描；
09:00仍是争取目标，若真实完成稍晚必须如实标时，不能提前填完成或用半量结果称通过。
实时状态及最终判定以该run的完整生成、评分、人工原图记录为准，无周期自动化。

## 2026-10-31 08:02：首轮完整DEV128，固定F9参照通过，同权重门尚未通过

本轮已完成TRAIN608×6=3648张、DEV128×6=768张实际模型CCD，SDK已释放。
原raw在实验室，canonical CCD/收据在计算端；紧凑报告、best/last、实际TRAIN特征及
完整生成PNG已有本地SHA备份，不宣称原raw已三端镜像。独立TEST500仍未使用。

首版原条件头适配best SHA
`dc151164a4e7222a9183cf43e24ed0e832377f5f8e940300eb32a2a2283702dc`，
只用608 TRAIN拟合原冻结F9的匹配SIM条件，原7,791,616参数、F6/alpha/Qwen/UNet LoRA不变。
原版SIM、未适配EXP、适配EXP、同适配权重SIM各128张均完整，见
`scene_F9_DEV128_registered_20261031/quality_adapt1`。
总体/台灯/靠垫×四项原指标的12门相对固定原F9同卡SIM全部下降≤5%，最差台灯CLIP-T
下降0.36823%；对历史固定F9结果的附加比较最差靠垫IR下降1.09973%，也通过。
但同适配权重的SIM–EXP对照中台灯IR下降7.39843%，因此**双参照总门尚未通过**。
不把较弱的适配后SIM替换原F9，也不把代理分数当属性准确率或独立TEST结论。

32板全部128上下文已逐板查看并保留`manual_visual_review.json`：主体较未适配诊断图明显恢复，
仍有原F9已有的花纹/材质/形状错误；一例台灯出现新灯罩不清晰风险，不能称全图完美。
历史SIM在另一GPU生成，128初始BF16噪声身份相同、PNG并非逐字节相同；同卡对照另存。
旧EXP_F9进程加载3ff源后才遇到be4磁盘更新，真实加载源补充记录在
`EXP_F9_preloaded_source_supplement.json`，未覆盖原execution记录。

第二个有限TRAIN-only LR对照已训完并进入完整DEV生成，尚不能宣称质量更好。
`ff659546f`另准备可选双域TRAIN约束：原头同时以实际CCD特征和原SIM特征重构同一个
冻结F9条件目标，权重1:1，不增加推理模块/参数、不开TEST/DEV梯度。默认权重0保持旧行为。
真实服务器CPU6项测试通过（含两域梯度），只是单元准入，不代表训练或画质达标。
仅在第二版仍未过双门、TRAIN诊断支持域偏移且09:00前有完整复核时间时，补一个有限对照；
不无限候选扫描、不改结构/曝光/坐标/LUT、不新增采集、无周期监督。

## 2026-10-31 06:10：TRAIN3648张完整，进入原电子头适配

`scene_F9_TRAIN608_registered_20261031/full_capture_workflow_report.json`记录六阶段
各608张实际CCD全部完成；采集返回0、SDK与本阶段GPU退出，原TRAIN608输入不变、TEST未用。
`completed_signal_audit.json`逐张检查读回1999.8us/gain0/auto0，六层饱和比例最大值0，
语言Global的p99范围6977–7303；这些仅证明本次数据／信号记录完整，不代表生成性能达标。
原raw在实验室、canonical CCD与收据在计算端，不能称原raw已全部三端备份。

原完整Qwen实测重放已开始，随后仅训练原7,791,616参数条件头，原F6、alpha及UNet LoRA冻结。
新源`3ff002c01`在每请求前完整验证六份CCD／ticket的SHA，保留同一前向内的数组快照，
不再在前向前、回调中和记录链时三次重读相同大文件；不跨请求缓存、不跳过身份核验。
同一receipt链摘要格式保持。服务器真实Torch CPU测试15项通过，均为合成单元fixture，
不代表生成质量。已发布源码仅按Git指定文件同步，其他窗口运行版本未覆盖。
旧纯重放进程的读盘开销已留证据后停止并归档到
`adaptation_20261031_before_verified_snapshot`；当时尚未产生特征缓存或任何头训练梯度。
已采3648张CCD完全保留、未重拍。新`adaptation_20261031`使用已完成cohort直接逐请求核验重放。

后续完整DEV128的768张新CCD及同原F9／适配权重的生成对照仍未完成，
因此目前**尚无5%质量通过结论**。当前实时阶段以原run状态、progress、report为准。

## 2026-10-31 09:00前目标：先真实微调，再完整DEV128核5%门

用户追加要求争取北京时间09:00前将EXP相对SIM下降控制在5%以内，不代表已保证或测得达标。
全TRAIN608×6后原条件头适配已排队；随后有限DEV流程另采16个未用于TRAIN的参考视图、
8个原自然提示场景，共128上下文×6=768张实际CCD。全部4416张样本CCD与暗白帧分开计数。
DEV会话显式`cohort_role=dev/evaluation_only=true`，绑定原请求SHA，不把DEV伪装TRAIN，
也不改原TRAIN ledger身份；两个cohort的RGB来源SHA集合已核不相交。独立TEST500不参与。

`scene_lab_render`经原renderer已有observer接口回填六个真实CCD；RGB、提示、种子、
12步、CFG7.5、IP.25、合并原任务LoRA及精度保持，不换生成器或FFT回退。
验收同时保留原F9 SIM、未适配EXP、适配EXP与同适配权重SIM；不能用弱化的新SIM替换原F9质量参照。
总体/lamp/pillow的CLIP-T、DINOv2-I、CLIP-I、IR compact共12项，下降用`(SIM-EXP)/abs(SIM)`，
负IR同式、近零另列绝对差；原F9参照和同权重参照分别核5%。附逐图分数及LPIPS/PSNR变化，
代理均值不等于属性准确率，最终仍须人工完整对照查看。若第一轮未达且预算允许，仅再做一次
明确的TRAIN学习率/批量对照，不对TEST训练，不无限扫描。原版本及所有best/last均保留。

源`704bf889e`让native准备复用已经完成的整层前置检查，但每张前缀仍逐SHA重核；
冻结ledger的旧模型source身份与新runtime source分开记录，不重置已采数据。服务器真实Torch
CPU回调/身份测试10项通过，系显式合成fixture而非实拍性能。新流程无周期自动化或新增聊天任务。

## 2026-10-31：已转为TRAIN608整层采集，不再重复TRAIN2试采

用户要求直接全量，现用`runs/hardware/scene_F9_TRAIN608_registered_20261031`。
608是原TRAIN的图文输入条件数，六阶段共3648张模型CCD；暗白校准帧另计。
独立TEST500仍封存，不用于本轮适配。旧TRAIN2四张仅是两输入×前两层，不是全量。
截至本次文档快照，视觉Router和视觉专家均608/608已完成、SDK释放并全SHA回传，共1216张。
专家样本首末帧间874.878秒、相邻间隔平均1.4413秒；p99范围12958–13768，最大饱和比例0，
每帧读回均1999.8us/gain0/auto0。第三阶段视觉Global的完整608输入计算已自动启动，尚未采该阶段。
实时数量以原run的`full_capture_workflow_state.json`及实际ledger收据为准。
Router原任务在180张处因输入传输超时结束，原帧保留；同会话续拍仅补缺失428张，没有重拍前180张。
所有608份Router输入与实际CCD已签收；失效的中转传输证据不删除，改为校验主机密钥的实验室—计算端直传。

整层前缀全部实际签收后，原模型依次重算下层输入，这是推理，不是微调。
固定F6/F9、空间映射、1999.8us/gain0/auto0/wait240及器件LUT不变，仍使用已验证分进程相机后端。
使用用户明确允许共享的GPU1 RTX4090，显存比例上限.25、B1、请求间暂停30ms，不中断其他作业或等待另一聊天GPU。
显存上限不是计算资源隔离。运行是有限的单次六阶段批任务，不是周期监督或定时任务。

`58c2add3c`移除采集接受路径的二次方重复I/O：开层一次全cohort上游核验，每张接受前仍逐SHA重核该样本
完整上游收据链；不跳过身份、相位编码或原帧校验，不改变物理/数值模型。
全层完成后`scene_lab_adapt`可导出同次实测完整Qwen隐藏状态，再仅适配原7,791,616参数条件头，
以冻结F9对匹配SIM TRAIN输入的条件为目标；前面光学、alpha、Qwen及UNet LoRA冻结，不加网络。
此TRAIN重构损失不是生成质量，未完成适配或独立物理生成核验前，不宣称SIM–EXP5%门槛通过。
该有限后续作业已排队，只有六个实际CCD数据cohort全部签收才进入GPU；当前仅CPU等待、不占SDK。
608个实测请求与原SIM TRAIN输入已逐项匹配，既有F9头严格加载及CPU合成梯度检查通过；
该检查不是实际微调效果。最多1200更新/900fit秒，best/last分开保存，不替换原F9。
传输源`9258efc98`另避免在纯数据进度查询中再次全量读上游，修复120秒查询超时；
真实模型/相机入口的整层前置放行及逐样本SHA核验不变，原超时证据保留，未重新拍摄。

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
