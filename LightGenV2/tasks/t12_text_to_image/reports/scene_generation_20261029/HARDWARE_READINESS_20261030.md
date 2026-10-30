# T12 F9：光路部署准备交接（2026-10-30，只读核验）

## 状态与边界

**准备已核验，正式采集未放行，实测5%尚无结果。** 本交接响应用户在“文生图”聊天
`01a0bdd0-28b3-77d3-ae28-ece2ae992646`要求联系光路实验对话的原始授权。
原线程控制训练、冻结500评价和模型选择；光路线程准备新场景模型专用接口。
本轮没有改模型、创建runner/ZIP、导入SDK、枚举相机、申请GPU或停止任何进程，T18不变。
没有使用定时任务。该报告不等于已获实验台排他占用权或已完成现场标定。

## 已核资产与来源

新模型是原生Qwen3-VL/SDXL场景生成，不是历史17M编辑器。下列服务器文件均实际计算
SHA256并与交接身份匹配，位置前缀为任务`runs/simulation/`，文件名均`best_checkpoint.pt`。

| 角色/目录 | 字节数 | SHA256 |
|---|---:|---|
| 电子E662：scene_two_sentence_qwen_paired_20261029 | 56458911 | 662ff90f9a46670b28bf86d61629c59e4fb619ed69a62e4ae52ed8527ea76f76 |
| 单组前端F6：qwen3_scene_LG_frontend_20261029 | 28371482 | 6f34f2d720060c9193cda0e19ffb8fb4d157bc03e3fd5e5c1433c60ad884471e |
| LG生成F9：scene_natural_single_EG_joint12_training_20261030 | 56460895 | f9fbdb7fd36d1c862686616a624e7c18f4e9149cef61b0704bbf97767de6eaec |

执行固定源码身份`d4ab78d5326cb90651c97fee1fa3b0e3f6c835ae`，当前审阅README/接口
commit `76e5f5a38e80fd1504522a15e0024ee874e6344e`。服务器主HEAD不代替执行身份。
F9依赖F6、完整Qwen/SDXL/IP-adapter及合并任务LoRA、对应特征来源与采样协议，
不能单独只传F9小PT后声称闭包完整。服务器三个`/tmp/t12_*_asset_cache_20261030`
路径存在，但路径存在不是完整持久化模型分片SHA清单，也不是实验室已经具备资产。
本机三个PT均未落地，实验室标准T12路径的F9文件亦未找到；其他路径是否已有副本未穷举。
下一步应由主线程提供持久资产清单与独立500放行包，再按manifest+SHA一次传输，源码只走Git。

## 冻结评价门：不得跳过

本次只读快照：2026-10-30北京时间约19:03（Unix1793358201.506）。
`scene_natural_frozen500_workflow_20261030`父4016796和电子子4017004仍存在；电子阶段实际
397/500，LG500/1000图评分报告尚不存在，未确认最终report/quality_gates/GPU释放。
这不是故障，也不是已完成；不重复启动，不进入TEST梯度或选模。

冻结清单在UTC10:42:30.495691先于预测建立：

- frozen_evaluation.json SHA：fa241dafc73f4d39aac3dfde2afc9c50c5045dc3afaa84ef8106070275bdd50c。
- request SHA：27b9ae4866937f5c92569ac68d260587da4dfaeec64eb0c290dd8f619ff7038e。
- 500请求来自10参考视图、80图文上下文、多seed，不能称500独立商品。
- 冻结电子30步与LG12步，输入256/输出512、IP.25、官方任务LoRA合并，v4自然两句提示。
- 原线程必须核最终report、质量门及自身PID释放，再明确放行新会话正式采集。

## 六段新接口与旧runner的不兼容点

`scene_optical_boundary.py`的provider签名：
`provider(stage, bounded_amplitude, total_phase_radians, ideal_or_None) -> calibrated CCD tensor`。
返回有限、非负、完整`[B,478,478]`强度；非诊断模式ideal为None，不允许缺CCD时静默FFT回退。
原模型学习到的codec、电残差、光学delta后的identity残差、top2与alpha融合不变。

顺序必须为：**vision_router → vision_expert → vision_global → language_router →
language_expert → language_global**。每模态保留视觉6/语言4原生TF，各一个router、一对E/G，
4专家top2，共6周期。478 active/518数值画布/224 expert；原始RGB与文字只经过一次Qwen，
不得回输生成图、使用参考VAE编码旁路。当前top2偏固定，不能宣称均衡动态路由。

旧`build_lab_package.py`调用17M sealed editor，旧`lab_shs8um/main_layerwise.py`只接受
17026642参数与固定旧PT，旧`ccd_bridge`及README是语言先行；不可用于新F9。
旧回调对输入相位和bounded映射的处理也不能直接复制。新边界只是SDK-free替换点，
还缺面向真实Qwen的按层全量暂停/恢复、帧归属、BMP/SDK适配、单位标定和模型重放runner。

### 振幅、相位与响应尺度

- 有界`tanh(abs(field)/.5)`在原图内仅一次，保零并保留输入符号/复相位。
  导出的总相位是angle(input)+angle(modulation)，不能仅导出mask相位或对绝对值再tanh。
- 振幅BMP为round255*a，无逐帧峰值归一；设备振幅响应逆映射、相位灰阶LUT另属现场会话。
- 当前clean eval相位层是单位模exp(iφ)，且zero_order训练扰动关闭。
  不要把“光学残差”混同T18的rho=.3复振幅漏光配方：此T12还含`optical=x+optical_delta`
  的软件identity残差。若未来使用非单位模调制，新provider必须重新核abs(modulation)，
  现边界不能凭名字自动保证所有rho配方等价。
- 原理想FFT重放一致不证明8bit振幅/相位量化和器件响应已匹配。48例中round255振幅
  最大绝对误差0.001960814；下一步仍需数字量化闭环检验、相位取整约定和现场响应核验。
- `checked_ccd`只检查形状/有限/非负，不会自动求dark/gain，也不能保证物理强度单位。
  必须固定暗场、背景/增益、曝光/ROI/方向、线性响应与SIM强度的尺度；不得逐帧按理想CCD拟合。

### 物理几何未放行

服务器通过实际settings加载核对：逻辑pixel_pitch=17µm，532nm，language_optical_distance=.1m，
expert_pitch=254。通用settings还含expert到global及global到detector的.05m字段，不能仅凭
字段名字推断当前每次OEO周期的硬件距离，应核最终构建的实际propagator与会话距离。
同一配置继承的硬件元数据为振幅SLM17µm/相位SLM8µm；这不是现场器件实测证明。
**不得把T18现场双8µm/1016像素映射，或旧T12的方向、2000us/400us曝光直接套入。**
相机型号/SDK、器件像素间距、屏幕尺寸、入射波长、传播距离、相位LUT适用性、坐标/朝向、
native ROI、曝光/增益/等待、暗场与线性强度尺度须绑定新会话后再accept。
既有phase LUT若只有linear voltage，不得写成已标定linear phase。

只读进程筛查未见匹配的Python采集入口，T18 SDK锁不存在，但命中多项IDE/WebView，
筛查不穷尽所有厂商GUI/他人会话。不能据此称实验台空闲或自动抢占。实验室盘可用约2.93TB，
不构成清理旧CCD的理由。

## 已做验证与尚缺闭环

1. 本地48个NPZ全部重算SHA，对应8请求×6段；幅度/总相位/理想CCD三数组均[1,478,478]，
   有限、振幅[0,1]且保零、相位[0,2pi)、理想CCD非负。它们仅为仿真示例。
2. 原服务器CPU的`tests/test_scene_optical_boundary.py`实际3 passed：负/复相位保留，
   malformed/missing CCD拒绝且无fallback，限定六段顺序/单组结构。没有分配GPU。
3. 原边界run已证普通/记录理想/无FFT理想重放隐藏状态及PNG相等；本轮未重复生成或评分。
4. 没有新硬件provider，没有相机连通证据、native BMP响应接受或现场校准会话。
5. 正式执行应先独立500过门→资产闭包→排他/新会话→TRAIN/校准DEV小pilot→按层全量。
   第k层采完后，所有第k+1层输入必须由本批前缀实测CCD重算，不得批量显示本包理想后段输入。
   CCD身份至少绑定source+PT+前端+prompt/seed+层+输入/phase/LUT/session SHA；所有后缀依赖可追溯。

## 5%实测接受口径

与**同一冻结LG**、同RGB/prompt/seed/采样的SIM逐样本配对；不是拿电子E662的10%门槛
替代本次SIM–EXP5%目标。总体与lamp/pillow分别列CLIP-T、DINOv2-I、CLIP-I和IR compact。
对正值指标，下降=(SIM-EXP)/abs(SIM)；IR负值同样使用此式，不能直接取EXP/SIM。
近零参照须报告绝对差而非制造百分比，所有指标方向以评分器定义为准。
附逐图失败与成对LPIPS/PSNR作为图像变化量；代理均值不是逐属性正确率。
只用明确TRAIN/校准DEV适配；独立TEST500不进梯度、超参选择或反复刷门槛。
若模型/头/权重变化，新SHA需新独立SIM参照，原冻结SIM与新匹配SIM同时保留，不混权重凑5%。
实际未达则如实报告，不能许诺未测精度。保留新会话/CCD/标定/权重/逐图结果及自有PID与SDK释放收据。

## 可复核私有证据（不进Git）

任务`runs/hardware/scene_F9_preparation_20261030/`：

- server_readonly_inventory.json SHA256：9cd0a75797a2ace414c635bfef2338bf2a2517eeeceadacd756997e334bc0af1。
- local_boundary_integrity.json SHA256：d453ecf34da84cccd3341c37e48bd29de2f4a36dccd26ef042a994087eab0222。

没有更新T12训练代码、主README、源线程工作流或T18结果；本文件是准备快照，
后续评价/设备状态必须重新读取，不能把本页397/500当作实时进展。
