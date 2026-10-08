# T12 小版光路鲁棒性：最新交付

主候选：`small_spatial_17m.pt`，17,026,642计入参数，SHA256 `5496204d6a546df5dddef80fef00bc3f4e45479b02f7fd91a310d448bc6c94ed`。部署用 **candidate_spatial_17m_final.zip**；压缩包中正式权重名仍为 `assets/small.pt`。旧 `candidate_spatial_17m.zip` 是同一权重的中间封装，不要与最终封装混用。

原 `small.pt` 为9,958,098参数的保守备选，SHA `3f82ab04…`，**不是17M主候选**。之前13cf发布候选在相邻 `t12_channel_robust_20260927`，原实测3901e4fb候选仍不覆盖。本次没有删除历史权重或部署到师弟电脑。

## 效果与限制

`spatial_object_contact.png`、`spatial_background_contact.png`、`spatial_joint_contact.png` 分别为换目标、换背景、联合编辑；六列依次输入、GT、clean、stress、severe、extreme，三行灯/桌/靠垫。各模式取首个固定样本，不是挑最好看的。`spatial_visual_audit.json` 保存104条固定VAL样本的指标、专家和六阶段幅度/强度统计。全部为合成通道评估，不是CCD实测。

完整VAL2304：

| 通道 | 原实测用3901权重 PSNR/SSIM | 上一13cf候选 | 最新17M候选 |
|---|---|---|---|
| clean |33.6471/.920208|33.5685/.919405|34.1925/.924963|
| stress |25.4891/.849413|25.9005/.854884|27.7766/.877212|
| severe |24.6270/.839467|25.0511/.845971|27.0017/.870978|
| extreme |24.0030/.831937|24.3912/.838293|26.3708/.864833|

相同RTX4090、256×256、batch4、种子与评估实现。每模式768条。强扰动仍比clean低7.19dB；物体union ROI强扰动PSNR：换目标25.9604，联合编辑25.2654。原物体残影和纹理模糊仍存在，不保证真实光路优于这些模拟。尚无本次实测指标，原23.1658/.816176不能算成新权重结果。

固定权重后的独立TEST2304（未用于选模）：

| 通道 | 上一13cf候选 PSNR/SSIM | 最新17M候选 PSNR/SSIM | MSE降低 |
|---|---|---|---|
| clean |33.3754/.918042|34.0190/.923624|14.82%|
| stress |25.8466/.853205|27.7058/.875446|30.92%|
| severe |24.8471/.842593|26.8296/.868111|31.04%|
| extreme |24.3024/.836348|26.2807/.862871|31.50%|

全部训练/评估结束后GPU2=13MiB、GPU4=15MiB，无遗留本任务GPU进程，其他AI进程未触碰。最终ZIP SHA256 `6736765c330f32dc8b979c458afbb5c7cc6968e75dd3a4be9cc458894512f055`，源码commit `ce185f46993460d8ae5813792b5fa5eba40b8f04`；后续文档commit不改变包中模型/源码。

## 架构、alpha、专家

保留原tokenizer/冻结Qwen词嵌入，两层缩窄Qwen-style文字头、语言与视觉各router/top2-expert/global、并行电子残差及CNN decoder。mask仍为478×478有效范围，四224×224专家分区；纯相位958,728参数。冻结311,164,928词嵌入按先前约定单列，不计20M预算。

新加160维文字条件→224×14×14空间先验，7,068,544参数，在两级视觉光电融合之后、CNN decoder入口注入；增强封闭监督任务的形状约束，不是查询商品模板。没有把GT掩码或目标图片输入模型。输入仍图像+文字+固定seed噪声，输出整幅RGB；不是GT贴图，但已有可学习源图软融合，所以仍可能产生轮廓残影。

alpha下限.35/上限.75；实际language=.4730375/.4617593，vision=.4615397/.4667621。alpha是RMS对齐特征融合占比，不是功率占比。完整VAL干净语言专家选择率100%/100%/0/0（明显偏斜）；视觉77.86%/69.40%/30.60%/22.14%（top2合计200%）。视觉在stress/severe/extreme下相对clean换专家比例27.17%/34.77%/38.45%；详细概率和逐图分配见evaluation JSON。没有为追求均衡强行改路由。

## 扰动和训练

30%零级泄漏指相干叠加前名义**功率比例**：`P(a*(sqrt(1-eta)*m+sqrt(eta)*exp(i*delta)))`，不是给RGB加常数。severe覆盖15–60%泄漏功率、±3逻辑像素错位、k-space幅相扰动、低频相位抖动.22rad、5%块相位旁路及CCD增益/偏置/读出与强度相关代理噪声。extreme为30–70%、±4、.35rad、10%旁路，仅留出压力评估。噪声是仿真强度单位，未按真实CCD标定。

TRAIN训练，VAL选模；1000steps选step800，不选最终step。combined/strong/severe轮换、渐进增强；干净教师anchor、区域损失、融合特征一致性、router KL、圆周phase TV共同保性能。前几轮失败/退化试验未部署；本轮未新加GAN。CCD分位数校正和14.7M卷积细化试验不启用。

**幅度/BMP编码不变**：相位调制前保零 `tanh(abs(field)/.5)` 且保留复相位；BMP仅round(255*a)，绝不再按峰值/批次缩放。

训练源码commit `4065cbfa2`；最终封装commit见 `candidate_spatial_17m_final.manifest.json`。分支 `codex/t12-physical-robust-v2-20260927`，训练脚本/源码与22项单测已Git推送。服务器工作树 `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t12_physical_robust_v2_20260927`。训练完整argv/数据SHA见 `spatial_protocol.json`；历史与选模见 `spatial_history.json`、`spatial_report.json`。`val_spatial`/`test_spatial`及对照目录含四通道汇总、逐图JSON和运行协议。

本任务不操作光路。请另一AI先用新目录、新六阶段CCD进行旧3901/新5496配对VAL验证；不复用旧权重CCD，不覆盖candidate_bounded。通过小规模验证后再采集完整2304条。
