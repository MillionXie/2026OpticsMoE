# T12 最终版本与复现入口

## 新增外部 baseline：官方 pix2pix-Turbo 微调（已完成）

官方repository `https://github.com/GaParmar/img2img-turbo`，SHA `86f54146590ffb4543c8cf85b5a36657da670924`，服务器干净外部clone `$AS/vendor/img2img-turbo-ssh`。不修改官方生成网络、VAE skips、LoRA目标或单步DDPM scheduler；仅将from_pretrained加载重定向到已验证的本地SD-Turbo FP16文件。官方日志的“Initializing model with random weights”指未加载任务LoRA checkpoint的初始化分支，**SD-Turbo骨干仍由from_pretrained加载**，不是随机骨干。模型文件SHA在protocol.json。

输入图→官方VAE encoder及skip特征→latent posterior sample→UNet(t999,原生逐token CLIP条件)→官方一步scheduler→VAE decoder及官方skip→clamp RGB。无Qwen、PCA或GT贴回。推理图仅UNet一次调用，确定分支仍有VAE posterior随机性；VAL/TEST逐图固定1042+索引并使用fork_rng，保留官方.sample()，不擅改成mode()。

| 组件 | 推理参数 | 微调参数 |
|---|---:|---:|
| CLIP文本encoder |340,387,840|0|
| UNet（含LoRA，rank8） |874,019,300|8,120,416|
| VAE（含LoRA rank4与skip） |85,038,607|1,384,744|
| 总计 |1,299,445,747|9,505,160|

词嵌入50,593,792按既有约定单列；排除后预算1,248,851,955。主表同时保留完整总参数与排除词嵌入参数，不能只用9.51M LoRA代表部署模型。VGG LPIPS、CLIP相似度网络、vision-aided discriminator仅训练，不计推理参数；无PCA固定条件库。此处总量是未merge LoRA计算图；后续如merge需要重审参数和速度，不能混报。

当前TRAIN20736/VAL2304/TEST2304，与现有ABO指令数据及真实透明掩码GT相同，256×256不裁剪、不左右翻转（避免破坏左右光照文本）。已完成3epochs，batch2累积4，lr5e-6、seed927、UNet checkpointing、BF16 autocast/FP32权重。官方推荐loss权重：RGB MSE1、VGG LPIPS5、CLIP similarity5、vision-aided GAN.5；完整VAL按逐图平均PSNR选择best step31104（20.863762dB），固定后独立TEST为20.746391dB/SSIM0.740299。保留best/last及全部2304条逐图结果；checkpoint SHA256 `3a347c58affb53d8e7efc583bb5aecdaa2ac33bd316d792c12c805fa837b4c7c`。训练源码 `7c75a00e3`，增加可复现全量TEST图像导出的源码 `a3674c071`；TEST评估未反向修改权重。

明确wrapper差异：当前将G的重建/感知/语义/GAN合成一次backward与optimizer更新，而官方trainer分成两次G更新；采用累积有效batch8、完整VAL PSNR选模而非官方小子集FID选模。生成结构不改，但不得写成官方训练脚本完全原样复现。独立TEST质量低于同任务Qwen baseline，且视觉上有偏灰、细节弱化；不能用这个实验声称外部预训练生成器普遍较差。

依赖隔离在 `$AS/pix2pix_dependencies`，未升级共享torch/diffusers。diffusers0.35.1、peft0.19.1、lpips0.1.4、vision-aided-loss0.1.0；完整CLIP源码 `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` 位于 `$AS/vendor/CLIP`。CPU数值/参数测试与真实GPU短程冒烟先行，失败run保留诊断，不部署。代码许可与SD-Turbo权重许可分别遵循其原始LICENSE，不声称模型权重属于ABO数据许可。

```bash
export CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549
AS=/DATA/DATA1/guest3/t12_assets
TASK=LightGenV2/tasks/t12_text_to_image
export PYTHONPATH=$AS/pix2pix_dependencies
$AS/venv/bin/python -u -m LightGenV2.tasks.t12_text_to_image.train_pix2pix_turbo_baseline --assets $AS --upstream $AS/vendor/img2img-turbo-ssh --output $TASK/runs/simulation/20260927_pix2pix_turbo_matched --epochs 3 --batch-size 2 --accumulation 4 --learning-rate 5e-6 --validation-every 5000
# 完成后只使用VAL选择的best，新的output，TEST不得用于调参：
$AS/venv/bin/python -u -m LightGenV2.tasks.t12_text_to_image.train_pix2pix_turbo_baseline --assets $AS --upstream $AS/vendor/img2img-turbo-ssh --output $TASK/runs/simulation/20260927_pix2pix_turbo_test --checkpoint $TASK/runs/simulation/20260927_pix2pix_turbo_matched/best_checkpoint.pt --evaluate test
```

固定TEST输出含 `images/reference`、`images/target`、`images/generated` 各2304张原生256 PNG，以及逐图CSV/JSON和report.json。训练在单张RTX4090、TEST在单张A100完成；训练PID527954、评估PID1075779均已退出。结果归档 `handoffs/t12_pix2pix_turbo_20260927/test_export`；五组汇总 `handoffs/t12_five_group_summary_20260928`，表格 `outputs/t12_five_group_summary_20260928/T12_five_group_performance.xlsx`。输出PNG只是浮点结果的round量化，不参与指标计算，不用GT掩码贴回、检索、锐化或超分。旧四组表保留原样。

## 四组原始汇总与五组扩展：2026-09-27/28 同任务 baseline

优先使用 run `20260927_matched_qwen28_baseline`（补训）和 `20260927_four_group_matched_baseline`（固定TEST导出），工作树 `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t12_physical_robust_v2_20260927`。历史段落中的跨任务baseline数值不作为当前主表。

训练源码 `b813628eaeb1687d4533bcedd9fb9998c5d5c6a5`，导出源码 `c657f959d877635549d68cfbb8bba4c668273adf`。PyTorch2.6.0+cu124、RTX4090、服务器t12_assets/venv。Qwen28/VAE冻结、原UNet和adapter训练，TRAIN20736、VAL2304、3epochs、batch4、accumulation2、lr2e-5、seed927。损失为latent MSE+.05 latent梯度L1，每8批加.1 RGB MSE。VAL逐图平均PSNR选中step15000（27.404973dB），末步15552不代替best；固定best后只做TEST评估。训练run保存protocol/history/report、best/last；新权重SHA `599805ad3062bebf67acf3b515f0a812fb843506643e18251004f180131b4b52`。

| 当前主表（TEST2304） | 预算M | PSNR dB | SSIM |
|---|---:|---:|---:|
| 小版原权重仿真5b4f |17.026642|34.277751|.927140|
| 小版decoder微调后EXP eeec |17.026642|31.552886|.906013|
| 大版仿真2a91 |149.755866|31.428599|.882211|
| 同任务补训Qwen28 baseline5998 |1834.345963|27.260453|.837653|
| 预训练pix2pix-Turbo，VAL选模后TEST |1248.851955|20.746391|.740299|

小版原权重EXP27.551289/.876325，微调权重仿真28.890612/.903006。主表前两行权重不同，不是同权重sim-exp对。小版物理微调只训练decoder、光学上游未变；原报告保留TRAIN-domain validation warm-start caveat。大版与baseline目前训练历史/损失不同，不能单独归因于光学优势。PSNR是配对保真指标，不是开放集生成证明。

指标：256原生RGB、clamp后映射[0,1]，每图MSE→PSNR后平均；SSIM11×11 Gaussian sigma1.5 valid。浮点指标在PNG量化前计算；旧整体MSE取log口径不能混用。TEST manifest SHA `331874abbb8d8c7a4ac5ee3d6e030f20611cb5a9e96d3092917b7a5c4d7d5d2f`；TRAIN/VAL SHA及缓存审核在protocol.json。导出seed1042+索引；小版与latent模型噪声形状不同，不宣称同一噪声张量。小版PNG floor与大版/baseline round仅导出量化不同，统一输入/GT实际像素差不超过1/255；不修改生成图。

```bash
export CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549
AS=/DATA/DATA1/guest3/t12_assets
TASK=LightGenV2/tasks/t12_text_to_image
QWEN=/DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
$AS/venv/bin/python -m LightGenV2.tasks.t12_text_to_image.train_matched_qwen_baseline --assets $AS --qwen $QWEN --output $TASK/runs/simulation/20260927_matched_qwen28_baseline --epochs 3 --batch-size 4 --accumulation 2 --learning-rate 0.00002
$AS/venv/bin/python -m LightGenV2.tasks.t12_text_to_image.export_matched_summary --assets $AS --qwen $QWEN --large /DATA/DATA1/guest3/t12_git_audited_20260926/$TASK/runs/simulation/20260926_large_detail/adapted_model.pt --baseline-checkpoint $TASK/runs/simulation/20260927_matched_qwen28_baseline/best_checkpoint.pt --reuse-large-export $TASK/runs/simulation/20260927_four_group_summary --output $TASK/runs/simulation/20260927_four_group_matched_baseline --batch-size 2
```

训练复现应使用新output，不能覆盖已封存run；固定权重复评不再训练。复用大版导出前严格校验其权重/数据/缓存/seed/2304行身份。本地交付目录 `handoffs/t12_four_group_summary_20260927`：四模型各2304张生成PNG以及输入/GT、逐图指标、SHA/CRC审核、同样本代表图。不包含缺失的微调后仿真PNG，不假造补齐。本轮未测新延迟、未操作光路。训练/导出自身GPU进程已退出，未停止其他AI。

日期：2026-09-26。正式代码分支：`codex/t12-audited-editors-20260926`。本轮训练源代码分别记录于报告 execution 字段；封装提交 bb7bfba2253108ef5da3d81203f4edc7f1510dc9，运行记录提交 15537441f，清理提交 dbda11c62179e62967f5169759437ee35093d978。

## 两套主权重

本地统一目录：
`C:/Users/Xml12/OneDrive/2026OpticsMoE/LightGenV2/tasks/t12_text_to_image/runs/simulation/20260926_consolidated/`

- large.pt / large_test.json / large_audit.json / large_overview.jpg
- small.pt / small_test.json / small_audit.json / small_overview.jpg
- cleanup_server.json：服务器删除清单和哈希。

服务器代码：`/DATA/DATA1/guest3/t12_git_audited_20260926`；任务 runs/simulation 下的 20260926_large_detail、20260926_small_detail 为本次完整训练记录，20260926_sealed_references 保留前一版可加载参考。

| 模型 | 预算参数/缓冲值 | MSE ↓ | PSNR ↑ | 边缘 L1 ↓ | FID变体 ↓ | KID ↓ |
|---|---:|---:|---:|---:|---:|---:|
| 大版 |149,755,866|0.00371867|30.32|0.0859617|35.37|0.005992|
| 小版 |9,958,098|0.00255774|31.94|0.0715071|86.24|0.016663|

统一 256×256、同一 2304 个测试指令对。MSE/PSNR衡量配对像素一致性，不等于纹理真实感；不能用小版更低MSE推出视觉更好。FID使用 torchvision ImageNet Inception-v3，**不是标准 TensorFlow FID**，仅作同协议内部比较。局部细节分数使用训练同类指标，不是独立 LPIPS。baseline 固定权重复评与计时见下节；它未按当前统一任务重新训练，不能用于结构优越性的公平结论。

大版上一轮 MSE 0.004986、PSNR29.04、FID变体48.68；小版上一轮 MSE0.002639、PSNR31.81、FID变体88.58。新大版改善明显，新小版改善有限；小版仍缺乏细纹理并存在局部重影，不能称已完全解决模糊。

## 执行和验证

Python：`/DATA/DATA1/guest3/t12_assets/venv/bin/python`。
入口：`audited_unified_run.py` 的 train/evaluate/audit/seal 子命令。具体完整训练/评估 argv、Git SHA、torch、设备和配置已写入每份结果的 execution 字段；以记录的 argv 复现，checkpoint 指向本目录 sealed 权重即可，不再需要旧 source 权重。

11 项结构测试通过，两套实际 GPU 审计均通过：language/vision真实调用、并行同输入、相位梯度、alpha下限、256输出、参数预算。使用的一张GPU已释放；未停止其他用户进程。

## 2026-09-26 baseline 固定权重速度与性能复评

源代码提交：190de2e494460db33c03c58e9ccbc0610ff5ce8b。run ID：20260926_baseline_comparison，位于任务 runs/simulation；本地同名目录含 timing.json、baseline_quality.json、execution.log。固定现有 baseline，不改变 Qwen+电子 decoder 架构、不重新训练。

GPU RTX 4090，batch=1，FP16 autocast（光学内部FP32），同一32-token指令与输入，256×256；预热10次、重复100次。CUDA事件起点为首个语言block输入，终点为RGB，不包括tokenizer、词嵌入、窄头输入投影/packing、加载和数据传输。是固定样例重复计时，不是全测试集平均延迟。

| 模型 | NN参数 | 固定条件缓冲值 | GPU全仿真均值/P95 ms | 全仿真加速 |
|---|---:|---:|---:|---:|
| Qwen28+电子decoder baseline |1,824,174,315|10,171,648|60.68 / 62.35|1×|
| 大版 |144,630,618|5,125,248|39.89 / 40.94|1.52×|
| 小版 |9,958,098|0|17.96 / 18.32|3.38×|

词嵌入311,164,928各自单列不计。baseline未使用Qwen vision tower/LM head，因此本合同内不是完整Qwen2.13B再加decoder；有效NN约1.824B。加入条件缓冲值，baseline预算1,834,345,963；大/小预算减少91.84%/99.46%。不计未参与推理的辅助分类router。

硬件代理计时使用缓存的router结果及expert/global CCD，保留电子幅度编码、装载、读出及融合；不能用这个代理输出评价质量。额外排除并行电子支路以遵循用户口径，仅为乐观下界，不是硬件端到端实测：

| 口径 | 大版ms / 加速 | 小版ms / 加速 |
|---|---:|---:|
| 去FFT与并行电支路，仅剩串行电子 |32.84|11.02|
| 上项 + 一次6.2682ms（此前约定） |39.10 / 1.55×|17.29 / 3.51×|
| 上项 + 两次6.2682ms（语言→视觉先后经过） |45.37 / 1.34×|23.56 / 2.58×|
| 保留电子并行支路串行GPU时间 + 两次光路 |48.80 / 1.24×|26.93 / 2.25×|

最后一行也不是物理并行实测；真实关键路径需要各阶段 max(T电子,T光)+外围电子，并计入设备传输。加速来自网络压缩及不同计算图，不能单独归因于光计算。

所有质量数值使用完整光学仿真/原baseline，未使用计时bypass、GT贴回或模板检索。同一2304对固定测试：

| 模型 | 整体MSE↓ | 整体PSNR↑ | 换背景PSNR↑（768对） | FID变体↓ | KID↓ |
|---|---:|---:|---:|---:|---:|
| 历史baseline迁移诊断 |0.123046|15.12|21.85|120.94|0.028966|
| 大版 |0.003719|30.32|29.04|35.37|0.005992|
| 小版 |0.002558|31.94|34.46|86.24|0.016663|

**baseline只训练过旧灯类换背景，未训练当前换目标/联合修改/扩展类别。**测试数据相同不代表训练协议相同；以上质量只能说明当前固定权重可用性，不能支持“光电模型公平胜过Qwen baseline”的论文结论。换背景子集也包含类别/数据迁移。公平质量比较仍需要保持Qwen+decoder架构、用同训练集/任务重新训练baseline（本次未做）。

复现命令（服务器代码根目录，AS=/DATA/DATA1/guest3/t12_assets，TASK=LightGenV2/tasks/t12_text_to_image，CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd）：

```bash
$AS/venv/bin/python -m LightGenV2.tasks.t12_text_to_image.benchmark_audited_editors --assets $AS --qwen /DATA/DATA1/guest3/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 --large $TASK/runs/simulation/20260926_large_detail/adapted_model.pt --small $TASK/runs/simulation/20260926_small_detail/adapted_model.pt --output $TASK/runs/simulation/20260926_baseline_comparison --warmup 10 --repeats 100
```

权重SHA256见 timing.json。数据清单 test.jsonl：331874abbb8d8c7a4ac5ee3d6e030f20611cb5a9e96d3092917b7a5c4d7d5d2f；instruction-cache：66f457115cc7b1058f3ee42be0fec5815f101780c22b4805ad08940966dae97f；embedding-cache：e7a855849ef22e470aafdcf8ee583968ca0c8d5d17ff42603b103c49e97502ac。11项结构测试再次通过，完成后本GPU显存回到15MiB，无自己的计时进程。

## 2026-09-27 bounded-channel 鲁棒性微调复现

服务器 Git 工作树 `/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t12_physical_robust_v2_20260927`，分支 `codex/t12-physical-robust-v2-20260927`；正式训练源码 commit `e06ea0685f6a9fca4b22129e3532c498ce36bd28`，完整 VAL commit `8cc585841745aa94be3ae1d6fdde8826e8f57aaf`。之后将完全相同的扰动参数显式移到 `configs/optical_channel_robust_v2.json`，不修改训练结果。16 项光路/架构测试通过。

原权重 SHA256 `3901e4fb5d4d249ef4472d4de33cc80858c5d7002ff404db2b53a9ff0ff08577`。主候选 SHA256 `13cf9a201bce42f58c19a0ff85fd11fe9940e18be036a86adf163320303edc3e`，路径 `$TASK/runs/simulation/20260927_channel_combined/best_checkpoint.pt`，selected step=600。参数 9,958,098，不新增部署模块或参数，冻结词嵌入沿用原统计约定。

只使用 TRAIN 训练（20736 指令对）；固定 96 VAL 选训练 checkpoint，最终完整 2304 VAL 比较两个候选。最小化 camera/combined MSE 的均值，并保护 clean 相对原权重 PSNR 降幅≤0.2 dB、SSIM 降幅≤0.002。选择在最终 TEST 前冻结，TEST 不参与梯度或选模。

| 完整 VAL（2304，逐图 PSNR 平均） | 原权重 PSNR / SSIM | camera 微调 PSNR / SSIM | **combined 主候选 PSNR / SSIM** |
|---|---:|---:|---:|
| clean |33.6471 / .920208|33.6429 / .920047|33.5685 / .919405|
| camera |28.9665 / .881981|29.1750 / .883676|29.2047 / .883973|
| combined |27.3636 / .867222|27.5323 / .868996|27.7775 / .871593|
| stress |25.4891 / .849413|25.6164 / .851273|25.9005 / .854884|

camera 为泄漏/错位/CCD 的组合；combined 在其基础加温和 k-space 和 phase bypass dropout；stress 增大扰动。三者是组合对照，不可据此声称单独某项增强有效。参数为假设的设备扰动代理，不是从 TEST 实测拟合所得的标定值。

选模后完整 TEST（2304，source commit `1eb670a3fd979af149c95aa4234aaabcd16879fa`）：

| 通道 | 原权重 PSNR / SSIM | 主候选 PSNR / SSIM |
|---|---:|---:|
| clean |33.4286 / .918732|33.3754 / .918042|
| camera |28.7182 / .879574|28.9739 / .881703|
| combined |27.2183 / .864712|27.6322 / .869324|
| stress |25.4729 / .848049|25.8466 / .853205|

TEST 未用于选择/修改权重。clean下降.0531dB，combined MSE下降14.56%，stress MSE下降15.48%。原权重本轮与原报告相差约.00023dB/.0000026 SSIM，新旧使用完全相同batch=4与实现。结束后GPU4显存15MiB，无遗留训练/评估进程；其他用户进程未触碰。

发布包 Git源码 commit `5164fc1ec868c1784dba15fec98be51c9038b78d`；ZIP SHA256 `d362271c9e429d00d4ea182d1e5da057acb2dc9933dbd07fa1be3e813c9a6c3c`。完整本地交付目录 `handoffs/t12_channel_robust_20260927`，含 small.pt、ZIP、manifest、选模记录、新旧 VAL/TEST 汇总及四通道逐图指标。

泄漏使用相位调制因子 `m_mix=sqrt(1-eta)*m+sqrt(eta)*exp(i*delta)`，再传播入射场 `P(a*m_mix)`。eta=.30 指相干叠加前名义支路功率比例，非总干涉强度中的固定 30%；delta 每图均匀采样 [-pi,pi]，保留干涉。camera/combined eta∈[.15,.35]，stress=.30。保留零入射场，不会凭空补光。phase dropout 8×8 块旁路相位 m→1，并非抹掉振幅。CCD 模型为强度增益、背景偏置和强度相关读出/散粒代理噪声；噪声单位为有界场强度单位，不是标定电子数。k-space 为平滑幅度衰减与离焦/像散相位响应。

600 steps、batch=4、phase LR=2e-5、其余 NN LR=2e-6；前 300 步扰动强度由 .2 增至 1。每步干净与扰动双前向，原权重为冻结教师，损失为 `.5*(MSE+.1L1)clean + .5*(MSE+.1L1)noisy + .1MSE(clean,teacher) + .05MSE(noisy,teacher)`。没有新加 GAN；本轮着重通道适配和保住干净性能。

```bash
export CUDA_VISIBLE_DEVICES=GPU-1b963983-7909-af6e-0528-f0f0661ab549
export OMP_NUM_THREADS=4
AS=/DATA/DATA1/guest3/t12_assets
TASK=LightGenV2/tasks/t12_text_to_image
SOURCE=/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t12_text_to_image/runs/simulation/bounded_tanh_clean_recovery_20260927/best.pt
$AS/venv/bin/python -m LightGenV2.tasks.t12_text_to_image.lab_shs8um.train_channel_robust --checkpoint $SOURCE --assets $AS --output $TASK/runs/simulation/20260927_channel_combined --profile combined --steps 600 --val-samples 96 --batch-size 4
# camera 对照更换 --profile camera，必须使用新的 output，禁止覆盖。
$AS/venv/bin/python -m LightGenV2.tasks.t12_text_to_image.lab_shs8um.train_channel_robust --checkpoint $TASK/runs/simulation/20260927_channel_combined/best_checkpoint.pt --assets $AS --output $TASK/runs/simulation/20260927_channel_full_val_combined --evaluate-only --split val --batch-size 4
$AS/venv/bin/python -m LightGenV2.tasks.t12_text_to_image.lab_shs8um.train_channel_robust --checkpoint $TASK/runs/simulation/20260927_channel_combined/best_checkpoint.pt --assets $AS --output $TASK/runs/simulation/20260927_channel_full_test_combined --evaluate-only --split test --batch-size 4
$AS/venv/bin/python -m LightGenV2.tasks.t12_text_to_image.build_lab_package --checkpoint $TASK/runs/simulation/20260927_channel_combined/best_checkpoint.pt --contract /DATA/DATA1/guest3/t12_channel_base_contract.json --output $TASK/releases/20260927_channel_combined.zip
```

运行目录保存实际命令、commit、环境、数据 SHA、指标及逐图 JSON。固定 seed=1042+实际数据索引；匹配原实测导出，不因 batch 重置。扰动评估各权重采用相同 batch=4/种子。PSNR 为逐图 log 后平均，不能拿整体 MSE 直接换算并当成同一指标；SSIM 使用 RGB 11×11 Gaussian sigma=1.5、valid、[0,1]，与既有实测相同。

部署 ZIP 由干净 Git HEAD 的 `git archive` 构建，含源代码、small.pt、contract 与逐文件 SHA manifest。部署不启用训练扰动通道。不得覆盖旧 `candidate_bounded`，必须放新目录，重新采集六阶段 CCD，不得混用旧权重 CCD。Windows 长路径 ZIP 用支持长路径的解压工具，或 Python zipfile 配合 `\\?\` 前缀。用户要求本任务只训练、离线验证和交付；不操作光路，新候选实测由另一 AI 完成，不能提前声称改善 23.1658 dB 的实测。

## 2026-09-27 强扰动补训：17M 候选（未实测）

训练代码 commit `4065cbfa2`；TRAIN 训练，固定 VAL96 按 clean guard 与 stress/severe 平均 MSE 选模，1000 steps 中选择 step800，并非最后一步。权重 SHA256 `5496204d6a546df5dddef80fef00bc3f4e45479b02f7fd91a310d448bc6c94ed`。计入参数 17,026,642，纯相位参数仍 958,728；冻结词嵌入仍按既有约定另外列出，不计入 20M 预算。保留 9,958,098 参数、SHA `3f82ab04…` 的保守备选，不覆盖此前发布版。

新增部分不是新光路：文字条件160维经可训练 Linear 生成224×14×14低分辨率空间特征，双线性放大至 decoder 入口，以 `.5*detached_RMS*tanh(prior)` 加到两级视觉光电融合之后。新增7,068,544参数，保留两层 Qwen-style Transformer、语言与视觉各 router/expert/global 六阶段光路、478×478有效 mask、并行电子残差；没有输入 GT mask、商品编号或目标图片。它增强监督任务的形状先验，不代表开放域生成，也不证明光本身独立恢复了所有形状。没有采用额外14.7M卷积细化试验或CCD分位数校正。

幅度编码完全不变：保零 `tanh(abs(E)/.5)` 并保留相位；BMP仅 round(255*a)。30%泄漏仍指相干叠加之前名义**功率**比例：`P(a*(sqrt(1-eta)*m + sqrt(eta)*exp(i*delta)))`。不是给图像加常数，也不是最终干涉强度固定30%。新 severe 功率 eta∈[.15,.60]、错位±3逻辑像素、k-space幅/相扰动、低频相位误差.22rad、5%块相位旁路、CCD增益/偏置/噪声/空间不均匀；extreme eta∈[.30,.70]、错位±4、相位误差.35rad、10%旁路，仅用于留出压力评估，未用于训练。CCD参数是有界仿真强度代理，不能视为实测标定。

训练组合 combined/strong/severe 轮换，前半程强度由.2升至1；phase LR3e-5、既有电子LR2e-6、新空间先验LR5e-4、alpha LR1e-4，alpha界限[.35,.75]。干净损失权重.8、原3901e4fb教师干净anchor1、扰动anchor.05、融合特征一致性.01、router KL .01、圆周相位平滑2e-5、物体联合区域损失.4、既有source_gate监督.005。没有新加GAN。真实透明掩码仅用于TRAIN损失/评估ROI，不作为推理条件。全局PSNR容易被背景主导，必须同时查看换目标/联合编辑的ROI指标和对照图，仍存在轮廓残影及纹理模糊。

复现训练命令见 `handoffs/t12_channel_severe_20260927/spatial_protocol.json` 的完整 argv；服务器运行目录 `runs/simulation/20260927_channel_severe_spatial`。干净及压力评估均使用固定2304指令、256×256、batch4、相同输入/扰动seed。封装源码新增可选项按 checkpoint 标志重建，旧权重默认不开启。完整训练、逐图指标及六阶段幅度/强度统计均交付；真实光路由用户安排另一AI验证，本任务不采集。

初轮强补训因clean guard失败停止；仅区域损失的600step试验同样未通过，不部署。9.96M干净anchor恢复成功但换目标残影明显；14.7M局部卷积细化改善平均指标而形状提升有限；因此才增加17M文字空间先验。不得将这些试验都称为成功候选。所有历史日志保留，未清理其他AI文件或进程。

完整VAL2304，同RTX4090、batch4与相同seed：

| 通道 | 上一13cf候选 PSNR/SSIM | 17M候选 PSNR/SSIM |
|---|---|---|
| clean |33.5685/.919405|34.1925/.924963|
| stress |25.9005/.854884|27.7766/.877212|
| severe |25.0511/.845971|27.0017/.870978|
| extreme |24.3912/.838293|26.3708/.864833|

新候选severe物体union ROI PSNR：仅换目标25.9604，联合编辑25.2654；每模式768条。severe仍比clean整体低7.1908dB，不意味着实测差距已解决。完整VAL干净vision专家top2选择率为77.86%/69.40%/30.60%/22.14%（合计200%，每图两个），language为100%/100%/0/0。vision top2相对clean切换率stress27.17%、severe34.77%、extreme38.45%；历史13cf为46.01%/55.51%/66.32%。实际alpha language .4730375/.4617593、vision .4615397/.4667621；为RMS融合特征比例，不是功率或参数比例。

选模冻结后的独立TEST2304（同RTX4090，未用于选择或修改）：13cf→17M，clean 33.3754/.918042→34.0190/.923624；stress25.8466/.853205→27.7058/.875446；severe24.8471/.842593→26.8296/.868111；extreme24.3024/.836348→26.2807/.862871。各通道MSE降低14.82%/30.92%/31.04%/31.50%。干净与扰动全部完整逐图JSON已下载，未做真实光路验证。

最终封装源码commit `ce185f46993460d8ae5813792b5fa5eba40b8f04`，ZIP SHA256 `6736765c330f32dc8b979c458afbb5c7cc6968e75dd3a4be9cc458894512f055`，文件 `handoffs/t12_channel_severe_20260927/candidate_spatial_17m_final.zip`、`small_spatial_17m.pt`。22项单测通过。训练、评估均结束：GPU2=13MiB、GPU4=15MiB，没有本任务残留GPU进程；未打断其他AI任务。后续文档提交不修改此包中源码或模型。

## 2026-09-27 语言专家均衡与extreme补训

结果分开报告：图像鲁棒性改善，语言实际top2均衡**失败**。部署备选SHA `5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac`，17,026,642参数。主干、tokenizer、478有效mask、六阶段、幅度/BMP编码均未改变；部署保留standardized_region_energy读出。纯相位958,728；词嵌入311,164,928仍按约定单列。

尝试run：language_balance_mild/strong（各1000步）、coverage800、semantic1000、input600、logscore600、log_mask800，前缀均为20260927_，全部保存实际命令/Git/环境/数据SHA、history、best/last。mild/strong从5496204d起始，coverage与semantic从strong最后一步分别起始；input从coverage最后一步、logscore从input、log_mask从logscore。负载损失仅训练，计算四专家batch平均probability与detach硬top2负载的内积；逐渐加权，不强制每条文字均匀，也不随机覆盖推理路由。

semantic/input/log试验额外TRAIN语义路由prior：background pair(0,1)、object(2,3)、joint(0,2)，lamp/table/pillow分别循环偏移0/1/2；soft target选中.45、未选.05。九种category×mode总体覆盖4/9–5/9。推理没有类别/模式标签输入、没有指定专家、没有商品ID检索。该prior没有改变实际top2的事实必须保留。

原标准化读出、调router/入光编码器学习率的试验均未解锁专家3/4。logscore/log_mask使用既有log_energy_fraction读出，最终VAL96平均probability=.32053/.30049/.20154/.17744，但selection仍100%/100%/0/0，不采用这两份实验权重。新读出选项已测试可序列化、旧权重兼容、参数量不变；默认仍旧读出，未修改实验室部署。

图像质量备选input600为**人工完整VAL质量选模**（不满足均衡要求）：trainer的best仍step0，故明确使用last而不是宣传best成功。input600的全VAL比coverage、semantic有更好的severe/extreme质量，干净比5496提高.2390dB/.003546，物体区域也改善；选择发生在TEST之前。input训练源码 `eff6c4552`；完整protocol保存所有参数。train profiles combined/severe/extreme/extreme，极强扰动已用于TRAIN，不再是未见通道；VAL/TEST数据独立。教师5496，phaseLR1e-4、语言routerLR.001、既有入光adapter/norm LR5e-4、其余电子3e-6、空间先验5e-5、alpha1e-4；负载.01、语义prior.1、feature.01、router KL.005、phase TV5e-5、clean.7、anchor1/noisyanchor.05、region.5、source_gate.005。没有增加GAN或参数。

完整VAL2304，同RTX4090/batch4/256×256/固定seed：

| 通道 | 前5496候选 PSNR/SSIM | 新5b4f质量备选 PSNR/SSIM |
|---|---|---|
| clean |34.1925/.924963|34.4315/.928509|
| stress |27.7766/.877212|29.3008/.892036|
| severe |27.0017/.870978|28.4256/.885481|
| extreme |26.3708/.864833|27.7628/.879783|

VAL extreme物体union ROI PSNR：object25.6819→26.959、joint25.1368→26.240；PSNR第5百分位19.912→20.961。背景之外的物体区域也改善，但轮廓残影/细纹模糊仍存在。实际alpha language .464571/.449866、vision .444871/.451167，下限.35。完整VAL语言selected100%/100%/0/0；vision73.57%/75.82%/24.18%/26.43%。vision top2切换率stress28.26%、severe34.68%、extreme40.36%，并非所有路由稳定指标都提高。

`handoffs/t12_language_balance_20260927`保存各run配置与结果、逐图指标及固定104样本对照图。25项单测通过，含codec零保留、量化、语义prior、负载梯度、分支几何、readout序列化/旧权重预算兼容。没有实测，仍需另一AI使用新的完整六阶段CCD验证，不能复用旧权重CCD。

固定权重后独立TEST2304（同RTX4090、未用于选择/修改）：5496→5b4f，clean34.0190/.923624→34.2739/.927115；stress27.7058/.875446→29.1930/.890222；severe26.8296/.868111→28.2985/.883245；extreme26.2807/.862871→27.7284/.878283。MSE降低8.02%/30.43%/28.64%/28.30%。extreme ROI PSNR新值object27.091、joint26.278；语言仍100%/100%/0/0，不能宣称均衡或真实光路改善。

交付small_robust_17m.pt与candidate_robust_17m.zip，ZIP源码commit `f15e46a6660731d8d73d4438ceb227545566f220`、SHA256 `1860f1ff02f62e0417485bb33caf1a6657dc570e6fe70754d91342f603eadec2`；模型SHA前述5b4f，本地复核匹配。全部训练、评估结束，GPU2=13MiB、GPU4=15MiB，无本任务GPU PID；其他AI任务未打断。后续文档提交不修改封装源码或权重。

## 清理记录（此前）

服务器仅删除明确清单内 35 份旧 .pt，4,808,586,899字节，包括旧错误光电模型及不采用的15M实验；保留数据、baseline、历史图/指标、sealed参考。服务器删除不可直接撤销，清单含SHA256。本地旧权重按明确目录送回收站；本目录两份正式权重不删除。

共享根工作树存在其他AI修改与分叉，未 reset、未强推、未全量暂存。独立整合分支是本任务唯一新的代码入口；不可将共享根的旧HEAD当作本轮训练代码。
