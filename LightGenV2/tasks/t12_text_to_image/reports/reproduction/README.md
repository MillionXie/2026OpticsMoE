# T12 最终版本与复现入口

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

## 清理记录（此前）

服务器仅删除明确清单内 35 份旧 .pt，4,808,586,899字节，包括旧错误光电模型及不采用的15M实验；保留数据、baseline、历史图/指标、sealed参考。服务器删除不可直接撤销，清单含SHA256。本地旧权重按明确目录送回收站；本目录两份正式权重不删除。

共享根工作树存在其他AI修改与分叉，未 reset、未强推、未全量暂存。独立整合分支是本任务唯一新的代码入口；不可将共享根的旧HEAD当作本轮训练代码。
