# T12 小版光路鲁棒性候选交付（2026-09-27）

## 给负责光路测试的 AI

主权重为本目录 `small.pt`，SHA256：
`13cf9a201bce42f58c19a0ff85fd11fe9940e18be036a86adf163320303edc3e`。

完整源码及部署合同包为 `candidate_channel_robust.zip`，ZIP SHA256：
`d362271c9e429d00d4ea182d1e5da057acb2dc9933dbd07fa1be3e813c9a6c3c`。
内含 Git 导出的 source、同一 small.pt、contract.json 和逐文件 manifest。源码 commit `5164fc1ec868c1784dba15fec98be51c9038b78d`，已推送分支 `codex/t12-physical-robust-v2-20260927`。

新建 candidate 目录，**不要覆盖 candidate_bounded**。不要混用旧权重的 CCD；测试新旧权重需各自重新完整采集。Windows 若遇 ZIP 长路径，使用长路径解压工具，或 Python zipfile 的扩展路径前缀 `\\?\`。可对已有只读 assets/datasets 建目录 junction；PYTHONPATH 指向本包 source。

模型结构与旧权重完全相同：9,958,098 参数，原 tokenizer/冻结嵌入及训练过的两层窄 Qwen-style 文字头不变，语言/视觉光电模块、478×478 phase mask 不变。按现行约定冻结词嵌入单列不计预算，本次没有增加部署参数，低于20M预算。

**部署幅度必须保零 tanh(abs(E)/0.5)，保留复场相位，BMP 仅 round(255*a)，绝不再做峰值或批次归一化。** 训练扰动不会在正式推理自动开启。保持原光学合同的几何、相位编码、ROI/CCD翻转和曝光/增益；避免把设备设置变化混入权重对照。

`lab_shs8um.run_layerwise` 保留原完整 TEST 默认入口，并新增 `--split val --max-samples 96` 的固定均匀 VAL 子集。VAL 子集不得使用旧 TEST reuse 文件夹；给 `--reuse` 一个全新目录。先做配对 VAL 新旧权重验证，再决定是否部署/完整 TEST。需要六阶段逐层采集；不可用新权重处理旧的 router/expert/global CCD 并称为新实测。

本任务已按用户要求停止光路工作，**没有启动 T12 实际采集**。新权重光路实测未知；旧权重实测 PSNR=23.1658 / SSIM=.816176 仅为历史参考。

## 训练与选模

原权重 SHA256 `3901e4fb5d4d249ef4472d4de33cc80858c5d7002ff404db2b53a9ff0ff08577`。只用 TRAIN 20736 指令对更新权重；固定96 VAL选checkpoint，完整2304 VAL选候选。选模已在 TEST 前冻结，见 selection.json。TRAIN/VAL数据、环境、命令和源码哈希在 training_combined.json 及 Git 复现文档。

camera 和 combined 各训练600步；主权重为 combined step600。干净/扰动双前向，加冻结原权重教师约束，前300步递增强度；不新增GAN、不增加模型容量。不能依据组合对照断言单个扰动有效。

泄漏的30%明确为干涉前名义支路**功率**比例：`P(a*(sqrt(1-eta)*m+sqrt(eta)*exp(i*delta)))`，delta随机，保留场干涉。不是加30%图像常数，也不是30%振幅。训练eta=[.15,.35]，stress=.30。另含CCD背景偏置、读出/强度相关噪声、错位、平滑k-space扰动、块状相位旁路dropout；这些是假设的通道代理，非由TEST实测拟合的标定值。

## 完整 VAL 结果（2304条）

| 通道 | 原权重 PSNR / SSIM | 新权重 PSNR / SSIM |
|---|---:|---:|
| 干净 |33.6471 / .920208|33.5685 / .919405|
| camera |28.9665 / .881981|29.2047 / .883973|
| combined |27.3636 / .867222|27.7775 / .871593|
| stress |25.4891 / .849413|25.9005 / .854884|

干净下降0.0787dB/SSIM .000803，通过保护阈值 .2dB/.002；camera/combined平均MSE下降约11.23%。收益有限，不代表已经解决实测差距。所有指标同一256×256输入输出、[0,1]逐图MSE/PSNR和RGB Gaussian SSIM口径。PSNR逐图log后平均，不是将整体平均MSE换算。

val_original.json / val_camera.json / val_combined.json 为完整VAL；test_original.json / test_combined.json 为选模后的完整TEST（评估完后交付）；各模型四种通道逐图指标放在对应目录。16项光路/架构测试通过。

## 选模后的完整 TEST（2304条，已完成）

| 通道 | 原权重 PSNR / SSIM | 新权重 PSNR / SSIM |
|---|---:|---:|
| 干净 |33.4286 / .918732|33.3754 / .918042|
| camera |28.7182 / .879574|28.9739 / .881703|
| combined |27.2183 / .864712|27.6322 / .869324|
| stress |25.4729 / .848049|25.8466 / .853205|

干净PSNR下降仅.0531dB；combined提升.4139dB、MSE下降14.56%；stress提升.3738dB、MSE下降15.48%。这里只是合成扰动评估，非新实测。只占用一张服务器GPU，全部结束后GPU4显存回到15MiB，没有遗留训练/评估进程。

TEST 数据清单 SHA256 `331874abbb8d8c7a4ac5ee3d6e030f20611cb5a9e96d3092917b7a5c4d7d5d2f`；冻结词嵌入缓存 SHA256 `e7a855849ef22e470aafdcf8ee583968ca0c8d5d17ff42603b103c49e97502ac`。本轮旧权重 batch=4 干净 TEST PSNR=33.428582 / SSIM=.918731552，与旧 batch/导出结果33.428809/.918734103只有约.00023dB/.0000026差异；新旧模型本轮严格使用相同batch/评估实现。

## 源码与服务器目录

服务器代码：`/DATA/DATA1/guest3/2026OpticsMoE/.worktrees/t12_physical_robust_v2_20260927`。
主权重：其下 `LightGenV2/tasks/t12_text_to_image/runs/simulation/20260927_channel_combined/best_checkpoint.pt`。
复现入口：`LightGenV2/tasks/t12_text_to_image/reports/reproduction/README.md`。
强度配置：`LightGenV2/tasks/t12_text_to_image/configs/optical_channel_robust_v2.json`。
训练源码：`LightGenV2/tasks/t12_text_to_image/lab_shs8um/train_channel_robust.py`。
通道源码：`LightGenV2/tasks/t12_text_to_image/lab_shs8um/robust_channel.py`。

原权重和 camera 次候选仍保留服务器，不删除。不覆盖共享根工作树中其他AI的修改。
