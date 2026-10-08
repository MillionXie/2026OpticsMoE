# T12 extreme 鲁棒性改善候选（语言均衡未解决）

本次交付不是“专家已均衡”版本。实际语言top2仍固定专家1/2。经过负载损失、TRAIN语义prior、router/现有入光编码器独立学习率及log能量读出对照，最后一组概率平均.3205/.3005/.2015/.1774，但硬选择仍100%/100%/0/0。没有用随机/标签强制路由来制造均衡，不采用log实验权重。

## 使用哪一份

- `small_robust_17m.pt`：图像鲁棒性改善候选，17,026,642计入参数；SHA256 `5b4f9a37f19ce95cf23e4b874badc8e88955559553d47929fb80cd3cbf527cac`。
- `candidate_robust_17m.zip`：配套部署包，内部 `assets/small.pt` 为上述权重，源码commit `f15e46a6660731d8d73d4438ceb227545566f220`。
- ZIP SHA256 `1860f1ff02f62e0417485bb33caf1a6657dc570e6fe70754d91342f603eadec2`；完整文件清单见manifest。
- 原5496204d候选仍在 `../t12_channel_severe_20260927`，未覆盖；原实测3901及9.96M备选同样保留。

**权重来自 `20260927_language_balance_input/last_checkpoint.pt`、step600，不是best_checkpoint.pt。**该run未达到均衡守卫，自动best为step0；这里仅按完整VAL的图像质量/干净保护人工选取last作为质量备选，并明确均衡失败。不能把selected_step0宣传成均衡成功，也不能把实验性log读出包混用。

## 完整VAL2304，同RTX4090、256×256、batch4与固定种子

| 通道 | 前5496候选 PSNR/SSIM | 新5b4f质量备选 PSNR/SSIM |
|---|---|---|
| clean |34.1925/.924963|34.4315/.928509|
| stress |27.7766/.877212|29.3008/.892036|
| severe |27.0017/.870978|28.4256/.885481|
| extreme |26.3708/.864833|27.7628/.879783|

物体union ROI extreme PSNR：换目标25.6819→26.959，联合编辑25.1368→26.240；逐图PSNR第5百分位19.912→20.961。改善不是只有背景，但轮廓残影与细纹模糊仍在，视觉提升有限，不能保证实测达到这些合成结果。

## 独立TEST2304（固定权重后，未用于选择/改参数）

| 通道 | 前5496候选 PSNR/SSIM | 新5b4f质量备选 PSNR/SSIM | MSE减少 |
|---|---|---|---|
| clean |34.0190/.923624|34.2739/.927115|8.02%|
| stress |27.7058/.875446|29.1930/.890222|30.43%|
| severe |26.8296/.868111|28.2985/.883245|28.64%|
| extreme |26.2807/.862871|27.7284/.878283|28.30%|

同RTX4090、相同batch4/profile/seed/指标口径。当前extreme物体ROI PSNR：换目标27.091、联合编辑26.278。语言硬路由仍100%/100%/0/0，无均衡成功。所有训练/评估已结束：GPU2=13MiB、GPU4=15MiB，查询无本任务GPU进程；其他任务未触碰。完整模型/ZIP SHA已本地复核。

`input_object_contact.png` 六列：输入、GT、clean、stress、severe、extreme；每类/模式取固定首样本，非挑图。`visual_audit.json`含前版/semantic/input三模型的104条固定VAL对照、六阶段幅度/强度统计、专家概率及alpha。各训练的完整配置/argv/Git/数据SHA/历史保存在training_*；完整VAL/TEST四通道逐图JSON在input_val/input_test。

## 架构与编码

保持两层Qwen-style文字头、语言/视觉各router/top2-expert/global、并行电子残差、原17M文字空间先验和CNN decoder；纯相位958,728，478有效mask/224专家分区/518传播画布不变。冻结311,164,928词嵌入按既有约定单列，不计20M预算。没有新增网络或GAN、没有GT图像/标签作为推理输入。

alpha下限.35；实际language .464571/.449866、vision .444871/.451167，为特征混合比例而非功率。完整VAL语言专家硬选择100%/100%/0/0；视觉73.57%/75.82%/24.18%/26.43%（每图两个，合计200%）。视觉expert切换率stress28.26%、severe34.68%、extreme40.36%；不是每项路由稳定性都改善。

相位调制前保零 `tanh(abs(field)/.5)` 并保留复相位；BMP仅round(255*a)，不重新按峰值/批次归一化。30%泄漏仍指相干叠加之前名义功率比例，使用sqrt(1-eta)/sqrt(eta)复场相干叠加，不是给图像加常数。相同扰动profile参数见protocol.json。

extreme现在参与TRAIN（combined/severe/extreme/extreme轮换），不再称为未见扰动。TRAIN训练、VAL质量选模，TEST只评估已固定权重。训练语义prior只用于训练损失：category×mode定义平衡的目标专家pair，推理不输入这些标签，也不指定专家。详细训练链与失败记录见任务唯一复现报告；本次没有删除历史run/权重。

## 给负责实测的AI

本任务不操作光路。使用新的候选目录及包内源码/合同，先做新旧权重配对VAL，不复用旧CCD、不覆盖candidate_bounded；通过小规模验证再采集完整2304。本次没有实测数字，不能把旧23.1658/.816176算作新权重结果。25项单测已通过，包含模型读出配置序列化与旧权重兼容；代码已推送分支 `codex/t12-physical-robust-v2-20260927`。
