# T03 显著性分析（SALICON）

本目录固定比较三组系统，三组共享同一数据清单、输入尺寸和指标实现：

1. `main_dc20`：光学 Router、4 专家、Top-2、Vision 两层光电网络、同尺度凸融合、20%–30% 相干零级分量与位置/读出噪声。
2. `d2nn_dc20`：不含 Router 的普通两层 D2NN；两张 `224×224` 相位等于主方法每样本实际激活的两位专家参数量。
3. 冻结 `Qwen3-VL-Embedding-2B` 视觉主干 + 训练过的显著性头 baseline。历史CC约0.88105；现增加不限定显卡的性能复现入口，速度/功耗仍只在5090D测。`qwen_pending`仅保留旧待测合同入口。

输出是 `224×224` 连续显著性概率图。主指标为 `CC`（越大越好），并同时报告 KLD、SIM、NSS、AUC-Judd 和 MAE。

注意：20%–30%未调制分量是训练扰动；现有标准测试在eval模式关闭随机光学扰动。
本页CC及0.87优化目标是理想光学仿真口径，不代表同分数已在漏光实测中验证。
具体边界见[复现说明](reports/reproduction/README.md)。

2026-09-08新增：[同规格读出头、alpha无下限/≥0.4、三阶段训练对照与命令](reports/reproduction/ALPHA_AND_HEAD_COMPARISON.md)。
100epoch三阶段训练已完成：无下限CC=0.85212668，alpha≥0.4 CC=0.85134034，
新同规格读出头冻结Qwen CC=0.88968476；均为5000张public-test选定权重复评。
后续保持结构不变的四组续训、学习率覆盖问题及完整操作命令见
[训练方法优化](reports/reproduction/TRAINING_REFINEMENT.md)。
2026-09-09：上述四组已完成，最佳weakaug CC=0.85468765，仍低于新Qwen的0.88968476。
该轮EMA/蒸馏100epoch已完成，最优`moe_alpha40_generalize_kd060_seed42`的best epoch=5，
CC=0.85812016，alpha=0.434135/0.441446；仍低于同规格头Qwen的0.88968476。
该次续训关闭增强，但初始化权重曾经历弱增强训练。
接下来四组30epoch从此best共同出发：低学习率弱持续蒸馏对照、同步水平翻转、
电子深度卷积3→5、前5epoch冻结电子主体的分阶段微调。协议与完整命令见同一份
[训练方法优化](reports/reproduction/TRAINING_REFINEMENT.md)末尾。
5×5版本只增加6144个电子参数，以`_ek5`区分权重合同；不改变光Router Top2、
alpha≥0.4、20%–30%零级分量、两级融合、478有效ROI或光学传播次数。
四组30epoch已完成，更新后最高CC依次0.85745/0.85637/0.85753/0.85759，均未超过初始化0.85812016，
best均保留源权重；相位和新增卷积外圈确实更新。下一轮从alpha限制训练之前的较早源开展
增强、5×5、GRN及组合的五组100epoch对照。
论文依据、参数预算及命令见[早期轻量电子残差试验](reports/reproduction/EARLY_LIGHTWEIGHT_RESIDUAL.md)。
上述早期五组已出现持续平台，当前最好均约0.8500–0.8505，未超越历史best。
现从0.85812016进行三组不改结构的受控精修：小步长保留KD、逐步撤KD、撤KD加强GT CC。
新增可选自动平台调速/早停，仅影响新profile，不修改正在运行的旧试验。
完整规则、指标口径及命令见[平台期受控精修](reports/reproduction/ADAPTIVE_REFINEMENT.md)。
三组受控精修均在22轮自动停止，没有超过0.85812016。当前转向现有电子残差内部的
展开空间深度卷积：新增6912参数，不加分支/attention，不改光路或解码头。
三组设计及命令见[电子残差空间FFN](reports/reproduction/SPATIAL_FFN_RESIDUAL.md)。

2026-09-10：空间FFN三组已在34轮早停，均未超越0.85812016，历史best保留。
现以CC≥0.87为优化目标，从较早的0.85468765来源开展同步弱增强/空间FFN/更强KD三组对照。
不重置alpha，不改变光路或读出头；训练后期关闭增强精修。完整配置、边界和命令见
[泛化优化](reports/reproduction/VIEW_REGULARIZATION.md)。目标尚未达成，结果以run完整复评为准。

[训练专用特征提示](reports/reproduction/FEATURE_HINTS.md)的control与普通cosine对照均完成50轮，
都未超过源权重，完整复评保留epoch0 CC=0.85812011；去空间均值版本另行判断。
这些对照不改变推理结构，训练专用投影不部署。13×13现有电子空间卷积的单变量对照
已启动，参数预算、初始功能保持与完整命令见[泛化优化](reports/reproduction/VIEW_REGULARIZATION.md)。

[两参数读出校准诊断](reports/reproduction/READOUT_CALIBRATION.md)已完成：
标准精度配对5000张测试，原0.858120→校准0.858608，增益仅0.000488且KLD略变差。
它是新增2参数的独立诊断，未替换正式部署权重；0.87目标仍未达到。

## 数据协议

- train：SALICON 2015r1 官方 train2014，10,000 张。
- public test：官方 val2014，5,000 张；本项目将它作为可复现测试集。
- validation：无。
- 官方 test：真值不公开，不用于本地指标。
- epoch 1、每 5 epoch、末轮测试；以最高 public-test CC 保存 `best_checkpoint.pt`。
- 正式目录只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。

## 运行

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all
python -m LightGenV2.tasks.t03_saliency.run --profile d2nn_dc20 --phase all
python -m LightGenV2.tasks.t03_saliency.run --profile qwen_pending --phase all
```

大模型待测边界、计时和功耗口径见 [BASELINE_5090D_TODO.md](BASELINE_5090D_TODO.md)。

## 正式单次结果（seed 42）

- 光 Router Top-2：CC 0.8291、KLD 0.1330、SIM 0.8063、NSS 0.9283、
  AUC-Judd 0.7631、MAE 0.0890。
- 参数匹配 D2NN：CC 0.8346、KLD 0.1296、SIM 0.8092、NSS 0.9344、
  AUC-Judd 0.7643、MAE 0.0884。
- 光 Router 四专家选择占比为 23.54% / 26.80% / 23.38% / 26.28%，有效专家数
  3.985/4，无未使用专家。

本次单 seed 下 D2NN 的 CC 高 0.0055，必须作为真实负差距保留。机器可读指标、相位图和
样例见 `reports/dc20_comparison/`。Frozen Qwen的架构、公平性、固定权重复评、从头训练命令与
本轮训练优化配置集中在 [复现说明](reports/reproduction/README.md)。
