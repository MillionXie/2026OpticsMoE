# Spatial SHS：仅最后电子读出头适配

## 最新追加：112条原test参与部署适配；446条留出0.609274，未达到0.65

本节是用户明确授权的**部分原test适配协议**，不能替代下节“不掺test”的完整558条结果。
从原558条按固定seed=20260914、NumPy default_rng一次性随机取ceil(558×0.2)=112条，
加入2250原train，合计2362条反传。剩余446条每epoch用于选模，未反传；因已有选模/调参历史，也不称未触碰独立test。
名单不按预测误差筛选，三个试验使用完全相同的split.json；未改变标签、缓存、相机PNG、相位、光学router或前置电子参数。
每组100epoch，仍只更新967458个原readout参数，以446条SRCC选raw/EMA best（RMSE同分裁决），保存best/last。

| 方案 | 最佳轮/状态 | 446条留出 SRCC | 112条适配 SRCC | 完整558条 SRCC（混合，含112条反传） |
|---|---|---:|---:|---:|
| 上一版不掺test权重，在同样名单复算 | 上一轮32/EMA | 0.6044956961 | 0.6544257870 | 0.6200222298（该权重没有test反传） |
| ranking：普通样本权重 | 11/raw | 0.6081725672 | 0.6691652412 | 0.6261592625 |
| regression：普通样本权重 | 5/raw | 0.6062175344 | 0.6792065211 | 0.6274032593 |
| weighted_target5：适配样本5倍权重 | **13/raw** | **0.6092739512** | 0.7227244286 | **0.6380783492** |

选择的是446条留出最佳，不是按完整558条混合分数最高的epoch选模型。
旧0.620022与新0.609274分母不同，不能直接解释为下降；同446条实际增加0.0047782551，仍很有限。
112条已见样本的提升明显更大；完整558条混合分数增加不能证明对未见视频同等泛化。
**选定的第13轮权重，无论446条留出还是完整558条混合，均未达到0.65。** 后期完整558条混合曲线超过0.65，但446条留出回落；不能把已训练样本带来的混合分数上涨视为泛化提升，因此仍保留第13轮。

weighted_target5指标详情：

| 指标 | 446留出 | 112适配（已反传） | 558混合 |
|---|---:|---:|---:|
| SRCC | 0.6092739512131374 | 0.7227244286257468 | 0.6380783491502683 |
| PLCC | 0.6463626211077566 | 0.7259104848636493 | 0.6620000848991943 |
| RMSE | 8.950145057119219 | 7.755330712136791 | 8.72346261088386 |
| MAE | 7.156285461288931 | 6.3101158652986795 | 6.9864449689037915 |

所有组batch128、EMA0.98、seed20260914、MOS十分位训练排列、AdamW weight_decay1e-4、cosine最低LR=初始×0.1、grad clip1。
ranking：lr1e-5、L2-SP0.03、reg/rank/corr=0.5/1/0.5；regression：lr3e-5、L2-SP0.03、1/0.2/0.1。
weighted_target5：lr1e-5、L2-SP0.1、1/0.5/0.3；只对112条的训练损失给5倍权重，原train权重1。
回归加权平均、成对排序按样本权重乘积、相关性用加权均值/协方差；评估完全不加权。
每条训练视频仍每epoch恰好一次；112条占训练样本约4.74%，占未归一化训练权重560/(2250+560)≈19.93%。
配置为 `configs/spatial_measured_partial_test20.json` 与 `configs/spatial_measured_partial_test20_weighted.json`。

本地唯一入口仍为 `runs/hardware/spatial_train2250_20260914/00_查看这里.md`。
新结果集中在其 `readout_partial_test20/`，最佳权重位于
`weighted_target5/train2250_plus_test112_holdout446/best_checkpoint.pt`，SHA256：
`d787514215f8c5d7777d65ccaea317f13b87c79132d65cfd49a3b50aa0354624`。
原“不掺test”best保留不覆盖。实验电脑在原train会话的 `readout_adaptation/partial_test20_20260915/`，三个候选各只保存best/last。
原始命令与环境在每组launch.json，100轮history、split和2808条逐视频预测完整保留。
`independent_comparison.json`按各身份组独立复算全部指标、检查100轮选模最大值与分组一致性；
`checkpoint_replay_check.json`严格加载磁盘PT，558条最大预测差7.6294e-6 MOS，留出与混合SRCC均一致。
下载核验在 `selected_download_SHA256.json`，可视化 `adaptation_comparison.png` 两侧明确区分446留出与558混合。
非readout冻结SHA仍为 `e855f30c589bca7942da50e8753ffde9d62c7fe29e8ae7ec5f5bdbe6731559e3`。
checkpoint内记录 `test_samples_in_gradient=112`、test_adapt_fraction、split_SHA及实际适配源码SHA；5倍组还记录test_adapt_weight=5。
未覆盖实验室原固定权重，未开启硬件；训练与回放进程完成后已退出。

原权重与两个特征缓存SHA同下节。前两组源码commit `6a518c0576dc8ee2886b7e5130dc3bd0149998c0`，
源码SHA `17d58496e583eac7777c9f0f7eab019ffdcffa5c3836719dda90866d87b42b43`；
加权组源码commit `81fe403e896a4310c077a26e5c33ee8317752609`，源码SHA
`9807d7373d7751a021ad8119c1dffafa5149813eda0e67e34a102246660bb341`。均已push后用builder的readout-only ZIP部署。
加权更新ZIP SHA `68eb09ab56557b7aabbd4acf803dff42fb6fad9633235071e391b13524233865`；
更新时备份触发Windows长路径限制，按前后SHA恢复并改成短备份路径，证据在 `readout_update_weighted/resume_audit.json`。
本地11项纯数据测试通过，Torch测试因本地DLL问题跳过；实验电脑12项全部通过，含加权损失梯度与默认兼容。
环境仍为RTX4060单卡、torch2.8.0+cu126、IEEE FP32，release字段的d89223b7是原模型包而非本次训练代码版本。

加权组实际命令（前两组替换上述配置参数即可；复跑须使用新的output，不覆盖旧结果）：

```powershell
Set-Location E:\code\guest\2026OpticsMoE\LGVQ_Spatial_Train_Lab_SHS_8um
$py = '..\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe'
$fit = 'sessions\train2250_language_verified_20260915\readout_adaptation'
$testCache = '..\LGVQ_Spatial_Lab_SHS_8um\sessions\test558_language_verified_20260915\test_readout_cache.pt'
& $py adapt.py train --cache "$fit\train_readout_cache.pt" --eval-cache $testCache --checkpoint weights\best_checkpoint.pt --output "$fit\partial_test20_20260915\weighted_target5" --device cuda --test-adapt-fraction 0.2 --test-adapt-weight 5 --epochs 100 --lr 0.00001 --batch-size 128 --seed 20260914 --anchor 0.1 --ema 0.98 --scope head --reg-weight 1 --rank-weight 0.5 --corr-weight 0.3 --batch-order mos_stratified
```

## 上一轮：追加四组训练策略，best SRCC 0.620022，未达到0.64（2026-09-15）

沿用下节完全相同的2250原训练/558原测试实测特征和原始checkpoint。所有四组各100epoch，
每epoch在完整558条test上比较raw/EMA、按SRCC选best（RMSE同分裁决）。test不反传，但参与epoch与超参数选择，**不是独立最终测试**。
不重采、不删样本、不改CCD归一化，也不新增网络；只更新原有readout的967458个参数。
数据、原权重、非readout冻结张量SHA与下节一致。RTX4060单卡顺序执行；完成后训练进程已退出。

| 方案 | LR / batch / L2-SP | 损失(reg/rank/corr) | best轮/状态 | test SRCC | PLCC | RMSE |
|---|---|---|---|---:|---:|---:|
| 上一轮 | 1e-5 / 64 / 0.1 | 1 / 0.2 / 0.1 | 19/raw | 0.6172733980 | 0.6452928605 | 8.88392553 |
| strong_anchor | 1e-5 / 64 / 1 | 1 / 0.2 / 0.1 | 41/EMA | 0.6083081391 | 0.6411158990 | 8.84157453 |
| large_batch | 1e-5 / 256 / 0.1 | 1 / 0.2 / 0.1 | 37/raw | 0.6182121456 | 0.6459189361 | 8.86024949 |
| slow_weak_anchor | 3e-6 / 128 / 0.01 | 1 / 0.2 / 0.1 | 90/EMA | 0.6185425359 | 0.6452125124 | 8.82475295 |
| rank_stratified | 1e-5 / 128 / 0.03 | 0.5 / 1 / 0.5 | **32/EMA** | **0.6200222298** | 0.6415857913 | 8.87796508 |

所有组seed=20260914、EMA=0.98、AdamW weight_decay=1e-4、cosine末LR=初始的0.1、grad clip=1。
最后一组新增训练集MOS十个分位组交错的batch排列，每epoch每条训练视频恰好出现一次，保留尾batch；
其他组仍随机打乱。排序/相关性损失并不直接等价于SRCC，实测是否提高仍由完整558条预测决定。
本轮SRCC增量只有0.0027488317，PLCC反而比上一轮略低，不能称为显著提升或达到0.64；旧best保留。
训练曲线显示前期改善、后期平台/回落，不能据此断言模型绝对上限，但不支持单纯继续增加epoch。

结果唯一目录：`runs/hardware/spatial_train2250_20260914/readout_tuning064/`。
各组有launch、100轮history、split、before、results及原始log；实验电脑各组仅保留best/last PT。
本地下载SRCC最高组的best/last（`rank_stratified/original_train2250_test558/`），其他三组保留小型对照记录。
最终best SHA256：`37e87189e59dc7aa74374f4863bfd3bd75f6aa98ff179c58b0328c45b8d9cb0d`。
`independent_comparison.json`独立复算558条指标并检查100轮最大值、划分与冻结身份；
`checkpoint_replay_check.json`重新加载磁盘PT，严格加载state_dict并回放558条head输入，最大预测差1.52587890625e-5 MOS、SRCC完全一致。
`selected_download_SHA256.json`为下载核验；曲线在`training_comparison.png`。未覆盖实验包原始固定SHA权重，不能直接覆盖后绕过身份检查。

排序训练源码commit：`cb2c9aa6341dd4de003683b4789a2d38d268a991`，已push GitHub；
原模型release仍为`d89223b7`，不能把release字段误认作本次适配代码版本。
builder生成readout-only ZIP并逐文件核验后更新实验电脑；manifest在`readout_update/installed.json`。
ZIP SHA256：`991d97c010b3161e10f66cd046118858e4fdde8dc91bf776ad025fd570e9fb53`。
新适配源码SHA256：`59210981114c4a87cc987087ecac496b73a643157639a4247f3250c2118f3326`；前三组的源码SHA256：
`5fa9b372000719eb62c0d381b00cb3de5c20e9a89e84e6a49ac9967626f05b99`。本地及实验电脑各9项测试通过。
全部训练与回放为torch2.8.0+cu126、IEEE FP32（CUDA matmul/cuDNN TF32关闭），无需相机或SLM。

排序组完整复现命令（先安装上述代码版本，已存在output会拒绝覆盖；复跑时改成新output）：

```powershell
Set-Location E:\code\guest\2026OpticsMoE\LGVQ_Spatial_Train_Lab_SHS_8um
$py = '..\ABO_Lab_SHS_8um\.venv_gpu\Scripts\python.exe'
$fit = 'sessions\train2250_language_verified_20260915\readout_adaptation'
$testCache = '..\LGVQ_Spatial_Lab_SHS_8um\sessions\test558_language_verified_20260915\test_readout_cache.pt'
& $py adapt.py train --cache "$fit\train_readout_cache.pt" --eval-cache $testCache --checkpoint weights\best_checkpoint.pt --output "$fit\tuning064_20260915\rank_stratified" --device cuda --epochs 100 --lr 0.00001 --batch-size 128 --seed 20260914 --anchor 0.03 --ema 0.98 --scope head --reg-weight 0.5 --rank-weight 1.0 --corr-weight 0.5 --batch-order mos_stratified
```

下一步候选（尚未执行）：原读出头内部的分阶段解冻、仅训练集特征的轻量扰动增强，分别对照，不增加推理网络。
这些只能作为后续实验方向，不能承诺一定达到0.64；不使用挑选测试样本、测试反传或更改标签来凑指标。

## 上一轮：原训练2250 / 测试558，100 epoch完成（2026-09-15）

正式run：`runs/hardware/spatial_train2250_20260914/readout_final`。
2250条原训练视频反传，558条原test只用于逐epoch选模、不反传；没有新增验证集，明确不是untouched test。
测试数据会话 `test558_language_verified_20260915`：保留视觉三层，三个语言阶段重采并通过采前/采后光场检查及文件审计。
最后一层558/558，采后PCC=0.9909695977。全部采集在11:12完成，硬件已释放。

| 558条test指标 | 微调前（统一FP32） | best（第19轮，raw） |
|---|---:|---:|
| SRCC | 0.5785539874874497 | 0.6172733980275069 |
| PLCC | 0.6152875846288357 | 0.6452928605479216 |
| RMSE | 8.937013012683929 | 8.88392553087357 |
| MAE | 7.207465527305466 | 7.122434592161555 |

SRCC绝对增加0.03871941，但仍未达到仿真约0.6710。训练集best SRCC=0.84512004，不能当作test成绩。
只更新原有readout的967458参数，不加网络；非readout权重SHA核验保持不变。
100 epoch均完成，AdamW lr=1e-5、batch=64、seed=20260914、L2-SP=0.1、EMA=0.98；best是raw分支而非EMA。
损失为SmoothL1+0.2排序+0.1相关性，cosine学习率；完整命令及环境在 `readout_final/launch.json`。
环境：RTX4060，torch2.8.0+cu126。保留best与last两个完整checkpoint及100轮history和逐视频预测。

训练前曾因默认TF32评估和IEEE FP32特征重放不一致而停止（SRCC差0.00015219）。
修复方式是重新用同一FP32设置评估并提取，而非放宽检查阈值或修改CCD；旧结果与失败审计保留在测试会话 `precision_alignment_evidence/`。
任务release commit为`d89223b7`，采集runtime更新和本地协调来源分别记录在run安装清单及阶段report中；FP32恢复命令为哈希记录的run artifact，不改变模型源码。

- 原权重SHA256：`95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b`。
- 新best SHA256：`dbde77832b34a6475198ea00dec9e857aa696e6c1b2bb74057a5a881d57116ba`。
- 训练缓存SHA256：`968e0e5445fb67198ef30c871631877ac458210987c754770f2681630f54955b`。
- 测试缓存SHA256：`82079950bc19c3e9d333beb626c07d06510138471ebcc7ccef3d39ea115726a2`。

本地 `readout_final/independent_result_check.json` 为全部558条逐视频预测的独立SciPy指标复算；
`download_SHA256.json`为下载校验，`original_train2250_test558/split.json`记录训练/测试身份与0条test反传。

## 以下为历史的test自身适配诊断，不与上述2250训练协议混用

运行：`runs/hardware/spatial_readout_adapt_20260914/adaptation`，源码commit `e0cd757c`。
服务器位于 `/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t06_video_quality_assessment/` 下同一run相对目录。
原实测为 `spatial_lang1600_20260914`，558条视频、六层CCD全部回填；没有挑选或删样本。
原checkpoint SHA256 `95e12397ccf8c960fa30ba9dfb400b69d2c2ebd02288ac6a4879870ab592828b`。
数据ZIP SHA256 `43e0c1670253c807c11a504d76f13c169739680cd051703d73bffac83bad3dd0`。

## 范围与方法

仅更新现有 `SpatialLowRankPrunedGridCompactResidualReadout`，967458参数。无新层/分支，
相位、光router、alpha、前面全部电子层和MOS尺度冻结。两版均100epoch、batch32、seed20260914、
AdamW lr1e-4、weight_decay1e-4、cosine最低lr1e-5，SmoothL1+0.2排序+0.1相关性损失，无教师损失。
入口及完整CLI见任务README；每版保存具体划分、逐epoch历史、逐视频预测和best/last完整权重。
两版均有36个读出头状态张量改变；非读出头状态SHA不变：
`e855f30c589bca7942da50e8753ffde9d62c7fe29e8ae7ec5f5bdbe6731559e3`。

## 结果（不能混淆分母与选模方式）

| 版本/评价数据 | 微调前SRCC | best SRCC | best epoch | 含义 |
|---|---:|---:|---:|---|
| 100%适配，全部558条 | 0.5786919523 | 0.9987881201 | 97 | 全部参与训练及选模，重代入，不是测试精度 |
| 80%适配，留出112条 | 0.6177596779 | 0.6271390224 | 1 | 未反向传播，但用于选模，不是独立最终test |
| 80%版权重，全558条 | 0.5786919523 | 0.6330450692 | 1 | 混合训练/留出，仅补充描述 |

80%版有446条训练、112条留出，互不重叠且覆盖全部558条。其留出PLCC为0.6416327548→0.6748212943，
RMSE为9.4048645147→9.4108581851，MAE为7.5481509992→7.6353605986；排序略有改善，绝对误差没有改善。
后续训练的留出SRCC下降，说明单纯继续拟合容易过拟合。全量0.9988不能宣称光路已恢复仿真精度或新视频泛化达到该数值。
原Windows实测SRCC 0.5786364901保持不变；服务器统一关闭TF32后重新回放是0.5786919523。
跨设备回放最大MOS误差0.0164528、SRCC差0.0000554622、RMSE差0.0000342666，通过预设三重容差。
GPU/CPU IEEE FP32对照与逐视频回放差异保存在run中；不把数值差异伪装成微调收益。

## 权重与核验

- full100 best SHA256：`5580593c42017022c09a82752b1e634e574fe0113c063e0c67514447a8f74037`
- split80 best SHA256：`8b3b82ded905b3b54f0300ae9e3477a6bc75a892364e0aef9a1669f403c61c8b`
- 权重在各版本子目录的`best_checkpoint.pt`、`last_checkpoint.pt`，未替换实验台原权重。
- 服务器torch2.6.0+cu124，RTX4090单卡；完成后进程/显存已释放。
- 结果、划分、历史已下载本地，按逐视频数据独立复算三个表格SRCC一致。
- 完整checkpoint/cache SHA、设备和原始命令在`adaptation/launch.json`及每版`results.json`。

这两版均不能称为独立测试结果；后续部署需适配checkpoint身份校验，不能直接覆盖原固定SHA实验包。

## 追加训练手段对照（同一446/112划分）

三组正则化入口`tune_measured_readout.py`（commit `b8a5c022`），参数由
`configs/spatial_hardware_readout_tuning.json`冻结；均从原始checkpoint开始，100epoch，单卡顺序运行。
L2-SP约束相对原始读出参数的平方差之和；EMA只保存为单个模型状态，不增加推理分支。
终端层对照commit `3035597a`，同一缓存/seed/划分，`train --train-fraction .8 --scope terminal --lr .0003 --ema .98 --anchor .1 --batch-size 64 --epochs 100`，只更新原有两处最后评分Linear，402参数、4个状态张量。

| 设置 | 最佳epoch/状态 | 留出112条SRCC | 留出RMSE |
|---|---|---:|---:|
| 未适配原头 | 0 | 0.6177596779 | 9.4048645147 |
| 上一轮常规微调 | 1/raw | 0.6271390224 | 9.4108581851 |
| lr1e-5、L2-SP0.1、EMA0.98、batch64 | 64/EMA | **0.6281555452** | **9.0845031880** |
| lr3e-5、L2-SP1、EMA0.98、batch64 | 6/EMA | 0.6223681992 | 9.2870552934 |
| lr3e-6、L2-SP1、EMA0.95、batch32 | 13/EMA | 0.6223767414 | 9.2776284760 |
| 仅末端402参数 | 3/raw | 0.6186096276 | 9.3514222897 |

最佳为`regularized/low_lr_ema/split80/best_checkpoint.pt`，SHA256
`ddd194175bd9eb637fdd4e2e74abf7dbb7c2d376b878007227368ac14619484f`；完整权重约36.45MB已下载本地并校验。
最佳留出PLCC0.6665443087、MAE7.3611434358；相比上一轮SRCC仅+0.0010165，PLCC略降，不能宣称显著提升。
该模型全558条SRCC0.7089853062包含446条训练样本，不作为测试精度。112条用于epoch和超参数选择，
搜索后的分数也不是独立最终test；每组具体划分已核对完全相同，逐视频SRCC独立复算一致。
全部前置/光学参数保持不变，所有训练进程与GPU显存已释放。未改动原CCD或向硬件部署任何候选。

本地/服务器产物统一仍在`runs/hardware/spatial_readout_adapt_20260914`：三组位于`regularized/`，
402参数对照位于`terminal_only/`；无需浏览其他runs。后续更完整验证应使用原训练集实测CCD做适配，
再用原测试集前向推理，不能把已用于训练的原test视频重新叫独立test。
