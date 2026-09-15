# Spatial SHS：仅最后电子读出头适配

## 最新：原训练2250 / 测试558，100 epoch完成（2026-09-15）

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
