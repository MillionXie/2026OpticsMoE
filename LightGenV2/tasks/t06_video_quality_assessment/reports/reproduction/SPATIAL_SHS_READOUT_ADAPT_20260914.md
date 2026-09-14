# Spatial SHS：仅最后电子读出头适配

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
