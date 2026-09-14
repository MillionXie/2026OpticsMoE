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
