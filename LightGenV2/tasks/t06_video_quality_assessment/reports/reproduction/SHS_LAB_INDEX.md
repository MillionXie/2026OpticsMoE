# 三项实验室采集：从这里查看

整理日期：2026-09-15。这里是结果位置索引；实时进度以对应JSON为准，历史log不是当前状态。

| 任务 | 本地入口（相对仓库根目录） | 正式数据会话 |
|---|---|---|
| 时间质量 | `LightGenV2/tasks/t06_video_quality_assessment/runs/hardware/temporal_full_20260914/00_查看这里.md` | `LGVQ_Temporal_Lab_SHS_8um/sessions/temporal_full_20260914` |
| 空间测试558 | `LightGenV2/tasks/t06_video_quality_assessment/runs/hardware/spatial_full_20260914/00_查看这里.md` | 新：`LGVQ_Spatial_Lab_SHS_8um/sessions/test558_language_verified_20260915` |
| 空间训练2250及读出微调 | `LightGenV2/tasks/t06_video_quality_assessment/runs/hardware/spatial_train2250_20260914/00_查看这里.md` | `LGVQ_Spatial_Train_Lab_SHS_8um/sessions/train2250_language_verified_20260915` |
| 显著性 | `LightGenV2/tasks/t03_saliency/runs/hardware/salicon_08625_shs_20260914/00_查看这里.md` | `SALICON_Lab_SHS_8um/sessions/full01` |

师弟电脑上述工程共同根目录：`E:\code\guest\2026OpticsMoE`。

## 只看哪些文件

- 时间：`results.json`，SRCC **0.7977138739203681**，PLCC **0.8091420695614706**；558条测试视频、210幅CCD。本地 `measured_data/ccd/` 可直接看。
- 空间训练：`train2250_six_stage_results.json`，训练集SRCC **0.7483952940287332**，PLCC **0.7760516566063294**。这不是测试集结果，也不是微调后的结果。
- 空间测试修复：`test558_repair_progress.json`；完成后才有 `repaired_results.json`。原 `results.json` 对应旧采集，语言相位存在异常疑点，不能用它评价新微调模型。
- 空间微调：训练run的 `readout_progress.json`、`readout_finetune.log`；完成后 `readout_final/original_train2250_test558/results.json`、`best_checkpoint.pt`、`last_checkpoint.pt`。
- 显著性：`full01_retry01/results.json` 是5000图全量结果，CC **0.8597739692151547**，对应仿真CC **0.8624925081777596**；`pilot01`不是论文成绩。

## 微调和硬件边界

只训练空间模型原有 `readout.*`；不改相位、路由、电子主体或新增网络。2250原训练视频反传，558原测试视频周期选模（明确不是未参与选择的独立test）。新测试会话只继承已核验的视觉三层，语言三层逐层重新生成输入、采集，禁止拼接旧语言CCD。

采集前后固定输入光场检查，结束后审计CCD及上游SHA。检查只证明对应采样时刻的一致性，不等于每一帧相位都被连续认证。更换图层由SDK持有；不要同时打开GUI或改变远程显示连接状态。

`ccd/<stage>/*.png` 是ROI透视映射后的478×478固定DN图，并非全传感器原图；未自动逐张min-max或log增强。语言曝光1600µs时模型按配置补偿4倍曝光；显示软件的自动对比度不代表推理数值。

## 清理原则

不移动工程根目录，避免破坏绝对路径。正式CCD、record、manifest、ROI/LUT、相位、输入特征、模型权重、逐样本结果、源commit及关键故障证据保留。

本次先清理空间旧会话中与正式会话字节一致的重复BMP/理论图；逐文件SHA核对后删除旧副本，保留副本位置记入 `cleanup_deleted_duplicates_20260915.json`，可按表复制恢复。当前正式会话与任何CCD不在删除范围。不同内容或身份未确认的文件不按“旧”直接删除。

时间、显著性的正式采集也不因任务完成而删除。模型/中间张量不提交Git；源码及本索引通过Git同步，数据保留本机与实验电脑的身份清单。
