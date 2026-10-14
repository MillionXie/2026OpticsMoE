# LGVQ 最新指定实測版：10个四帧定性示例

交付位置：本任务 `runs/hardware/qualitative_measured10_20261014/`。
每个视频一个编号文件夹，包含四帧448×448模型输入图、四帧原图、分数JSON/CSV、
三组bar对照PNG/PDF/SVG。根目录含10例总览、`selected10_scores.csv`、完整558条
`all558_comparison.csv`、筛选规则、原文件SHA和全部交付文件SHA。
本次没有训练、重新采集或改模型，仅整理保留的实测逐视频结果。

## 严格绑定的四份结果

| 数据 | 完整558 SRCC | 原始路径 / SHA256 |
| --- | ---: | --- |
| Temporal Ours六层CCD实测 | .7977138739 | `runs/hardware/temporal_full_20260914/results.json`；`5e32c2aa537133c7189d1090481cf7cda0d5b11aeb60a070bcf8d94065616041` |
| Spatial Ours实测读出适配 | .6380783492 | `runs/hardware/spatial_train2250_20260914/readout_partial_test20/weighted_target5/train2250_plus_test112_holdout446/results.json`；`8720eaac4f34a3e662058738f5d5531b4f3a7449f201a16700f06ac8cd0a33b8` |
| Temporal Qwen B1 | .7663428037 | 仓库 `LightGenV2/reports/20260927_lgvq_temporal_multi_baseline_a100/raw_remote/frozen_batch_20261013/qwen_batch01/predictions.csv`；`f0696d3809537ce696bd6679b4c7d26d98770ea5026432ae13745159d011209f` |
| Spatial Qwen | .6907725734 | 仓库 `LightGenV2/reports/20260914_latest_qwen_baselines_a100_FIRSTBLOCK_FINAL/evidence/T06_spatial/per_video_predictions_and_timing.csv`；`0ad9916b485f6de30cc42fe46ad5a18c0ae53ee45a2530af1a77cebe2b6e6791` |

Ours不是仿真预测；Qwen是实际GPU baseline推理记录，不经CCD。
两任务/Qwen共4份表均558个唯一ID、名单完全一致，逐ID真值误差小于.001 MOS，
本次用SciPy从原分数独立复算SRCC，不使用聊天记忆填分。

Spatial `.6381` 为558混合成绩，其中112条参与部署适配，446条未参与反传且用于
选模，后者SRCC `.6092739512`。这不是558独立测试成绩；精选的10例全部来自446条
留出，不混入112条已训练样本。留出参与选模也不应声称完全独立的泛化估计。
整体Spatial Qwen仍高于Ours，图只展示Ours局部预测更接近MOS的精选案例。

## 筛选与视觉核验

同时要求两任务Ours绝对MOS误差小于Qwen、且均不超过5分。446留出中50条满足。
按两任务误差改善的较小值降序，单生成模型最多3条，选10条；没有修改预测值或真值，
没有逐样本校准、增强原图亮度或对图像做修饰。三组bar统一0–100 MOS，保留坐标原点。
原数值在scores中完整保留，画图仅显示1位小数。3项纯CPU身份/SHA测试通过。
165份交付文件的SHA核验通过，总约68.2MB，含只读下载的10段原视频。

优先主图推荐：01麦田、02薰衣草、04蒲公英路边、06服装设计师、07奶昔。
完整10例已目视检查。05网球视频画面主体缺失、基本只有草地，适合展示生成失败
导致的低质量，而非美观画面；如篇幅有限优先使用上述五例。

10例次序：麦田、薰衣草、墨西哥辣椒碗、蒲公英路边、网球草地、服装设计师、
草莓奶昔、公园湖面、医院医生、坚果水果沙拉。目录编号与总览图Example标签一致。

### 抽帧差异不能隐藏

Ours原前端采用10/36.67/63.33/90%；Qwen四帧采用10/37/63/90%。少数短视频的
整数索引相差1帧；主图是Ours四帧，在scores记录双方索引，有差异则另附
`qwen_sampling/` 四张Qwen真实输入帧。不能声称所有样例的两模型输入帧逐像素相同。
主图37%/63%标签为显示舍入。65%短边中心裁剪、448缩放与原前端相同；保留原帧
供绘图人员选择，不将原帧图冒称精确模型输入。

原视频从guest3只读SFTP下载，来自
`/DATA/DATA1/lixinyue/xyli/data/LGVQ/videos/<sample_id>`，服务器sha256sum与本地SHA一致。
本地实测/原baseline是已保留的结果备份；本轮guest3主目录同路径未找到这四份原表，
不声称重新从其主目录下载或现场重新评估。没连接师姐服务器、没启动实验室设备。

## 复现导出

从仓库根目录运行；先select，再fetch，再render。select拒绝覆盖已有输出，
render仅允许同视频身份的已有导出目录。凭据通过环境变量传入，不落盘、不进Git。

```powershell
$out = 'LightGenV2/tasks/t06_video_quality_assessment/runs/hardware/<新的示例run名>'
python -m LightGenV2.tasks.t06_video_quality_assessment.export_measured_examples --phase select --output $out
# 手动设置LGVQ_SSH_PASSWORD；不要将密码写进Git。
python -m LightGenV2.tasks.t06_video_quality_assessment.export_measured_examples --phase fetch --output $out
python -m LightGenV2.tasks.t06_video_quality_assessment.export_measured_examples --phase render --output $out
```

只同步导出源码/说明，不将视频、输入图片、完整逐样本结果或交付目录上传Git。
