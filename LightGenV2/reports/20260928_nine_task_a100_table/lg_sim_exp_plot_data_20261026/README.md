# LightGen-II Sim./Exp. 绘图数据

本目录的九本Excel是给绘图同学使用的选定结果，每本仅含“整体指标”和“逐测试样本”两页。表内只放一组Sim./Exp.，不放历史版本、回放对照、SHA核验或过程说明。原报告、补算来源和版本身份保留在本说明、`plot_data.json`及`source_records/`，不要求绘图同学再次选版。

## 本次采用值

| 文件 | Sim. | Exp. | 已恢复的逐样本 |
| --- | ---: | ---: | --- |
| [01 LGVQ时间](01_LGVQ时间质量_LG_Sim_Exp.xlsx) | .8043868643 | .7977138739 | 各558视频 |
| [02 LGVQ空间](02_LGVQ空间质量_LG_Sim_Exp.xlsx) | .6710079009 | .6380783492 | 各558视频 |
| [03 ABO图搜文](03_ABO图搜文_LG_Sim_Exp.xlsx) | **.8058333333** | **.8029166667** | 各2400查询；Sim为本次冻结补推理 |
| [04 ABO图搜图](04_ABO图搜图_LG_Sim_Exp.xlsx) | .8225 | .81125 | 各800查询 |
| [05 ABO文搜图](05_ABO文搜图_LG_Sim_Exp.xlsx) | .86 | .85 | 各100查询 |
| [06 LSP关键点](06_LSP关键点_LG_Sim_Exp.xlsx) | .7347857143 | .7315714286 | 各1000图、14000关节 |
| [07 SALICON](07_SALICON显著性_LG_Sim_Exp.xlsx) | .8624925082 | .8597739692 | 各5000图 |
| [08 OpenMoji](08_OpenMoji语义交互_LG_Sim_Exp.xlsx) | .8765 | .8655 | 各1000指令 |
| [09 ABO条件图像编辑](09_ABO条件图像编辑_LG_Sim_Exp.xlsx) | 34.2777509774 dB | 31.5528864772 dB | 各2304图 |

Excel共导出27,440条选定版本记录，两个缺项均已补齐。LGVQ空间原Sim的558条逐视频预测已从服务器原运行目录恢复，SRCC/KRCC与原报告完全一致；此前“未恢复”是搜索遗漏，现已纠正。图搜文按用户授权，用指定epoch25 EMA和原编码/评估函数重新推理2400张TEST、100标题，Top-1复现1934/2400=.8058333。其余排名指标采用本次推理结果，不将它们说成找回的历史逐样本。逐样本分组、样本ID及全部导出数值均经过回读核对。

## 图搜文采用依据

用户在2026-10-26确认：若`.8029`基于`.8058`同一模型部署，即采用这次完整实拍作为论文Exp。身份检查满足该条件：

- Sim来自原`final_report.json`及`FINAL_VERIFICATION.json`：epoch25 EMA，1934/2400=.8058333333333333。
- Exp来自本地原完整实拍报告`archive/legacy_lab_projects/ABO_Lab_SHS_8um/results/20260913_400us240ms/manual_language_global_20260913_232642/metrics.json`：原汇总和2400条预测均为1927/2400=.8029166666666666。
- 两者checkpoint SHA256完全一致：`a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af`。未在这次执行中发现额外读出适配的证据。
- 本地实拍原报告SHA256：`9bef994bbb3cc4018d68bcd0393a056e4c1dd725d52aed16e282353e64b47e8e`。其2400条首正例排名/Top-1与找到的实验室embedding逐条一致。
- “保住光路性能-abo”和“保住光路性能-实验室性能测试”已独立核对原资料。二者确认上述仿真身份及旧实拍计数；在用户本次指定采用之前，均未查到另一个已明确最终采用的`.8019`有效结果。

`.8019`不再用于绘图。同PT的1931命中A100仿真回放也不进入这组绘图数据。该回放和原旧汇总仍完整存档，不修改或删除。

### 2026-10-26 图搜文冻结补推理

服务器正式run为`LightGenV2/tasks/t08_abo_image_text_retrieval/runs/simulation/sim_frozen_restore_20261026`，原入口Git `16978e65c`。仅CPU、4线程、固定batch4，调用原`encode_student_samples`、`_encode_titles_student`和`_metrics`，没有训练、算子替换、择优扫描或设备调用，耗时376.18秒。原PT SHA仍为`a2aa9a93…`，学生state推理前后SHA均为`29a9982fcf6bd2a80113604c98106f4e017b019a74cf4369e227c980e29f3ee6`。

本次Hit@1/5/10分别为`.8058333333/.9579166667/.9875`，MRR及单正例MAP为`.8713793829`。Top-1与历史一致，但其余指标不同于旧GPU报告的`.9575/.9804166667/.8716098658`；因此表中Sim各项整体指标都来自本次完整排名，不混搭旧汇总。新2400条排名亦与保存的FP32 embedding独立复算逐条相符。原逐样本文件仍未找回，本次提供的是用户授权的同PT冻结补推理。

本次环境为Torch `2.6.0+cu124`、Transformers `4.57.3`、CPU；历史包为Torch `2.8.0+cu128`、RTX5090。未升级环境或声称已证明差异仅由某一设备/库产生。原run报告中的`historical_summary_reproduced=true`实际只检查1934次Top-1命中，不能理解为所有排名指标复现；后续源码已把字段名明确为`historical_top1_reproduced`，原收据不改写。

原报告、identity、2400条CSV、FP32 embedding、manifest、状态和日志均已逐SHA下载至`source_records/server/`。报告SHA为`5424375767ad381b16147a6289f47ec12d458fe35be35b71028c7032a0d69830`；排名CSV SHA为`1d13c88b1776ccdd0954744d0db07af10a1b5d56b65af665bb0cfa8ca76a2b55`。本次不是速度/能耗评估，也不改动任何正式成本数据。

这次选择的是上述已完成实拍批次，不是随后Vision Expert重采的结果。原`production_qualified=false`、父会话`old_metrics_not_current=true`及下游重采记录原样保留。用户指定采用该批次不等于已完成后续重采、相位LUT认证或解除原质量限制；不能在论文中据此额外声称设备资格已认证。

## 其余版本与指标口径

- LGVQ时间：同PT及同558视频ID/MOS；SRCC/KRCC/PLCC是整体指标，逐视频保存MOS、预测和误差。
- LGVQ空间：Sim采用原`reports/paper_results/spatial_readout_1m_srcc067/result.json`中`.6710079009`及同轮其余指标。原`runs/simulation/spatial_readout_1m_srcc067/test_predictions_optical_on.csv`已恢复558条，SHA `9a9545550ee5fd7a1bb36a7722e2f8c2d53e1a33e73454d6681ca24c0478190f`；原评估日志绑定PT `95e12397…`。独立复算SRCC/KRCC完全一致，PLCC浮点末位差2e−16，MAE/RMSE的float32/64归约差小于3e−7。不再以发布包`.6710968960`复评代替。Exp仍为用户所选`.6380783492`，其558条中112条旧TEST参与适配、446条用于选择，保留开发口径。
- ABO图搜图：保留用户指定`.8225/.81125`。Exp含rank72 RGB旁路，结构并非与该Sim完全一致；不静默替换成其另一个Sim值。正确性按精确SKU，8个正例完整AP与AP@10分开计算，绘图表主要指标用完整MAP。
- ABO文搜图：主体`cc977b83…`与TRAIN600/VAL200选择的epoch9读出`89e25360…`；100查询、2400候选、每查询24正例。Sim AP由当次保存FP16 embedding补齐，完整MAP=.7499900524，原FP32汇总=.7499857597另存档，不混入绘图表。
- LSP：Sim旧staged e50 `495b9c2c…`；Exp为有界振幅底座e15 `b525e613…`加TRAIN100原读出头适配last30 `58236644…`。用户已在LSP对话确认`.7316`。PCK双侧分组按有效关节数加权，Overall含头颈共14关节。
- SALICON：同PT `036bc8ca…`。原逐图CC保留；其余五指标由原输入缓存和原15000帧CCD固定CPU重放补齐，表内整体值与补算逐图均值对应。CPU核验CC与原CC的设备浮点差异另存，不在绘图表重复一组。
- OpenMoji：同e45 PT `03cb861c…`及TEST manifest SHA。Exp原6000张CCD固定CPU重放，1000条网格、掩膜及八项指标全部补齐，整体与四操作指标同原报告。Accuracy指changed-cell accuracy。
- ABO编辑：Sim `5b4f…`，Exp适配解码器`eeec…`；2304个sample/source/mode逐条一致。原生256×256 RGB float[0,1]逐图算PSNR后平均，非显示PNG重新量化指标。二者不是同完整PT只换光路的纯退化。

## 原始证据、补算与保护

`plot_data.json`同时保存完整来源清单、SHA、未导入Excel的参考记录及明确的`plot_view`选定数据投影。`version_audit.csv`用于留档，不是给画图同学重新选版的表。

原始归档共30,398条，其中2,958条为非指定轮次参考，已从Excel移出：旧图搜文Sim A100回放2400条、LGVQ空间Sim发布包复评558条。这些数据保留在原文件与JSON，不删除、不冒充指定轮次的预测。

固定CCD补算调用原runtime，CPU/eval/无梯度，不训练、不调用相机/SLM/SDK、不修改远端源码或数据。OpenMoji使用已发布源码`d202c6987`，SALICON使用`6c07ced42`，完整输入/PT/逐帧SHA及模型state前后SHA均留存。

补算原件在`source_records/derived/`。OpenMoji原请求曝光20000µs、6000份收据实际回读20002µs，gain1/wait240ms分别记录，未缩放图像来改曝光。SALICON原CC主值不覆盖，CPU重放均值与原整体差异在预设3e-5核验容差内；逐图CC最大差异另有原审计。

可恢复快照在`source_records/`，包含`initial_workbook_export_20261026.zip`、`pre_completion_workbook_export_20261026.zip`、`pre_i2t_identity_correction_20261026.zip`、`pre_plot_only_i2t_20261026.zip`和`pre_sim_prediction_recovery_20261026.zip`。它们是相应修改前的完整快照，并非新的测量版本。

导出后独立ZIP/XML回读每个数值，另外核验主值与选版、原特征排名、AP定义和样本身份；LGVQ三项相关系数亦由独立纯Python实现复核。所有选定Sim./Exp.均有完整逐样本数据；验证报告中的`provenance`保留原始记录或冻结补推理的区别，不能将完整补算称为恢复历史文件。运行数据、PT、CCD、ZIP和Excel不进Git；外部数据上传仍暂停。源码仅按精确路径提交main，不动其他窗口暂存区。
