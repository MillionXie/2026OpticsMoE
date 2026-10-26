# LightGen-II Sim./Exp. 绘图数据

本目录的九本Excel是给绘图同学使用的选定结果，每本仅含“整体指标”和“逐测试样本”两页。表内只放一组Sim./Exp.，不放历史版本、回放对照、SHA核验或过程说明。原报告、补算来源和版本身份保留在本说明、`plot_data.json`及`source_records/`，不要求绘图同学再次选版。

## 本次采用值

| 文件 | Sim. | Exp. | 已恢复的逐样本 |
| --- | ---: | ---: | --- |
| [01 LGVQ时间](01_LGVQ时间质量_LG_Sim_Exp.xlsx) | .8043868643 | .7977138739 | 各558视频 |
| [02 LGVQ空间](02_LGVQ空间质量_LG_Sim_Exp.xlsx) | .6710079009 | .6380783492 | Exp558视频；原Sim目前为整体报告 |
| [03 ABO图搜文](03_ABO图搜文_LG_Sim_Exp.xlsx) | **.8058333333** | **.8029166667** | Exp2400查询；原Sim目前为整体报告 |
| [04 ABO图搜图](04_ABO图搜图_LG_Sim_Exp.xlsx) | .8225 | .81125 | 各800查询 |
| [05 ABO文搜图](05_ABO文搜图_LG_Sim_Exp.xlsx) | .86 | .85 | 各100查询 |
| [06 LSP关键点](06_LSP关键点_LG_Sim_Exp.xlsx) | .7347857143 | .7315714286 | 各1000图、14000关节 |
| [07 SALICON](07_SALICON显著性_LG_Sim_Exp.xlsx) | .8624925082 | .8597739692 | 各5000图 |
| [08 OpenMoji](08_OpenMoji语义交互_LG_Sim_Exp.xlsx) | .8765 | .8655 | 各1000指令 |
| [09 ABO条件图像编辑](09_ABO条件图像编辑_LG_Sim_Exp.xlsx) | 34.2777509774 dB | 31.5528864772 dB | 各2304图 |

Excel共导出24,482条选定版本记录。图搜文原Sim的1934命中排名、LGVQ空间原Sim的逐视频预测尚未恢复，因此这两项只提供原整体指标；不使用另一轮回放填充它们。其余逐样本分组、样本ID及全部数值均经过导出回读核对。

## 图搜文采用依据

用户在2026-10-26确认：若`.8029`基于`.8058`同一模型部署，即采用这次完整实拍作为论文Exp。身份检查满足该条件：

- Sim来自原`final_report.json`及`FINAL_VERIFICATION.json`：epoch25 EMA，1934/2400=.8058333333333333。
- Exp来自本地原完整实拍报告`archive/legacy_lab_projects/ABO_Lab_SHS_8um/results/20260913_400us240ms/manual_language_global_20260913_232642/metrics.json`：原汇总和2400条预测均为1927/2400=.8029166666666666。
- 两者checkpoint SHA256完全一致：`a2aa9a93028410dfb780df7d320a9be65d07b6e7ec87ede08b53c4bddcbaf7af`。未在这次执行中发现额外读出适配的证据。
- 本地实拍原报告SHA256：`9bef994bbb3cc4018d68bcd0393a056e4c1dd725d52aed16e282353e64b47e8e`。其2400条首正例排名/Top-1与找到的实验室embedding逐条一致。
- “保住光路性能-abo”和“保住光路性能-实验室性能测试”已独立核对原资料。二者确认上述仿真身份及旧实拍计数；在用户本次指定采用之前，均未查到另一个已明确最终采用的`.8019`有效结果。

`.8019`不再用于绘图。同PT的1931命中A100仿真回放也不进入这组绘图数据。该回放和原旧汇总仍完整存档，不修改或删除。

这次选择的是上述已完成实拍批次，不是随后Vision Expert重采的结果。原`production_qualified=false`、父会话`old_metrics_not_current=true`及下游重采记录原样保留。用户指定采用该批次不等于已完成后续重采、相位LUT认证或解除原质量限制；不能在论文中据此额外声称设备资格已认证。

## 其余版本与指标口径

- LGVQ时间：同PT及同558视频ID/MOS；SRCC/KRCC/PLCC是整体指标，逐视频保存MOS、预测和误差。
- LGVQ空间：Sim采用原`reports/paper_results/spatial_readout_1m_srcc067/result.json`中`.6710079009`及同轮其余指标，不再以发布包`.6710968960`复评代替。Exp仍为用户所选`.6380783492`，其558条中112条旧TEST参与适配、446条用于选择，保留开发口径。
- ABO图搜图：保留用户指定`.8225/.81125`。Exp含rank72 RGB旁路，结构并非与该Sim完全一致；不静默替换成其另一个Sim值。正确性按精确SKU，8个正例完整AP与AP@10分开计算，绘图表主要指标用完整MAP。
- ABO文搜图：主体`cc977b83…`与TRAIN600/VAL200选择的epoch9读出`89e25360…`；100查询、2400候选、每查询24正例。Sim AP由当次保存FP16 embedding补齐，完整MAP=.7499900524，原FP32汇总=.7499857597另存档，不混入绘图表。
- LSP：Sim旧staged e50 `495b9c2c…`；Exp为有界振幅底座e15 `b525e613…`加TRAIN100原读出头适配last30 `58236644…`。用户已在LSP对话确认`.7316`。PCK双侧分组按有效关节数加权，Overall含头颈共14关节。
- SALICON：同PT `036bc8ca…`。原逐图CC保留；其余五指标由原输入缓存和原15000帧CCD固定CPU重放补齐，表内整体值与补算逐图均值对应。CPU核验CC与原CC的设备浮点差异另存，不在绘图表重复一组。
- OpenMoji：同e45 PT `03cb861c…`及TEST manifest SHA。Exp原6000张CCD固定CPU重放，1000条网格、掩膜及八项指标全部补齐，整体与四操作指标同原报告。Accuracy指changed-cell accuracy。
- ABO编辑：Sim `5b4f…`，Exp适配解码器`eeec…`；2304个sample/source/mode逐条一致。原生256×256 RGB float[0,1]逐图算PSNR后平均，非显示PNG重新量化指标。二者不是同完整PT只换光路的纯退化。

## 原始证据、补算与保护

`plot_data.json`同时保存完整来源清单、SHA、未导入Excel的参考记录及明确的`plot_view`选定数据投影。`version_audit.csv`用于留档，不是给画图同学重新选版的表。

原始归档共27,440条，其中2,958条为非指定轮次参考，已从Excel移出：图搜文Sim回放2400条、LGVQ空间Sim发布包复评558条。这些数据保留在原文件与JSON，不删除、不冒充指定轮次的预测。

固定CCD补算调用原runtime，CPU/eval/无梯度，不训练、不调用相机/SLM/SDK、不修改远端源码或数据。OpenMoji使用已发布源码`d202c6987`，SALICON使用`6c07ced42`，完整输入/PT/逐帧SHA及模型state前后SHA均留存。

补算原件在`source_records/derived/`。OpenMoji原请求曝光20000µs、6000份收据实际回读20002µs，gain1/wait240ms分别记录，未缩放图像来改曝光。SALICON原CC主值不覆盖，CPU重放均值与原整体差异在预设3e-5核验容差内；逐图CC最大差异另有原审计。

可恢复快照在`source_records/`，包含`initial_workbook_export_20261026.zip`、`pre_completion_workbook_export_20261026.zip`、`pre_i2t_identity_correction_20261026.zip`和`pre_plot_only_i2t_20261026.zip`。最后一个是本次九本表精简前的完整快照，并非新的测量版本。

导出后独立ZIP/XML回读每个数值，另外核验主值与选版、原特征排名、AP定义和样本身份。验证报告明确区分“导出正确”与“两项Sim原逐样本尚未恢复”。运行数据、PT、CCD、ZIP和Excel不进Git；外部数据上传仍暂停。源码仅按精确路径提交main，不动其他窗口暂存区。
