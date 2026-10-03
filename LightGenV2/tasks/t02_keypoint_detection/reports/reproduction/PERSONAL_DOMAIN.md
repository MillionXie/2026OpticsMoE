# 自采照片小样本域迁移（2026-09-28）

## 画图交付（LightGen2-only）

本地`data/lsp_pose/LightGen2_figure_delivery_20260928`，PREVIEW.html只显示Input、GT、LightGen2、叠加。
54照片/55人体各独立目录含四张无标题PNG、preview拼图、metrics.md/JSON及keypoints.json。
一级SUMMARY.md/JSON总指标、SAMPLE_METRICS.json逐样本索引、README、FILES_SHA256清单。
图例蓝色GT/橙色LightGen2；GT仍RTMPose自动预标注，12点展示与评价，非人工GT或CCD实验。
固定83.3333%最佳参数重新批量评估一次导出坐标，run `personal_curated20_figure_eval_20260928`，
无训练；545/654命中，均误差14.466378像素，NME0.14528649。与历史14.46527的微小变化
来自固定权重复评数值差异，交付图和逐样本指标使用此次同一批量预测，PCK完全一致。
旧visuals保存的是index*4，正式评估为(index+.5)*4-.5；导出工具不沿用旧绘图坐标，
而使用evaluate_model返回的像素中心坐标，并强制核对逐样本汇总和评估器一致。
所有54张测试照片保留，无按指标筛图；896PNG由224输入放大用于排版，不是高分辨率推理。
入口`personal_figure_delivery --dataset ... --source-run <head_testselect100> --evaluated-rows <figure_eval>/evaluated_predictions.json --output <新目录>`。

## 恢复原相位，继续姿态头并按周期test选模

用户明确要求：返回83.3333% head-only best（SHA `0c57938c87deef3d901605214e69514f2b43c8464447db25ed6f8278e1c2fe13`），
不加载phase-head100的权重。原20/54划分、原相位及alpha完整冻结，只训133425参数末端头。
新增100epoch：`--head-only --continue-head --test-interval 5 --epochs 100`，
source仍为`personal_curated20_ours_head_evalcore_pilot_s42_20260928/best_checkpoint.pt`。
LR1e-4余弦至1e-5；epoch0、5、10…100测试，按PCK最高保存best，同分比test heatmap MSE。
只保存best/last；逐轮训练及全部周期test指标保留，不保存每5轮的重复checkpoint。
该TEST明确参与选模，不再称作封存测试；Qwen原始权重不微调、不重新选模。
run `personal_curated20_head_testselect100_pilot_s42_20260928`，预标注诊断性质不变。

已完成100轮（源码`f1b51af2a`）：起点83.3333%仍最高，best_epoch=0；
新增轮次最高epoch10为83.1804%，最后epoch100为79.8165%。因此最终best保留起点参数，
没有提高；Qwen仍87.4618%。core_unchanged=true，原相位/alpha/电子前端逐轮不变。
新best容器SHA `d8287ddb8dc4ae8e06a502cc4037b1c459a9e5f4452f952dd1ef543f2b7e5c7a`，
与起点PT文件SHA不同是manifest/epoch更新，不代表模型权重更新。
完整20次周期test见run内training_history.json；best/last及预测已下载本地并校验SHA。
新的可视化在数据目录`head_testselect100_comparison/COMPARE.html`，明确标记test参与选模。

## 相位+末端头继续微调

从下文83.3333%的head-only best继续100epoch，不重划20/54张。
源码`f79bdf0ce`；run `personal_curated20_phasehead100_pilot_s42_20260928`。
新增`--phase-head-only --phase-lr 0.001 --epochs 100`，source使用
`personal_curated20_ours_head_evalcore_pilot_s42_20260928/best_checkpoint.pt`。
专家/global相位LR1e-3、光router相位1e-4、pose head3e-5，余弦降至各自0.1倍；
原噪声/正则恢复训练模式，所有非相位core参数含alpha和中间CCD读出固定，逐轮hash校验。
原始Qwen不重训；按train MSE选best，最终一次test；记录raw phase变化RMS验证相位更新。
仍为未经人工确认的12点预标注诊断，不声称实际光路改善。

完成：best按train MSE选择epoch100，test PCK12由0.83333333降至0.81651376，
像素误差14.46527→15.82204；Qwen未微调仍0.87461774。未取得改善，保留原head-only版本。
非相位core hash不变、alpha不变；raw参数RMS变化router0.003357，4专家0.011275–0.015844，
global0.016079，确认相位被更新（这些是raw参数单位，不是弧度）。
新checkpoint SHA `1dd8104104be31fdf654415e18f8d61b619114f0902c086ba5c031fa8efa432c`，
已下载本地并验证SHA。全测试对照为`personal_curated20_20260928/phasehead100_comparison/COMPARE.html`。
训练误差继续降低但测试下降，只能说本次续训未改善泛化，不能归因于唯一机制。

## 用户筛选版本：20张、仅Ours末端姿态头适配

数据`data/lsp_pose/personal_curated20_20260928`独立保存，不删除原始图像或历史实验。
`personal_curate`按用户29个人体ID排除（27张照片退出），剩74张/75人体，
重新编号photo_000–073；id_mapping.json保留原始人体ID，exclusions.json保存排除清单。
原003/089仅标注人像不完整，未排除。此筛选发生在查看预测之后，须明确报告筛选条件，
不能替代原101张整体性能；不能将筛选后的提升都归因于训练。

seed42整组分20张训练（6组）、54张测试（55人体），两模型共用同一测试。
从原始LSP权重开始：Ours仅133425参数的最终pose head可训练，完整core（包括CCD中间
读出、电子mixer、alpha、路由和相位）冻结并逐轮校验state digest；不增加结构。
60epoch，LR1e-4余弦至1e-5，训练图像增强同前，仅train MSE选模，test不参与。
Qwen完整Vision+原Deconv128保持原LSP权重，仅evaluate-only，不使用自摄训练图。
这是不对称适配协议，不是双方等量适配比较；仍为12点预标注诊断。

执行入口：`personal_curate --source <personal_20260928> --output <personal_curated20_20260928>`；
`personal_finetune`使用新目录annotations_provisional.json，Ours加`--head-only --epochs 60`，
Qwen加`--evaluate-only`，两者加`--allow-provisional`，不要再次加fewshot-photos重分数据。
服务器和本地标注SHA：`c442100dabf7e2fc9e213e3f4cef53a9e7155c1a12bac9dce276dfc5d0843422`。

先行run `personal_curated20_ours_head_pilot_s42_20260928`权重冻结成功，但core专用forward_groups
绕过了原运行模式hook；保留作为带训练扰动的试跑，manifest中的core_eval声明不成立，不作主结果。
修正源码`25294766d`从学生wrapper进入时设置core.eval，再从原LSP权重重跑
`personal_curated20_ours_head_evalcore_pilot_s42_20260928`。Qwen run
`personal_curated20_baseline_head_pilot_s42_20260928`仅推理，不受该hook影响。

最终完成：共同测试54张/55人体，654有效四肢点。Ours选epoch60，core_unchanged=true。

| 方法 | 预标注PCK12 | 平均像素误差（224裁剪） |
|---|---:|---:|
| Ours 原始LSP权重 | 0.80428135 | 16.41619 |
| Ours 20张仅末端头微调 | 0.83333333 | 14.46527 |
| Qwen 原始权重，不微调 | 0.87461774 | 10.76496 |

差距7.0336→4.1284百分点，未追平；alpha保持0.41805938/0.41805139。
新best SHA `0c57938c87deef3d901605214e69514f2b43c8464447db25ed6f8278e1c2fe13`。
权重、逐样本预测、训练记录已下载；本地展示为新数据目录`comparison/COMPARE.html`。
这些数值为筛选后自摄照片的预标注一致性，不是人工GT性能或光路实测。

## 10张少样本配对试验

追加`--fewshot-photos 10 --seed 42 --epochs 60`，不加evaluate参数；源权重仍使用原LSP版本。
由固定seed打乱拍摄组，再用整组子集和选择恰好10张训练，其余91张测试；不读取模型误差选图。
run manifest记录双方同一训练/测试图片ID和组号。此划分是看过全量结果后的探索性实验，
不称为封存测试。未人工确认标签时仍为pilot，监督与评估12个四肢点。
Ours保持原结构、alpha>=0.4、原噪声/正则，先3轮读出适配再低学习率联合微调；
baseline冻结完整Qwen视觉主干、微调原Deconv128。双方60轮，按clean train MSE选模。
每份报告均给出同一91张test上微调前/后，禁止用101张零样本结果代替before。

已完成：训练commit `dd31d232d`，两者60轮，train MSE均选择epoch60。
训练组capture_10/18/21共10张（034,035,058–062,072–074）；test91张/94人体，1119有效四肢点。

| 模型 | 同一test微调前PCK12 | 微调后PCK12 | 平均像素误差前→后 |
|---|---:|---:|---:|
| Ours | 0.747096 | 0.761394 | 17.3829→16.1354 |
| Qwen baseline | 0.851653 | 0.892761 | 11.1240→9.8427 |

未缩小差距；不得声称追平。Ours电子836248不变、alpha=0.41805777/0.41805008。
run ID `personal_few10_ours_pilot_s42_20260928`、`personal_few10_baseline_pilot_s42_20260928`。
best SHA分别`4b634796d271b6e564413bb4c353c0e9d6b965882a6e16112f4a56fd624ea91e`、
`b1a8f655433beba94bea6852dea2e3cd1288b34d36b600d457f3a3626d4555e2`。
本地完整对照页`data/lsp_pose/personal_20260928/few10_comparison/COMPARE.html`，含全部测试人体，
权重和run已下载并核对SHA。未人工审核，以上仅为预标注一致性。

## 零样本仿真优先（用户最新协议）

原LSP训练/结果报告仍保留；自摄照片用于可展示图像与额外域外验证，两套指标分开。
先使用下文两份原始LSP checkpoint，不使用personal_pilot微调权重，在全部101张照片、
104个前景人体上评估。已有81/20划分不改写，使用只读的全量评估视图。
在下文personal_finetune命令中追加 `--evaluate-only --evaluate-all`，设置新run目录
`personal_zeroshot_<method>_all_pilot_20260928`。此模式不执行训练、不保存新checkpoint。
未人工审核时仍加`--allow-provisional`并使用provisional标注，仅报告PCK12预标注一致性。
人工核验后可固定同一原始权重复评，无需为核验标签而重新训练。
若后续少样本迁移，先按拍摄组固定约10张训练照片；其同组连拍不得留在测试集，
两模型从各自原始LSP权重开始。零样本/迁移后均重报同一个剩余test，不能将全101张
零样本分数直接当作不同测试子集的迁移提升。已看过全量结果，须披露这是探索性划分。
本轮不自动根据测试分数挑选训练图或启动额外训练。

已完成（执行源码`1eafc5df1`，单GPU2 RTX3090串行）：

| 原始LSP权重 | 全101张/104人体预标注PCK12 | 224裁剪内平均像素误差 |
|---|---:|---:|
| Ours alpha≥0.4 | 0.7699757869 | 16.19598 |
| 完整冻结Qwen Vision + Deconv128 | 0.8627925747 | 10.71309 |

共同分母1239有效四肢点；没有人工GT，不作论文正式准确率。头颈排除，PCKh无有效分母，
JSON中的PCKh=0是空计数占位，不能报告为性能0。与下文20张子集不同，不能直接比较。
运行`personal_zeroshot_ours_all_pilot_20260928`及`personal_zeroshot_baseline_all_pilot_20260928`
均在任务`runs/simulation/`；final_report、run_manifest、完整命令、resolved config、
逐人体预测和before_images已从服务器下载本地。原始checkpoint与标注SHA同下文。

## 本次已完成的预标注试跑（不是论文GT性能）

源码：`fbb775347`（完整SHA见每个run的`run_manifest.json`），分支
`codex/t02-personal-pose-20260928`；本地隔离源码 `.worktrees/t02_personal_pose`。
服务器源码 `/DATA/DATA1/guest3/t02_personal_source_20260928`。
24项测试及两模型各一轮GPU流程检查通过后，单GPU2 RTX3090串行各20轮完成。

| 模型 | 迁移前：预标注PCK12 | 迁移后：预标注PCK12 | 最优训练MSE轮数 |
|---|---:|---:|---:|
| Ours | 0.911392 | 0.940928 | 20 |
| Qwen Deconv128 | 0.928270 | 0.991561 | 20 |

分母是20张留出照片中的237个有效四肢点；头颈不计，**不是原LSP14点PCK**。
只有5个独立留出拍摄组，组内连拍相关性较强，不能按237个独立观测夸大统计证据。
baseline此轮更好；Ours的部分遮挡手腕仍不准，全部测试图都在对照页中保留。
Ours alpha=0.41805884/0.41805112，结构未扩容。

运行ID：`personal_pilot_ours_s42_20260928`、`personal_pilot_baseline_s42_20260928`，
本地和服务器都位于任务`runs/simulation/`。
两份best SHA分别为：

- Ours：`a5f6b32656311904b46d09cad78b51e0cde6746d3454b3e00704d42187c27135`
- Baseline：`8362afea87b88c14438812565e2b2e4308d96eaad87c1035eb066731b8c5d314`

数据标注SHA：`6dbb0745e6a4a4b0f5468f8622b3782deac325fd9560c853d0982b35575b47d2`。
传输ZIP SHA：`cdd64198abee6e9c485ab330c2036159817a16e5da09c4757b8a3c95bb19db7d`，服务器解压前核验。
原图未修改；权重、日志、前后预测图已下载本地，并再次核对best权重SHA。
`data/lsp_pose/personal_20260928/COMPARE.html`展示全部测试图；`review.html`供人工核验。
正式14点域迁移仍待人工确认标注，不把此试跑标为正式完成。

## 数据、标注、可报告范围

原图 `data/lsp_pose/Lsp/` 共101张，保留不动。用户明确允许训练及论文展示。
派生数据 `data/lsp_pose/personal_20260928/` 不进Git，包括照片、预标注、人工审核页、224裁剪。
EXIF转正后五页总览逐页检查，当前图像全部正立；横拍照片保留横构图，不按宽高强制旋转。
压缩工作副本最长边1600，去除EXIF/GPS，未拉伸。人体正方形crop→224×224，输出14×56×56。
这是给定人体位置的top-down关键点任务，不是整图多人检测精度；多人同图分不同person条目。

按连拍、同姿势、双机位相似照片人工分28组，seed42整组划分81张train/20张test。
前景目标筛选后84个训练人体、20个测试人体。背景提案保留但排除，必要时审核页重新启用。
人物/场景并不互斥；本实验只能说明**同拍摄场次的小样本迁移**，不能宣称跨人跨场景泛化。
增强在划分之后在线执行：尺度±10%、中心±3%、水平翻转0.5（同步交换左右关节）、
亮度/对比度±10%。测试无增强，不把增强图算成新独立样本。

RTMPose-m+YOLOX-m通过rtmlib生成独立预标注，与被评估的两模型无关。
来源/模型URL：https://github.com/Tau-J/rtmlib ，输出JSON记录完整模型地址。
COCO的12个四肢关节直接映射；颈部/头顶仅几何占位，**不是LSP真值**。
`review.html` 可直接本地浏览器打开：选人、拖点、屏蔽不可定位点、交换左右、确认后导出JSON。
把下载的`annotations_reviewed.json`放回同一文件夹。原图不会被修改。
不要把程序生成的`reviewed:false`批量改为true冒充人工审核。

`--allow-provisional`显式试跑只监督/计算12个四肢点，排除不可信头颈点。
该结果仅为**与独立自动预标注的一致性**，不能当论文GT准确率或与原LSP14点直接比较。
正式入口默认拒绝未经核对的人体；确认14点或将不可判断点设无效后才做正式重训。
两模型共用相同图片、裁剪、split、增强、损失及随机种子。

## 模型与微调

- Ours：`alpha40_distill_seed42_20260911/best_checkpoint.pt`，SHA256
  `dbc059e2a7eddefac73d3b9bb158bf0140d440dfb396e0aa2956ad41670fc96a`。
  α保持[0.4,0.95]，初值约0.418；电子836248参数不扩容，光路/Top2不改。
- Baseline：历史opt2的`teacher_best_train_loss.pt`，SHA256
  `0a4569f288f6de424b1b412452fa804a96e58d8f7e8655efaa23f461ab8cc735`。
  完整冻结Qwen原生Vision+1102990参数Deconv128头；不换较弱Deconv40，不加LoRA。
- 20 epoch初始试跑；batch8、seed42，两模型使用masked heatmap MSE，坐标损失0。
  Ours保留原光学正则，前3轮只训原CCD读出和姿态头，后续小步联合。
  LR electronic3e-6/router3e-5/phase3e-4/CCD1e-4/head1e-4，余弦降至0.1倍。
  Baseline只训练原头，LR1e-4同余弦日程。
- 最佳权重按未增强**训练集**热图MSE选择；测试仅迁移前和选定权重后评估，不用test挑epoch。
  小样本训练误差最优不代表泛化最优；这是试跑，不声称解决过拟合。
- 输出best/last、参数/数据/源码SHA、before/after数值、逐图预测坐标和前后叠加图。
  所有运行进入本任务runs。一次仅用一张GPU，ours和baseline串行。

## 从头操作（源码仓库根目录）

只在新输出目录运行预处理，不覆盖已审核数据：

```powershell
# 本机已生成，不要重复运行下列准备命令；直接打开review.html即可。
# 换机器复现时，先checkout上述分支，再设置绝对数据路径。
Set-Location C:\Users\Xml12\OneDrive\2026OpticsMoE\.worktrees\t02_personal_pose
$poseData = 'C:\Users\Xml12\OneDrive\2026OpticsMoE\data\lsp_pose'
python -m LightGenV2.tasks.t02_keypoint_detection.personal_prepare --source "$poseData/Lsp" --output "$poseData/personal_20260928"
python -m LightGenV2.tasks.t02_keypoint_detection.personal_prelabel --dataset "$poseData/personal_20260928"
python -m LightGenV2.tasks.t02_keypoint_detection.personal_split --dataset "$poseData/personal_20260928"
# 打开 data/lsp_pose/personal_20260928/review.html 审核；导出 annotations_reviewed.json 放回同目录
```

预标注环境与训练环境分开；本地已验证Python3.11、rtmlib0.0.16、onnxruntime1.20.1、Pillow12.3。
不要修改用户的xml环境。服务器沿用xml/PyTorch2.6.0cu124/Transformers4.57.3。

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
TASK=$ROOT/LightGenV2/tasks/t02_keypoint_detection
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
# 先确认GPU空闲，再指定CUDA_VISIBLE_DEVICES。这里不硬编码设备编号。
python -m LightGenV2.tasks.t02_keypoint_detection.personal_finetune --method ours \
 --annotations "$ROOT/data/lsp_pose/personal_20260928/annotations_reviewed.json" \
 --source "$TASK/runs/simulation/alpha40_distill_seed42_20260911/best_checkpoint.pt" \
 --cache-dir /DATA/DATA1/guest3/.cache/huggingface/hub \
 --run-dir "$TASK/runs/simulation/personal_reviewed_ours_s42" --epochs 20
python -m LightGenV2.tasks.t02_keypoint_detection.personal_finetune --method baseline \
 --annotations "$ROOT/data/lsp_pose/personal_20260928/annotations_reviewed.json" \
 --source "$ROOT/experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/runs/lsp_pose_optical_moe16_opt2/checkpoints/teacher_best_train_loss.pt" \
 --cache-dir /DATA/DATA1/guest3/.cache/huggingface/hub \
 --run-dir "$TASK/runs/simulation/personal_reviewed_baseline_s42" --epochs 20
```

本地生成全测试集对照页（需先下载两份run；comparisons目录不存在时执行）：

```powershell
python -m LightGenV2.tasks.t02_keypoint_detection.personal_report --dataset "$poseData/personal_20260928" --ours C:/Users/Xml12/OneDrive/2026OpticsMoE/LightGenV2/tasks/t02_keypoint_detection/runs/simulation/personal_pilot_ours_s42_20260928 --baseline C:/Users/Xml12/OneDrive/2026OpticsMoE/LightGenV2/tasks/t02_keypoint_detection/runs/simulation/personal_pilot_baseline_s42_20260928
```

预标注诊断必须改用`annotations_provisional.json`、增加`--allow-provisional`，run名带`pilot`。
诊断与正式输出分开，不覆盖原权重。标签纠正遵循图像事实，两模型同时重评，不按模型表现修改标签。
