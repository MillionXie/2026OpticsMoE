# 自采照片小样本域迁移（2026-09-28）

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
