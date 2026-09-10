# LSP 两级 alpha≥0.4、PCK≥0.73 目标续训

## 60轮已完成结果与下一轮精修

原`alpha40_staged_seed42_20260910`已完成60轮，best为epoch20：
完整1000张test PCK=0.72692857，PCKh=0.84792857，alpha=0.41823137/0.41819504。
未达到0.73，差0.00307143。相同权重去光PCK=0.70478571；
5张固定随机global相位的平均PCK=0.72347143，标准差0.00216588。
原始证据为该run和`global_noise_alpha40_20260910`的`final_report.json`。

新的`alpha40_polish`从上述best继续40轮，不加载last，不重置alpha，
不增加任何电子参数或损失。前30轮电子/路由/相位/CCD/头初始LR分别
3e-6/3e-5/3e-4/2e-5/3e-5，余弦降至20%；后10轮固定主支/光学/alpha，
只训练CCD 5e-6和头1e-5。保留原光学扰动；router训练探索噪声从来源epoch20的
原始2/3强度逐步降至零，不重新升到初次warm-start水平。无推理增强或额外后处理。

源SHA：`655805a0a5f1d10661aca9207387060cc89020423c938903c6e85c61601a1cf4`。
启动时严格验证core状态张量全部与来源一致、alpha满足下限、电子预算未变；
先完整复评新起点，原best作为epoch0候选保留。新run仍只保存best/last，周期test选模。
运行下文命令时将profile改为`alpha40_polish`，source改为
`$TASK/runs/simulation/alpha40_staged_seed42_20260910/best_checkpoint.pt`，
run-dir改为`$TASK/runs/simulation/alpha40_polish_seed42_20260910`。
配置/完整学习率日程见`configs/moe_alpha40_polish.yaml`及run_manifest。

2026-09-10 22:58北京时间正式启动于GPU2 RTX3090，PID2091458，源码
`d7829addb389185276e6cc7476e036383aad26a7`（17项测试及真实GPU小样本训练通过）。
干净worktree为`/DATA/DATA1/guest3/lsp_alpha40_polish_source_20260910`。
PID2091459在不占GPU的情况下等待完成报告，再用同一张GPU执行5种随机global评估，
输出`runs/simulation/global_noise_alpha40_polish_20260910`。
评估使用`--profile alpha40`，因为polish没有改变alpha40的推理配置或架构合同。
不得把当前启动记录当作已达0.73的结果。

后文是首轮alpha40的启动记录与命令，不代表该轮尚未完成。

2026-09-10 20:53北京时间已在实验室GPU2 RTX3090启动，PID1542218；
源码commit `60c3775e4fbf4e4d336de22a4532f84d8405bda7`（16项测试和GPU smoke通过）。
worktree：`/DATA/DATA1/guest3/lsp_alpha40_source_20260910`。
随机global消融等待进程PID1542221，run为`global_noise_alpha40_20260910`，
等待训练最终报告和GPU空闲，不在等待阶段占GPU；最多等待12小时。
原alpha50训练保留作对照，不修改其已运行的配置。

新profile `alpha40`，不覆盖alpha50或旧低alpha结果。0.73为达标条件，不是已取得的结果。
最终报告同时检查1000张test的PCK和两处alpha，smoke结果不算达标。

## 不变的结构和数据

沿用Top-2光router、四专家、最后global、478有效场、17um、532nm、10cm及原噪声和同尺度融合。
仅改变alpha硬范围和训练日程，无新增电子层、通道、任务头或损失。
两处alpha均为 `0.40 + 0.55 * sigmoid(logit)`，加载旧权重后重置到0.42；训练和推理均受约束。
电子参数预算固定：电子主支616423、CCD读出86400、姿态头133425，合计836248；
光router相位50176、专家/global相位429188。启动时严格核对预算，旧PT严格加载。

从旧73.4786% EMA best开始，SHA256为
`495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518`。
抬高alpha后先完整复评，这个新起点不保证仍有73.48%。
10428训练图、1000测试图；epoch1、每5轮及最后一轮测试，按PCK选EMA best。
沿用周期test选模协议，不称为封存测试。

## 60轮训练

- 1–3：只调整原CCD读出和姿态头，学习率1e-4；相位、电子主支和alpha不动。
- 4–50：联合训练，电子2e-5、router3e-4、专家/global相位2e-3、CCD/头1e-4，余弦降至20%。
- 51–60：固定光学、电子主支和alpha，只微调原CCD读出/头2e-5。

不减少原光学噪声，不启用TTA。记录每层实际alpha和相位变化；结束后自动去光评估。
`global_noise`支持alpha40，训练结束后可自动评估5个随机global相位，所有其他权重冻结。

## 实验室命令

在GitHub对应commit的干净worktree执行（不要在正在运行的旧worktree切换源码）：

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
TASK=$ROOT/LightGenV2/tasks/t02_keypoint_detection
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
/home/guest3/miniconda3/envs/xml/bin/python -m LightGenV2.tasks.t02_keypoint_detection.refine \
  --profile alpha40 --seed 42 --batch-size 24 --workers 4 \
  --source "$TASK/runs/simulation/refinement_20260909/staged_heatmap/best_checkpoint.pt" \
  --data-root "$ROOT/data/lsp_pose" --cache-dir /DATA/DATA1/guest3/.cache/huggingface/hub \
  --run-dir "$TASK/runs/simulation/alpha40_staged_seed42_20260910"
```

新run目录必须不存在。精确命令、commit、GPU、参数预算、数据划分、配置和训练历史保存于run。
训练完成读取`final_report.json`中的`target_met`、`test`、`fusion`、`optical_off`和checkpoint SHA。
`target_met=false`必须如实报告，不能通过降低测试时alpha或者扩大电子结构补齐数值。
