# LSP 两级 alpha≥0.4、PCK≥0.73 目标续训

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
