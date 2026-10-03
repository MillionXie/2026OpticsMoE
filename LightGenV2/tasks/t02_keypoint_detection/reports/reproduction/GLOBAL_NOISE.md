# 最后 global 相位替换为噪声（2026-09-10）

仅改变 `hybrid.optical_branch.core.global_phase.phase.raw_phase`；实际相位为
478×478、逐像素独立均匀分布 `[0,2pi)`。通过逆sigmoid写入同一相位层，
保留10cm传播、478有效场、FFT padding、CCD读出/归一化、两级alpha、专家、router和电子头。
不是替换CCD特征，不是叠加探测噪声，不是移除整个global block。

不重新训练，所有权重冻结。每张噪声mask在1000张test上固定不变；种子42–46，
报告五次结果、均值、总体标准差和范围，不挑最好seed。首先在同设备重评原训练global。
各条件保存逐关节预测；保存5张实际相位NPY及SHA；结束后严格核对所有core/head状态已恢复。
源权重快照保留为新run的best_checkpoint.pt，不覆盖来源文件。

## 已完成结果：旧最佳低alpha模型

完整1000张LSP test、14000个关节；同一模型、同一设备、无重训练。

| 最后global相位 | PCK@0.2（%） | PCKh@0.5（%） | NME（越低越好） |
|---|---:|---:|---:|
| 原训练相位 | 73.4786 | 85.2429 | 0.229755 |
| 均匀随机，seed42 | 73.4500 | 85.2071 | 0.229996 |
| 均匀随机，seed43 | 73.4286 | 85.1500 | 0.230500 |
| 均匀随机，seed44 | 73.3429 | 85.1500 | 0.230928 |
| 均匀随机，seed45 | 73.4286 | 85.1643 | 0.230508 |
| 均匀随机，seed46 | 73.4357 | 85.1786 | 0.230413 |
| 随机相位均值 | 73.4171 | 85.1700 | 0.230469 |

随机相位PCK总体标准差0.03796个百分点，范围73.3429%–73.4500%；
相对原训练相位平均下降 **0.06143个百分点**。5次是同一checkpoint的5张随机相位，
不是5次独立训练，标准差不是统计显著性结论。

这表明该低alpha模型对学到的最后global相位依赖很弱，但不代表全部光学分支无作用：
router和专家相位保持不变，最后global对应alpha仅0.072869。
不能据此预判alpha≥0.5重新训练后的结果；该项仍等待训练完成。

证据：`runs/simulation/global_noise_lowalpha_20260910/final_report.json`。
源checkpoint SHA256：`495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518`。
报告确认`only_global_phase_changed=true`、`all_state_restored_exactly=true`、`retraining=false`。
完整原始报告的轻量副本保存在同目录`global_noise_lowalpha_20260910.json`。

## 两个独立run

- `runs/simulation/global_noise_lowalpha_20260910`：原73.48%最佳模型，alpha约0.1336/0.0729。
- `runs/simulation/global_noise_alpha50_20260910`：等待alpha50训练结束，校验final_report和best的SHA，
  再用它的最终best做同一消融；不会拿尚未训练好的epoch0来代替最终高alpha结果。

2026-09-10 20:26北京时间已启动，实验室GPU0 RTX4090，源码commit
`7f21c335478011db921167438f64509aa7bc0711`，15项测试通过。
低alpha PID1420535，高alpha等待进程PID1420536；等待阶段不占GPU，最长12小时。
高alpha正式评估前还会等待GPU空闲，其他任务不会被停止。

## 命令

从 `/DATA/DATA1/guest3/lsp_global_noise_source_20260910` 执行；新run-dir必须不存在。

```bash
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
TASK=$ROOT/LightGenV2/tasks/t02_keypoint_detection
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
/home/guest3/miniconda3/envs/xml/bin/python -m LightGenV2.tasks.t02_keypoint_detection.global_noise \
  --profile main_dc20_no_shift_warmstart \
  --checkpoint "$TASK/runs/simulation/refinement_20260909/staged_heatmap/best_checkpoint.pt" \
  --data-root "$ROOT/data/lsp_pose" --cache-dir /DATA/DATA1/guest3/.cache/huggingface/hub \
  --run-dir "$TASK/runs/simulation/global_noise_lowalpha_20260910"
```

高alpha使用 `--profile alpha50`，checkpoint指向`alpha50_staged_seed42_20260910/best_checkpoint.pt`，
run-dir换为`global_noise_alpha50_20260910`；训练未完成时加
`--wait-for-report "$TASK/runs/simulation/alpha50_staged_seed42_20260910/final_report.json"`。
检查status.json是否complete，再读final_report.json；等待/未完成不是有效结果。

这项测试回答“已训练模型是否依赖学到的global相位”，不等于比较随机固定global重新训练后的最佳性能。
