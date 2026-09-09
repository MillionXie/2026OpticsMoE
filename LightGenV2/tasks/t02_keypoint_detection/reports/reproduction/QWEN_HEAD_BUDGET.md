# LSP：Qwen 读出头预算对照

2026-09-09。目的：检查读出头参数量的影响，保留原大头 baseline，不以缩小头来人为降低对照成绩。

## 只缩通道，不改 Qwen

| 头 | 输入通道 | 中间通道 | 参数 |
|---|---|---|---|
| 历史 Qwen Deconv128 | 1024 | 全部128 | 1,102,990 |
| 新 Qwen Deconv40 | 1024 | 全部40 | 138,422 |
| 当前光学姿态头 | 192 | 160→128→96 | 133,425 |

Deconv40 比光学任务头多 3.75% 参数，约为原 Qwen 头的 12.55%。
这是**头参数预算接近**，不是三个头计算图完全相同，也不是模型总参数一致。
光学其他可训练电子部分另有 702,823 参数，不能漏算。

新 baseline 完整执行冻结 Qwen3-VL-Embedding-2B 的 **24 个原生视觉 Transformer blocks**，
不执行 language，不用光学，不改主干、不做 LoRA，也不缓存增强前的图像特征替代训练。
给定标注人体位置裁剪到224×224，输出196×1024，恢复为1024×14×14。
只训练：LayerNorm + Linear1024→40 → 两层3×3卷积 → 两级4×4 stride2反卷积
（14→28→56）→ 3×3细化 → 1×1输出14张热图；保持原 GroupNorm/GELU。
沿用已有 `DeconvPoseHead`，没有复制或改写主干/算子。

## 训练及报告口径

- 10,428 train（9428 HR-LSPET + 前1000 LSP），1000 test（后1000 LSP）。
- 随机初始化新头；原大头、光学权重均不覆盖。
- 40 epoch、batch8、AdamW LR 0.001、weight decay沿用旧配置、seed42。
- 沿用人体裁剪、增强、Gaussian heatmap MSE + 0.1坐标SmoothL1。
- 保持 backbone 冻结且检查它没有梯度。测试无TTA，hardargmax及PCK/PCKh/NME定义不变。
- 每轮完整 test，选择最高PCK（同分NME/loss/最早epoch），不使用EMA。
  历史大头按train loss选模；因此新旧差异还包含选模口径，不能只凭最高test数值作纯头容量因果结论。
- 不预设轻量头性能应该更低；在原大头旁边增列，而不是替换或隐去较强结果。
- 本次只训练性能，不将与其他训练共享GPU时的wall time作为正式推理速度/能耗。

## 运行

独立源码工作树 `/DATA/DATA1/guest3/lsp_qwen_head_source_20260909`。

```bash
cd /DATA/DATA1/guest3/lsp_qwen_head_source_20260909
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=1 /home/guest3/miniconda3/envs/xml/bin/python -u \
  -m LightGenV2.tasks.t02_keypoint_detection.train_qwen_head \
  --data-root /DATA/DATA1/guest3/2026OpticsMoE/data/lsp_pose \
  --cache-dir /DATA/DATA1/guest3/.cache/huggingface/hub \
  --run-dir /DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t02_keypoint_detection/runs/simulation/qwen_deconv40_seed42_20260909
```

run-dir 必须未存在。机器迁移时修改数据/cache路径。
只验证代码时添加 `--smoke --workers 0` 并使用新的 `runs/smoke/` 路径；smoke仅2 train/2 test，不是性能。

输出：`run_manifest.json`（源码/命令/环境/模型快照/数据清单SHA）、`resolved_config.yaml`、
`training_history.json`、`status.json`、best/last两个checkpoint及其SHA、
`final_report.json`、`metrics/` 下最终1000张图的关节预测。模型权重不进Git。

核对顺序：先看 `status.json`；训练中看history；结束后用final_report，不拿最后一轮代替best。
原baseline详细定义见 [BASELINE_METHODS.md](BASELINE_METHODS.md)，光学续训见 [STAGED_REFINEMENT.md](STAGED_REFINEMENT.md)。
