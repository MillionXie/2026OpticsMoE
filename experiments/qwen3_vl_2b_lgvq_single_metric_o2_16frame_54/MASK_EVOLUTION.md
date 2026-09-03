# 每 5 epoch 相位 mask 快照

正式训练每 5 epoch 自动写入：

```text
<output_dir>/phase_snapshots/
├── epoch_0005.pt
├── epoch_0010.pt
├── ...
└── manifest.json
```

这些文件是**相位分析快照**，不是完整模型 checkpoint。它们不含电子读出头或优化器，
不能传给模型的 `load_state_dict`。正常恢复/部署仍使用
`best_observed_test_checkpoint.pt`。

每个 `.pt` 的关键内容：

```python
{
    "contract": "optical_phase_evolution_snapshot_v1",
    "epoch": 5,
    "target_name": "temporal",
    "fusion_alpha": {
        "vision_expert": 0.57,
        "vision_global": 0.57,
        "language_expert": 0.57,
        "language_global": 0.57,
    },
    "unmodulated_leakage": {...},
    "planes": {
        "parallel_router.raw_router_phase": {
            "shape": [...],
            "raw_parameter": Tensor,
            "phase_rad": Tensor,  # 范围 [0, 2π]
        },
        ...
    },
}
```

推荐使用严格读取函数，避免把 raw 参数误认为弧度：

```python
from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.phase_snapshots import (
    load_phase_snapshot,
)

snapshot = load_phase_snapshot(".../phase_snapshots/epoch_0020.pt")
phase = snapshot["planes"]["parallel_optics.raw_expert_phase"]["phase_rad"]
print(snapshot["epoch"], phase.shape, phase.min().item(), phase.max().item())
```

批量检查变化：

```bash
python -m experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.phase_snapshots \
  experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54/runs/lgvq_temporal_qwenfront_o2_16f54_dc20/phase_snapshots
```

程序会生成 `phase_evolution_summary.csv/json`。其中
`wrapped_delta_from_first_rms_rad` 使用圆周相位差：

```text
atan2(sin(phi_t - phi_0), cos(phi_t - phi_0))
```

因此跨越 `0/2π` 不会被错误统计为一次巨大的变化。`manifest.json` 同时记录每个
快照的 SHA256、epoch、测试 SRCC 和四层 alpha，文件传给其他同学后可以核验是否损坏。

## 20% 直流未调制分量的定义

训练期每次光传播从配置区间采样名义未调制光功率比例 `η`，相位调制写为：

```text
M = sqrt(1-η) * exp(i*phi) + sqrt(η)
```

正式配置使用 `η ∈ [0.20,0.35]`，确定性仿真测试使用 `η=0.20`。这是相干复振幅
模型，CCD 上还包含两项之间的干涉，所以不能把每个像素的最终强度机械解释成“恰好
20% 直流”。它表达的是在传播前保留至少 20% 的名义未调制光功率，同时不增加第二
条光路或第二次传播。
