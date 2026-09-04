# 光学相位权重演化分析交接

这份说明对应 Temporal-9 正式候选，不是旧 16 帧模型，也不是 Spatial 模型。每个
`phase_snapshots/epoch_XXXX.pt` 只保存光学相位、四个融合系数和当次测试指标，未保存
优化器和完整电子网络；完整网络定义见 `modeling.py`，正式最佳模型另见
`best_observed_test_checkpoint.pt`。

## 1. 快照内到底是什么

读取方式：

```python
from experiments.qwen3_vl_2b_lgvq_single_metric_o2_16frame_54.phase_snapshots import load_phase_snapshot

s = load_phase_snapshot("phase_snapshots/epoch_0005.pt")
phi = s["planes"]["parallel_optics.raw_expert_phase"]["phase_rad"]
print(s["epoch"], phi.shape, phi.min().item(), phi.max().item())
```

必须分析 `phase_rad`，单位是 rad、范围为 `[0, 2π]`。`raw_parameter` 是训练内部的
无界参数，物理相位由 `phase_rad = 2π·sigmoid(raw_parameter)` 得到，不能把二者混为
一谈。

六类张量如下：

| 键 | 形状 | 含义 |
|---|---:|---|
| `parallel_router.raw_router_phase` | `[9,77,77]` | 9 帧各自的视觉光 router |
| `parallel_optics.raw_expert_phase` | `[36,77,77]` | 9 帧 × 每帧 4 个视觉专家 |
| `parallel_optics.raw_global_phase` | `[478,478]` | 视觉全局相位 |
| `serial_router.raw_router_phase` | `[109,109]` | 文本条件序列的光 router |
| `serial_optics.raw_expert_phase` | `[4,109,109]` | 4 个序列专家 |
| `serial_optics.raw_global_phase` | `[478,478]` | 最后一层全局相位 |

`parallel_optics.raw_expert_phase` 的索引是 `index = frame_index*4 + expert_index`。
例如只分析“第 1 帧的专家 1”，用索引 0；分析同一个专家在 9 帧中的差异，用索引
`0,4,8,...,32`。不要把 36 个面误认为 36 个独立 MoE 专家：逻辑专家始终只有 4 个。

## 2. 推荐的一专家分析

直接运行：

```powershell
conda activate xml
python analyze_phase_evolution.py `
  --snapshot-dir phase_snapshots `
  --plane parallel_optics.raw_expert_phase `
  --expert 0 --frame 0 `
  --output-dir analysis_one_expert
```

输出包含相位随 epoch 的图、相对第一个快照的环形相位 RMS、变化像素比例、SVD/PCA
轨迹、可供 MATLAB/Python 二次分析的 CSV 和 NPY。分析脚本默认只取一个
`77×77` 面，因此不会把不同帧/不同专家混到一个矩阵中。

相位有 `0↔2π` 周期边界，不能直接把 rad 做普通欧氏 PCA。脚本采用
`[cos(phi), sin(phi)]` 两通道展开后做 SVD；这是推荐主结果。`phase_rad_matrix.npy`
仍被保留，仅用于可视化或明确做过相位展开的算法。

## 3. 与完整网络的关系

前 3 个物理 pass 使用 9 帧 3×3 并行布局，单帧 lane 为 156×156；每帧内部 2×2
专家，每个专家 77×77、间隙 2 像素。后 3 个 pass 是含 9 个 frame token 与固定
Temporal prompt token 的序列光场，专家为 109×109。两次 router 都由 CCD 四区域
能量产生 Top-2，不存在电子 router。四个融合点均执行独立 RMS 尺度对齐，再用
`(1-alpha)·electronic + alpha·optical`。

若修改 `modeling.py` 后读取正式 checkpoint，必须仍满足 checkpoint 中的
`architecture` 与 config 生成的 `architecture_label` 完全一致；禁止使用
`strict=False` 掩盖结构差异。

