# T02 关键点检测（LSP）

本任务只处理视觉分支，输出 LSP 的 14 张关键点热图。冻结的
Qwen3-VL-Embedding-2B 仅执行 patch embedding；原生 Vision Transformer
block 不执行。两种正式方法共享数据划分、电子 mixer、同尺度融合、姿态读出头、
10 cm/17 μm 传播、k 空间约束、位移/增益/偏置/读噪声和 20%–30% 相干 0 级分量。

上述说明只针对两种光学仿真方法。论文中的大模型 baseline 是另一条直接路径：输入
224×224 图像后完整执行冻结 Qwen 的所有原生 Vision Transformer blocks，取最后一层
原生视觉 token 恢复为 14×14 空间图，再接 `lsp_pose_opt2.yaml` 已训练的
`DeconvPoseHead`（14→28→56 两级可学习反卷积），输出 14 张 56×56 关键点热图。
它不是只拿 patch embedding，也没有绕过 Qwen Vision。

该直接大模型 baseline 已在 RTX 5090 D 上用完整 1000 张官方 test 重测：PCK@0.2
**0.7217**、PCKh@0.5 **0.8846**、NME **0.2084**。从第一个原生 Vision block 到
14 张 56×56 热图的 mean/median/P95 为 **9.504/9.470/9.632 ms/image**；读出头为
`DeconvPoseHead`，1,102,990 个可训练参数，Qwen 参数全部冻结。

## 两个正式 profile

- `main_dc20`：一次光 Router CCD 把样本送给 Top-2/4 个 224×224 专家；
  随后还有一层 478×478 global phase，视觉分支共两次 feature CCD。
- `d2nn_dc20`：无 Router、无空间专家布局；使用两张 224×224 dense phase，
  中间保留 CCD 归一化和电子重载。每张图激活的相位参数量严格等于主方法
  Top-2 专家，即 `2×224²=100,352`。

两支路先分别按每个样本的有效 token/channel RMS 同尺度化，再用
`(1-alpha)E + alpha O` 融合并恢复电子支路 RMS，避免电子数值范围淹没光支路。
训练使用周期性 test（epoch 1、每 5 epoch、最终 epoch），按 PCK@0.2 最大值
选择 best；这是明确的数据使用协议，不把 test 描述成独立封存测试。

## 服务器命令

从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile main_dc20 --phase all

CUDA_VISIBLE_DEVICES=1 python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile d2nn_dc20 --phase all
```

仅关闭输入、相位与 CCD 三类像素平移扰动，同时保留光 Router、DC20、强度噪声、
phase dropout、k-space 和同尺度融合的 100-epoch 定位误差消融：

```bash
CUDA_VISIBLE_DEVICES=0 python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile main_dc20_no_shift --phase all
```

若从随机公共初始化训练的无位移版本仍明显低于历史模型，可运行兼容 warm start。
它复用旧 0.7131 模型中形状完全一致的二维 mixer、姿态头及 feature/global phase，
但不会加载旧电子 gate；光 Router 相位始终重新初始化并训练，物理光路和 DC20 条件不变：

```bash
CUDA_VISIBLE_DEVICES=1 python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile main_dc20_no_shift_warmstart --phase all
```

每个正式 run 只保留：

- `best_checkpoint.pt`：周期性 test PCK@0.2 最佳的 EMA 权重；
- `last_checkpoint.pt`：最终 epoch 的 live 权重；
- `metrics/training_history.csv`：完整曲线；
- `selected_checkpoint_test_evaluation.json` 和逐样本预测；
- `best_visualization/`：由 best 权重直接绘制的相位图与统计。

不要在正式 run 内保存每 5 epoch 的 PT。若以后研究相位演化，应另建明确标为
analysis 的 run。

训练完成后生成汇总表和论文图：

```bash
python -m LightGenV2.tasks.t02_keypoint_detection.report \
  --main LightGenV2/tasks/t02_keypoint_detection/runs/simulation/moe_router_scale_dc20_seed42 \
  --d2nn LightGenV2/tasks/t02_keypoint_detection/runs/simulation/d2nn_matched_dc20_seed42
```

## 正式结果

100 epoch、seed 42 的 DC20 单次复跑已经完成。两组都在 epoch 100 取得最高
周期 test PCK@0.2：光 Router Top-2 主方法为 **57.73%**（PCKh 73.63%，NME
0.3488），参数匹配的普通 D2NN 为 **67.51%**（PCKh 80.54%，NME 0.2736）。

历史目录中还能看到 PCK 0.6024 和 0.7131；它们分别缺少当前 DC20 条件，或使用旧
16 µm/电子 gate/非同尺度融合架构，不能替代当前正式结果。旧 0.7131 运行实际同样是
478×478、4 专家 Top-2，其父工程的 `moe16` 名称不代表运行时架构。完整核查见
[`reports/LSP_METRIC_AUDIT.md`](reports/LSP_METRIC_AUDIT.md)。
因此本次 LSP 协议下，普通 D2NN 明确优于光 Router MoE 9.78 个百分点；不能把
历史不同协议的 71.3% 候选混入本表。结果表、论文图和最佳相位总览见
[`reports/dc20_comparison/RESULTS.md`](reports/dc20_comparison/RESULTS.md)。
