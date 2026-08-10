# LSP Pose Optical MoE16 — Router 优化尝试记录

日期：2026-08-09
GPU：0（NVIDIA RTX 4090）
运行目录：`runs/lsp_pose_optical_moe16_opt`
训练日志：`/tmp/lsp_opt_train.log`

---

## 1. 背景

`qwen3_vl_embedding_2b_lsp_pose_optical_moe16` 学生模型首次完整训练（100 epoch）后：
- test PCK@0.2 torso = **0.323**，PCKh@0.5 = **0.564**，mean pixel error = **36.1px**
- 电子 teacher（epoch 37）：PCK = 0.511，PCKh = 0.797，MPE = 18.3px
- 学生 train loss 停滞在 0.064（teacher 为 0.033），收敛极慢

## 2. 诊断（tools_router_diagnostic.py，实测 checkpoint）

用已训练的 checkpoint 跑 4 个 batch 并分解 router 梯度：

| 梯度来源 | 大小（L2/batch） | 说明 |
|---|---|---|
| task（heatmap+coord） | **0.0015** | 几乎为零 |
| balance 损失 | **0.1441** | 主导，是 task 的 ~96 倍 |
| importance 损失 | 0.0001 | 可忽略 |
| phase_dc | 0.0000 | — |

**结论**：
- router 收不到有效任务信号，只被 balance 损失"钉"在均匀分布（normalized entropy = 1.0）。
- 根因：OEO 转换中的 **per-expert LayerNorm** 会把 router 的振幅增益梯度归一化掉（代码注释明确写了 "Per-expert LayerNorm removes the incoming amplitude scale"），唯一存活的梯度路径（归一化后重新乘以 routing weights）信号太弱。
- 结果：MoE 退化成近随机的 top-4 固定路由，16 个专家中只有 ~6 个被实际用到。

## 3. 优化方案（configs/lsp_pose_opt.yaml）

| 改动 | 原值 | 新值 | 目的 |
|---|---|---|---|
| router gate 初始化 std | 0.01 | **0.3** | 打破均匀吸引子，给 router 初始差异化 |
| router 温度 | 1.0 | **0.5** | 放大路由决策的影响力和梯度 |
| noisy top-k gating | 无 | **noise_std=0.3**（线性退火到 0） | 训练期强制探索，打破对称性（Shazeer et al.） |
| router 学习率 | 0.0005 | **0.004** | 与相位学习率对齐，加速 router 学习 |
| balance 权重 | 0.03 | **0.01** | 减少"钉在均匀"的压力 |
| importance 权重 | 0.005 | **0.002** | 同上 |
| phase_dc 权重 | （原运行未用） | **0.0** | 原 checkpoint 没有该损失；加上会淹没任务信号并破坏已收敛相位 |
| 训练方式 | 从头 | **从 epoch93 checkpoint 恢复 + 重初始化 router** | 保留已收敛的相位面和 head，省 ~3 小时 |
| student_epochs | 100 | **150** | 给噪声退火和 router 分化留足时间 |

## 4. 代码改动（全部向后兼容）

| 文件 | 改动 |
|---|---|
| `experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/optics/router.py` | `InputTopKRouter` 增加 `noise_std`、`gate_init_std` 参数（默认 0.0/0.01 保持原行为）；训练时给 logits 加高斯噪声；新增 `set_noise_std` |
| `.../optics/moe.py` | 用 `getattr(settings, ..., 默认值)` 把新参数传给 router（其他实验不受影响） |
| LSP `settings.py` | 新增 `router_noise_std`、`router_gate_init_std`、`resume_student_checkpoint`、`reinit_router`、`interlayer_detector_integration_factor`（默认 1，修复共享 moe.py 与 LSP settings 的兼容问题） |
| LSP `training.py` | `train_student` 支持从 checkpoint 恢复 + 重初始化 router；噪声按 epoch 线性退火到 0；日志/CSV 增加 `router_entropy` 列 |
| 新配置 | `configs/lsp_pose_opt.yaml`、`configs/lsp_pose_opt_smoke.yaml` |

## 5. 验证与结果

**Smoke 测试**（32 train / 16 test，1 epoch，恢复+重初始化 router）：
- entropy：1.0 → **0.31**（router 开始分化）✓
- test PCK = 0.326 / PCKh = 0.616（16 样本子集，原最优 0.323/0.564）✓
- 损失回到任务主导尺度（0.086，phase_dc=0）✓

**正式训练**：已在 GPU 0 启动（150 epoch，预计 ~4.5 小时），输出到 `runs/lsp_pose_optical_moe16_opt`。

## 6. 待验证指标（训练完成后）

- [ ] router entropy 是否持续 < 1（不再退化回均匀）
- [ ] 各 expert 的 load 是否均衡（16 个专家是否都被用到）
- [ ] test PCK 是否超过原最优 0.323 / PCKh 0.564
- [ ] 与 teacher（0.511/0.797）的差距是否缩小

## 7. 复现命令

```bash
# smoke
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt_smoke.yaml \
  --phase student_train

# 正式训练
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt.yaml \
  --phase student_train

# 推理（完成后）
CUDA_VISIBLE_DEVICES=0 python -m experiments.qwen3_vl_embedding_2b_lsp_pose_optical_moe16 \
  --config experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/configs/lsp_pose_opt.yaml \
  --phase student_inference
```

## 8. 诊断脚本

`tools_router_diagnostic.py`：加载已训练 checkpoint，分解 router 梯度（task vs balance vs importance），用于复现诊断。

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/tools_router_diagnostic.py
```
