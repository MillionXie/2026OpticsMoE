# LSP：同结构分阶段续训（2026-09-09）

状态：方案及执行入口。新结果以各 run 的 `final_report.json` 为准，不预填提升。

## 已核实的起点与结构

目前可追溯的光学起点为 `moe_router_scale_dc20_no_shift_warmstart0713_seed42`，
epoch 75 的 EMA，1000 张 LSP test PCK@0.2 = **0.7305**。
SHA256：`fc1c6be4196593d7f27d83097c5bfd53b6e0511e84726fd55d712af0a4cd7741`。
此前表格里的 0.7983 不能作为已验证的 LSP 结果；已找到的对应数值来自 ABO 检索。

| 部分 | 当前光学 MoE | 当前大模型 baseline |
|---|---|---|
| 输入 | 标注人体框裁剪、224×224；冻结 Qwen 图像嵌入 | 相同任务协议，224×224 人体裁剪 |
| 原生 Qwen Vision Transformer | 不执行；用两级光电融合代替 | 完整执行且冻结，不是只用 embedding |
| 中间表示 | 196×1024 → 196×192 | 原生视觉输出 196×1024 |
| 光学 | 光 router Top-2/4 → 2×2 专家 → global phase | 无 |
| 电子融合 | 每级 depthwise/pointwise 空间 mixer + 192→384→192 MLP；与光支路同 RMS 尺度后 `(1-alpha)E+alpha O` | 无光电残差融合 |
| 姿态头 | 192→160，轻量卷积+双线性上采样 14→28→56，133,425 参数 | 1024→128，卷积+两级可学习反卷积，1,102,990 参数 |
| 输出 | 14×56×56 关节热图，argmax 得坐标 | 相同输出形式 |
| Language | 不执行 | 不执行 |
| 已有 PCK | 0.7305 | A100 固定权重重评约 0.7226 |

差距只有约 0.79 个百分点。头、初始化和选模不同，不能直接归因于光学优越性。
baseline 历史为训练 loss 选模；光学按周期 test 选 EMA。
两者使用标注关节点导出的身体裁剪，不是无检测器的整图多人姿态任务。
PCK 的 torso 尺度为工程既有定义；不要与不同论文的 PCK/PCKh 定义混用。

光场：532 nm、17 μm 采样、10 cm；专家 224×224，2×2 排在 478×478 有效场，
FFT 外围 padding 不等于实际播放区域。光 router 一次 CCD，专家和 global 各一次 CCD。
alpha 范围沿用 [0.01, 0.95]，继承已训练值，不重新初始化。
保留 DC20–30% 相干零级、原有强度/偏置/读噪声、k-space 和 phase dropout；
沿用起点的 no-shift 设定，因此本轮不能声称改善了位移鲁棒性。

## 三个训练对照（不新增电子层）

统一从上述同一个 best 完整 strict-load `core/head`，新建 AdamW，seed 42，
batch 24，完整 10,428 张训练数据，每个方案续训 60 epoch。
测试 1000 张：训练前、第 1、每 5、最后 epoch；按 PCK 最大选 EMA，
同分依次按 NME、loss、早期 epoch。保留训练前 checkpoint 为候选，防止续训退化后覆盖原成果。
test 明确参与选择，不称作独立封存测试。

- `joint`：连续联合续训对照。LR：电子 2e-5、router 3e-4、feature phase 3e-3、
  CCD readout 5e-5、pose head 1e-4，余弦降至十分之一。
- `staged`：1–5 epoch 只训姿态头；6–15 冻结电子 mixer 等，只训光学/router、CCD readout、头，
  feature phase LR 6e-3；16–50 联合低 LR；51–60 固定光学与电子主干，只微调 CCD readout 和头。
- `staged_heatmap`：同一阶段计划，仅把**训练**中的坐标 SmoothL1 辅助项权重从 0.1 设为 0。
  保留 Gaussian heatmap MSE 和既有光学/router 正则；评估仍用原定义。
  这是明确的 loss 对照，不与“仅训练阶段变化”混称。

每 epoch 记录分组 LR、解冻参数数、raw phase 变化、`2pi*sigmoid(raw_phase)` 物理相位 RMS 变化。
测试时记录两个 alpha。最终保存 best/last、曲线、源码 commit、原权重 SHA、完整数据清单。
最终额外关闭光支路再测一次：这是固定模型的反事实消融，不是重新训练的电子 baseline。

## 实验室执行

从已提交的独立源码 worktree 根执行。不要在其他同学的脏 worktree 改文件。
实际资产在原工程；下面使用已确认存在的实验室离线 cache。
迁移机器时先修正 `--cache-dir`，不要联网换模型。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/home/guest3/miniconda3/envs/xml/bin/python
ROOT=/DATA/DATA1/guest3/2026OpticsMoE
TASK=$ROOT/LightGenV2/tasks/t02_keypoint_detection
# 先用未被占用的 GPU，且 run-dir 必须是全新目录。
CUDA_VISIBLE_DEVICES=1 $PY -m LightGenV2.tasks.t02_keypoint_detection.refine \
  --profile staged \
  --source "$TASK/runs/simulation/moe_router_scale_dc20_no_shift_warmstart0713_seed42/best_checkpoint.pt" \
  --data-root "$ROOT/data/lsp_pose" --cache-dir /DATA/DATA1/guest3/.cache/huggingface/hub \
  --run-dir "$TASK/runs/simulation/refinement_20260909/staged" \
  --batch-size 24 --workers 4
```

另外两组只改 `--profile joint` / `staged_heatmap`，并使用不同 run-dir。
`--smoke --batch-size 4 --workers 0` 仅检验 4 train/4 test、1 epoch，不得作为正式性能。
先运行 `python -m pytest LightGenV2/tasks/t02_keypoint_detection/tests -q`。

2026-09-09 实验室 `xml` 环境已通过 12 项测试和 4 train/4 test 的完整单步 smoke，
包括 backward、EMA、选模、重载、关光评估。单步 feature/global 物理相位 RMS 变化约
0.00116–0.00172 rad，确认相位确实被更新；smoke 的准确率没有统计意义。

训练结束先读 `source_anchor_test.json`（核对起点），再读 `final_report.json`。
中途查看 `training_history.json`，不要把 live-last 当 best。
本轮不承诺达到 79.83%，也不改 baseline 来人为扩大差距。

## 后续电子结构实验的边界

先完成上述同结构实验，再决定是否尝试与 baseline 同款 Deconv head（约增加 97 万头参数），
或只改变上采样方式。必须单列参数量、头延迟、PCK/PCKh/NME、关光损失，
不可把加大电子头的收益都称为光学收益。当前轮次不实施这些结构改动。
