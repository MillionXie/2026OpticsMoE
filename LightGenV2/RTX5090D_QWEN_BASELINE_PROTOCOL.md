# RTX 5090 D 冻结 Qwen baseline 待测协议

本文只定义论文表格中“大模型 baseline”的统一测量边界，不在当前共享训练服务器填写任何推测值。
SALICON 与 OpenMoji 的具体命令和输出格式分别见任务目录中的 `BASELINE_5090D_TODO.md`。

## 三组结果各自测什么

1. `Ours`：光学 Router、Top-2 专家和同尺度光电融合；报告仿真任务性能。论文采用的六次光传播核心时间暂记为 9.084 ms，实验台端到端时间和系统功耗另行实测。
2. `D2NN`：移除 Router 的普通 dense D2NN，激活相位参数预算与 Ours 对齐；当前服务器只测任务性能，不把 CUDA 光学仿真耗时当成物理光路速度，也不测其 GPU 功耗。
3. `Frozen Qwen`：Qwen 主干权重完全冻结；性能、速度和功耗全部留到同一台 RTX 5090 D 测量。需要任务读出头时只能训练读出头，必须单独报告其结构和参数量。

## 各任务大模型 baseline

| 任务 | 冻结模型 | 输出及性能指标 | 状态 |
| --- | --- | --- | --- |
| T01 Caltech101 检索 | `Qwen3-VL-Embedding-2B` | 归一化 embedding、固定 gallery 排序；Top-1/Top-3/MRR | 已有性能候选，5090 D 速度/功耗待统一复测 |
| T02 LSP 关键点 | `Qwen3-VL-Embedding-2B` + 轻量关键点读出头 | PCK@0.2、PCKh@0.5、NME；Qwen 不微调 | 5090 D 待测 |
| T03 SALICON 显著性 | `Qwen3-VL-Embedding-2B` + 轻量显著性读出头 | CC、KLD、SIM、NSS、AUC-Judd、MAE；Qwen 不微调 | 5090 D 待测 |
| T04 OpenMoji 语义交互 | `Qwen3-VL-2B-Instruct`，完整图像与文本指令 | 解析 6×6 语义/编辑网格；Changed、Category、IoU、F1、Exact、解析失败率 | 5090 D 待测 |
| T06 LGVQ 视频质量 | `Qwen3-VL-2B-Instruct`，任务对应 prompt | 单一 Spatial 或 Temporal MOS；SRCC/KRCC/PLCC/RMSE/MAE | 既有值保留为历史证据，最终表需按本文边界复测 |

ABO 的 T07/T08 按当前要求暂不纳入；T05 数据协议未冻结，不先造 baseline 数字。

## 速度边界

- 设备固定为 RTX 5090 D，`batch=1`；记录驱动、CUDA、PyTorch、精度模式和功率上限。
- 起点：已经形成 block 输入 hidden states，即将进入第一个原生 Transformer block。
- 终点：任务结果已经形成。检索任务包含 query embedding 归一化以及对预先固定 gallery embedding 的相似度与 Top-K；像素任务包含读出头；生成任务包含生成和结构化解析。
- 不计文件读取、图像/视频解码、tokenizer、patch/token embedding 以及第一个 block 之前的工作。
- 论文表统一采用受控热稳态口径：模型只加载一次，显式 warm-up 50 次且不统计，然后
  连续测量 200 条 test；若任务完整 test 本身只有约 1,000 条且性能依赖同一次自回归
  生成（如 OpenMoji），则 warm-up 后连续测完整 test。报告 mean、median、P5、P95，
  单位统一为 `ms/sample`，不得剔除首条计时样本或其他慢样本。
- 不允许把单个 block 的耗时、模型加载时间或跨任务旧日志填入该列。

## 功耗边界

- 与速度在同一批推理中测量，板卡 `power.draw` 采样频率至少 20 Hz；采样只在模型
  推理窗口内记为 active，图像/视频解码间隙不得稀释 active mean。
- 先记录稳定 idle 功率，再报告 active mean、peak 和扣除 idle 后的 `J/sample`：
  `energy/sample = integral(P_active - P_idle) dt / N`。
- 不得用 TDP 代替实测，不得把共享服务器或不同型号 GPU 的结果换算成 5090 D 数值。

## 待填结果合同

每个任务保存一个 JSON，未测字段必须是 `null`，不能写 0：

```json
{
  "status": "pending_5090d_measurement",
  "model": "exact model id",
  "frozen": true,
  "performance": null,
  "speed_ms": null,
  "power": {
    "idle_w": null,
    "active_mean_w": null,
    "peak_w": null,
    "incremental_j_per_sample": null
  },
  "hardware": "NVIDIA GeForce RTX 5090 D"
}
```

结果落盘时必须同时保存 train/test split SHA256、命令、Git commit、checkpoint SHA256 和原始逐次时间/功率记录，表格里的每个数值都应能回溯到这些证据。

## 5090 D 正式入口

下列命令均为 `batch=1`，并要求输出目录位于对应任务自己的
`runs/simulation/<run_id>`。路径根据 5090 D 数据盘实际位置替换：

```bash
# T01 Caltech101：单图检索
python -m LightGenV2.tasks.t01_object_retrieval.baseline_5090d \
  --model /path/Qwen3-VL-Embedding-2B --data-root /path/data/Caltech101 \
  --run-dir LightGenV2/tasks/t01_object_retrieval/runs/simulation/qwen_frozen_5090d

# T02 LSP：单图关键点
python -m LightGenV2.tasks.t02_keypoint_detection.baseline_5090d \
  --model /path/Qwen3-VL-Embedding-2B --data-root /path/data/lsp_pose \
  --config /path/lsp/resolved_config.yaml --checkpoint /path/lsp/teacher_best_train_loss.pt \
  --run-dir LightGenV2/tasks/t02_keypoint_detection/runs/simulation/qwen_frozen_5090d

# T03 SALICON：单图显著性
python -m LightGenV2.tasks.t03_saliency.baseline_5090d \
  --model /path/Qwen3-VL-Embedding-2B --data-root /path/data/SALICON \
  --config /path/salicon/resolved_config.json --checkpoint /path/salicon/teacher_best.pt \
  --run-dir LightGenV2/tasks/t03_saliency/runs/simulation/qwen_frozen_5090d

# T04 OpenMoji：单图 + 文本指令，自回归输出两个 6x6 网格
python -m LightGenV2.tasks.t04_semantic_interaction.baseline_5090d \
  --model /path/Qwen3-VL-2B-Instruct --data-root /path/openmoji/pilot_gpu \
  --run-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/qwen_frozen_5090d

# T06 LGVQ Spatial：每视频 4 帧、五质量词读出头
python -m LightGenV2.tasks.t06_video_quality_assessment.quality_token_resolution \
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/baselines/qwen3vl_spatial_quality_tokens_4f_r448.yaml \
  --phase all --model /path/Qwen3-VL-2B-Instruct --manifest /path/lgvq_split.csv
```

每个报告同时给出实测 `active_mean_w` 和 RTX 5090 D 的 575 W 额定上界，并由同一
任务的平均时延分别换算实测能量与额定上界能量。额定值只能标为 theoretical upper
bound，不能写成平均功率。
