# 冻结 Qwen baseline：RTX 5090 D 合同

必须区分两种实验，不能把它们的数值写在同一 baseline 行：

1. `baseline_5090d.py` 是零样本自由生成诊断。它让 Qwen 自回归生成 JSON，包含
   生成与 CPU JSON 解析；解析失败必须保留。它不能代表正常的任务 baseline。
2. `baseline_structured_5090d.py` 是论文表使用的正常 baseline。它冻结 Qwen 原生
   Vision 与 Language Transformer，只训练一个普通的结构化 `6×6` 任务读出头，
   不使用 LoRA、不增加 Transformer、不做数据增强。

## 模型与性能

- 模型：`Qwen/Qwen3-VL-2B-Instruct`，Vision/Language 权重全部冻结。
- 输入是同一 OpenMoji 图像和完整文本指令，使用同一个 5,000/1,000 split contract。
- 正常 baseline 仅训练结构化任务读出头，输出 `6×6` 类别网格和编辑网格；报告
  changed-cell accuracy、foreground category accuracy、edit-grid IoU、object F1 和
  scene exact match。
- 特征缓存只用于冻结主干的读出头训练加速，不参与正式推理与计时，不改变数值。
- 零样本诊断若自由文本不能稳定解析，必须单独报告 parse-failure rate，不得删掉失败样本。

## 速度

- GPU：RTX 5090 D；记录驱动、CUDA、PyTorch、精度、功耗上限和 batch size。
- 计时起点：图像/文本 hidden state 即将进入第一个原生 Transformer block。
- 正常 baseline 计时终点：结构化任务头产生 `6×6` 类别与编辑结果；包含全部原生
  Vision/Language Transformer blocks 和任务读出头，不包含自回归生成或 JSON 解析。
- 零样本诊断另行包含自回归生成与 CPU JSON 解析。
- 不计文件读取、PNG 解码、tokenizer 和 block 之前的 embedding。
- batch=1，模型只加载一次；先显式 warm-up 50 次且不统计，再对完整 1,000 条 test
  连续计时。CUDA Event 与同步 host 计时均保留，报告 mean、median、P5、P95。

## 功耗

- 与速度同一次测试，用 NVML 至少 20 Hz 采样整卡功率。
- 报告 idle、mean、peak，以及扣除 idle 后的 `J/sample`；不能使用 TDP 代替实测值。

运行入口：

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.baseline_structured_5090d all \
  --model /path/to/Qwen3-VL-2B-Instruct \
  --data-root /path/to/pilot_gpu \
  --cache-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/qwen_structured_5090d/cache \
  --run-dir LightGenV2/tasks/t04_semantic_interaction/runs/simulation/qwen_structured_5090d
```
