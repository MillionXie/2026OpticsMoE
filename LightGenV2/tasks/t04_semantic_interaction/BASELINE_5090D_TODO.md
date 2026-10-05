# 冻结 Qwen baseline：RTX 5090 D 合同

必须区分两种实验，不能把它们的数值写在同一 baseline 行：

1. `baseline_5090d.py` 是零样本自由生成诊断。它让 Qwen 自回归生成 JSON，包含
   生成与 CPU JSON 解析；解析失败必须保留。它不能代表正常的任务 baseline。
2. `baseline_structured_5090d.py` 是论文表使用的正常 baseline。它冻结 Qwen 原生
   Vision 与 Language Transformer，只训练一个普通的结构化 `6×6` 任务读出头，
   不使用 LoRA、不增加 Transformer、不做数据增强。

## 模型与性能

### 历史完整 Qwen 对齐监督变体（2026-10-06 源码恢复）

`baseline_structured_5090d.py --baseline-protocol aligned_label_only` 恢复旧A100
运行代码的可选监督协议：原结构化头加四类操作线性读出，使用原有类别、编辑、Dice、
保留格与任务损失；完整Qwen仍冻结。默认 `legacy` 不变，旧PT未记录协议时按普通头
严格复载。两种协议不能合成一条性能或测速记录；下方0.5475仍是原5090D普通头。
八项CPU协议测试、两种头CPU形状／损失／梯度测试通过；六个科学定义与归档源码
AST一致。没有重跑完整Qwen、实拍、训练或测速。迁移版保护已有best/last与报告，
记录实际GPU功率上限，数据准备及H2D放在计时区外；旧测速不转换不覆盖。
恢复身份见 [源码清单](structured_baseline_import_20261006.json)。

### 历史匹配电子对照（2026-10-06 源码恢复）

`benchmark_electronic_control.py` 与 `configs/qwen_matched_electronic_control.yaml`
是旧 A100 工程的四块纯电子任务对照，不是完整原生 Qwen baseline，也不是当前
rank64 光路模型。其计时使用预计算指令 hidden，不包含指令主干执行；没有显式预热，
第一条 TEST 计入计时。这些历史边界保留，不与上面的正式5090D数字混用。
迁移版拒绝缺项／多项权重和已存在输出目录，按实际可见GPU记录功率上限。
仅完成CPU协议检查，未运行模型、读取PT或重新测速；原始源码与恢复身份见
`electronic_control_import_20261006.json`。原测速和运行产物未修改。

### 历史 A100 零样本诊断源码收敛（2026-10-06）

`baseline_5090d` 已恢复实际服务器的 `--output-contract sparse_changes` 和
`--max-samples` 选项；二者仅为显式诊断，不替代正式结构化 baseline。默认仍为
`full_grid`、224输入、50次预热和 `--expected-gpu 5090`；A100诊断须显式传
`--expected-gpu A100`，不得沿用5090D测速数字。已有输出目录会在查询GPU前拒绝。
新报告使用schema 2明确标注诊断与实际可见GPU的功率上限；旧报告不转换、不覆盖。
8项无Torch／设备CPU协议测试通过，没有运行Qwen、训练、读取数据或测速。
源Git身份及投影SHA见 [恢复清单](generation_diagnostic_import_20261006.json)。

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

正式 5090 D run ID：`qwen_structured_native_visual_5090d_20260907`。1000 test 的
changed-cell accuracy 为 0.5475，mean/median/P95 为 27.166/26.628/30.280 ms；
完整结果和功率以 run 内 `baseline_report.json` 为准。
