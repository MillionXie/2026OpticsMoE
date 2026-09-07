# Caltech101：固定专家库生成

用户已批准首轮固定专家库实验。实现与结果只在本任务维护，不写回 LightGenV2 的正式对照表。

## 首轮协议

- 光路直接复用 LightGenV2 T01 main_dc20；vision/language 各四专家 Top-2，Router/global 正常直接优化。
- A direct；B small_hyper；C qwen_frozen；D qwen_lora。生成器只接收固定任务/模态/expert 条件。
- D 使用独立 Qwen3-VL-2B-Instruct 的语言 Transformer，所有 q_proj/v_proj 挂 rank=8、alpha=16 LoRA。
  Qwen 基座冻结，LoRA 与 float32 patch 解码器训练；不改变原任务前端。
- 使用 `anchor + decoder(context) - initial_reference` 生成 raw phase。anchor 是同 seed 的固定随机相位，
  initial_reference 是训练前生成值，二者均为不可训练 buffer。因此四组初始光学相位完全一致；
  后续没有独立可训练的 expert 像素参数（A 除外）。这替代草案中的 B/C 初始化拟合。
- 经 torch parametrization 将生成 tensor 接入原 PhaseLayer，不改变 FFT、相位转换、DC 正则和噪声实现。
- 复用原 2,625 train / 30 gallery / 200 test 划分，再仅从 train 固定抽每类 30 张，合计 300 张。
  PK=10×3；5 epoch，默认每 epoch 10 optimizer steps。此轮是 pilot，不能和全量正式结果混表。
- 任务 loss 与 backend 一致：supervised contrastive + episodic prototype，附原路由/CCD/DC 正则。
- 只在最后一轮评估 EMA，不按 test 选择 checkpoint；`best_checkpoint.pt` 明确表示最终 EMA。
  `last_checkpoint.pt` 为 live 可恢复状态；保留 `expert_bank.pt` 独立部署相位。
- 记录 task loss 到 expert / LoRA 的非零梯度、显存、训练日志，以及移除生成器后的预测一致性。

## 运行

从仓库根目录，在已有 xml 环境运行；GPU 由外部环境选择。

```bash
python -m unittest discover -s TransferFromElectricity/tasks/t01_object_retrieval/tests -v
CUDA_VISIBLE_DEVICES=3 python -m TransferFromElectricity.tasks.t01_object_retrieval.run \
  --method qwen_lora \
  --run-dir TransferFromElectricity/tasks/t01_object_retrieval/runs/simulation/YYYYMMDD_static_qwen_lora_pilot_s42
```

相同命令替换 method 为 direct / small_hyper / qwen_frozen，使用独立 run-dir。
`--generator-source` 可指定本机缓存 snapshot；默认通过已有模型缓存解析。
`--epochs 1 --steps-per-epoch 2` 是实际光路短检查，用独立 runs/smoke 目录。
中断后同配置加 `--resume`，只读取 last；恢复要求相同代码 commit。

## 证据与限制

每 run 保存完整 backend config、pilot 配置、样本清单、Git SHA、初始化 checkpoint SHA、
环境、原始命令、实际参数组、LoRA 模块名、loss/梯度、最终指标与导出误差。
固定 anchor 与初始生成值仅用于相同起点，不是从已训练 baseline 蒸馏。
小生成器与 Qwen 的总参数/训练成本不相同；本轮仅按相同样本与 optimizer steps 比较。
CUDA 仿真耗时不是物理光路延时；生成器仅在训练和导出使用。
真实 SLM/CCD 闭环不属于本轮。

尚未完成真实数据实验；后续结果在本节和 reports/ 中补充。
