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

## 2026-09-07 首轮结果

四组均已完成：300 train、30 gallery、200 test，seed=42，5 epoch / 50 updates，最终 EMA。
训练 commit：`56d7e6b66b9d820b64ad476ceccb5bff56f55947`；运行在服务器 GPU 3（RTX 4090）。
源码、数据划分和预算一致；本地轻量文件从服务器经 SFTP 同步并逐文件校验 SHA256。

| 方法 | Top-1 | Top-3 | MRR | 峰值显存 GiB |
|---|---:|---:|---:|---:|
| direct | 82.0% | 93.0% | 0.8860 | 7.43 |
| small_hyper | 82.0% | 93.0% | 0.8868 | 7.44 |
| qwen_frozen | 81.5% | 93.0% | 0.8843 | 10.64 |
| qwen_lora | 81.5% | 93.0% | 0.8835 | 11.25 |

机制验证通过：D 的任务 loss→expert 梯度范数 0.05483，任务 loss→LoRA B 梯度范数
0.0008387；最终 EMA expert 相位相对起点的 RMS 变化 0.01331 rad。
四组固化 expert 后输出误差均为 0，样本逆序误差为 0；单样本与 batch 推理最大 embedding
绝对差为 0.00223–0.00298（BF16 路径，低于预设 0.005 容差），不能表述为逐位一致。

结论仅为链路可训练、可导出；本轮没有观察到大模型生成优于直接优化。
D 与 A 的 Top-1 差距仅 1/200 个查询，单 seed 短训练不能据此判断方法优劣。
Language Router 四组均只使用前两个专家（训练选择次数 1500/1500/0/0），尚未解决路由集中。
相位变化较小且四组 loss 曲线非常接近，不能将 loss 下降全部归因于生成器。
小生成器首 batch task loss 与 A 相差约 0.00154，虽初始无梯度相位校验误差为 0，
该端到端微小差异尚未逐算子定位；B 保留为辅助对照。

完整数值、run ID 和证据哈希见 [summary.json](reports/pilot_20260907/summary.json)、
[evidence_manifest.json](reports/pilot_20260907/evidence_manifest.json)；
[对照图](reports/pilot_20260907/pilot_comparison.png) 给出每 epoch 平均 task loss 与最终 Top-1。
原始日志、配置和 sample manifest 在任务 runs 下；checkpoint 留在服务器同名 run。
后续优先诊断 Language Router 集中及生成相位的实际贡献，再考虑延长训练和多 seed。
