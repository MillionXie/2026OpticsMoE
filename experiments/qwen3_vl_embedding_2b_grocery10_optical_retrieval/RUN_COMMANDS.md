# Grocery10：保留版本与硬件命令

所有命令从仓库根目录执行。当前只维护两套模型配置：

| 配置 | 结构 | 用途 |
|---|---|---|
| `grocery10_moe16_best.yaml` | 4×4、16 experts、Top-4 | 历史最佳，可加载 checkpoint Top-1 73.46% |
| `grocery10_moe4_latest.yaml` | 2×2、4 experts、Top-2、2×2 CCD integration | 当前实物鲁棒版本 |

## 训练

MoE4 从零训练：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe4_latest.yaml \
  --phase all
```

MoE16 从保留的 epoch-141 起点复现：

```bash
CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_optical_retrieval \
  --config experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/configs/grocery10_moe16_best.yaml \
  --phase all \
  --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/runs/qwen3_vl_embedding_2b_grocery10_replaced_continue_epoch141_stronger_augmentation_ema/checkpoints/pre_resume_epoch_0141/resume_checkpoint.pt
```

## 四平面硬件采集

仿真、真实采集、振幅/相位 SLM demo 已统一放在 [共享硬件命令](../hardware_sdk/RUN_COMMANDS.md)。

该流程按 `vision expert → vision global → language expert → language global` 运行；每层人工确认相位 mask，程序自动播放整批振幅、拍摄 CCD、输出 theory 对照、执行电子处理并生成下一层振幅。

## 测试

```bash
pytest experiments/qwen3_vl_embedding_2b_grocery10_optical_retrieval/tests -q
```
