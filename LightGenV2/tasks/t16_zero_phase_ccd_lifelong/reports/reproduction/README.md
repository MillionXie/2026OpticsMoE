# 多模态终身学习复现入口

版本事实以[任务 README](../../README.md)为准。最终三张矩阵、必要回放对照、
完整训练方法及解释边界见[正式报告](../FINAL_RESULTS_AND_TRAINING_20260926.md)。

实际运行源：`/DATA/DATA1/guest3/t12_cross_modal_20260920/LightGenV2/tasks/t16_zero_phase_ccd_lifelong`。
原 `runs/simulation`、每阶段 `config.json` / `command.txt` / `history.json` /
`selected_test.json` / `best_checkpoint.pt` / `last_checkpoint.pt` 保留服务器。
精确权重 SHA 必须从对应 `selected_test.json` / config 核验，不能只选“最新文件”。

正式主要身份：

- MoE C：`stage3_moe_balanced_sharedvision_s17_e696`，后续原D与Physical探索性D共用这个起点。
- D2NN 无 replay D：`stage4_d2nn_noreplay_sharedvision_s17_d9d5`。
- D2NN 全量 replay D：`stage4_d2nn_replay_balanced_sharedvision_s17_e031`。
- 独立 D2NN 固定4×4：`d2nn_fixed4x4_sharedvision_s17_d9d5`。
- 探索性 Physical D：`stage4_moe_physical_pairwise2_sharedvision_s17_e19a`。

重训前必须按原run命令绑定各数据协议JSON与共享视觉checkpoint（不是只执行默认参数）。
`train_lifelong_moe.py --help` 给出 `--eurosat/--clevr/--speech/--physical`、
`--previous-checkpoint`、`--vision-checkpoint`、阶段、回放方式、损失和读出学习率；
每阶段依赖前阶段，不能并行启动同一链四阶段。此轮整理不授权重训。

本地只验证迁移结构：

```powershell
# 使用已有包含torch/pytest的环境；全程CPU，不打开SLM/CCD
python -m pytest LightGenV2/tasks/t16_zero_phase_ccd_lifelong/tests -q
python maintenance/git_safety/check_task_registry.py
```

2026-10-02 本地29项通过；原数据矩阵没有重算，单种子及后验探索性限制不因迁移消失。
