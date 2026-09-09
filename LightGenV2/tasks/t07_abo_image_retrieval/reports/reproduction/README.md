# t07_abo_image_retrieval 复现说明入口

## 2026-09-09 新一轮图搜图（六次光传播）

环境：服务器 `/home/guest3/miniconda3/envs/xml/bin/python`；需要 PyTorch CUDA、transformers、Pillow、PyYAML、numpy、matplotlib。
数据保持 `/DATA/DATA1/guest3/2026OpticsMoE/data/abo_similarity10_data`，不移动原数据。
Qwen 与 warmstart 的服务器绝对路径在 `configs/optical_top2_dc20.yaml`，其他电脑先改这两个位置及 dataset_root。
代码必须先从 GitHub checkout 对应 commit；以下从仓库根目录执行。GPU 编号需按空闲显存调整。

```bash
python -m unittest discover -s LightGenV2/tasks/t07_abo_image_retrieval/tests -v

# 1. 重跑冻结大模型，自动输出 native/square × 2048/64D 四组结果及教师缓存。
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run \
  --mode baseline \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/frozen_qwen_20260909

# 2. 首次先做一轮小检查（评估仍覆盖完整图库/测试集）。
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run \
  --mode optical --epochs 1 --steps 2 --eval-interval 1 \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/smoke/optical_top2_dc20_20260909

# 3. 正式训练；不能复用已有 checkpoint 的 run-dir。
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run \
  --mode optical --epochs 40 \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/optical_top2_dc20_20260909
```

baseline 看 `baseline_report.json`，光学看 `history.json`/`status.json`/`final_report.json`；
最后的相位图为 `best_phase_overview.png`。run_manifest 记录源码 commit、环境、原命令和数据清单 SHA。
`features.pt` 是固定教师缓存，不是可部署的光学权重；部署/后续微调用 `best_checkpoint.pt`。
baseline 2048D 原长宽比的历史 Hit@1=95.2083%，本轮是否复现必须以新报告为准。
训练仅用 train；test 每 5 epoch 参与选模，明确不作为独立泛化估计。

## 历史审计说明（保留）

[Baseline????????](BASELINE_METHODS.md)?????baseline??????????????????????2026-09-09?

本目录集中保存baseline及主方法的可复现性证据；本次只建立入口，**尚未进行本任务的新一轮复现**。
当前任务结构和已有结果见 [任务README](../../README.md)。不得因为存在本文件就声称已复现。

后续每个正式结果需在这里记录：

1. baseline定义，冻结/训练的参数，预处理、输出头与主方法差异。
2. 原始数据版本、train/test清单与SHA256、标签生成方法、指标与选模口径。
3. 源码commit、完整命令、依赖环境、模型及checkpoint的SHA256和获取位置。
4. 固定权重复评与从头重训分别报告；标明样本数、seed、运行ID、结果和误差。
5. 速度/能耗的硬件、计时边界、功率积分口径；未测的不得填估计值冒充实测。

原始日志及逐样本结果留在本任务runs，文档只引用。参考 [SALICON复现说明](../../../t03_saliency/reports/reproduction/README.md)。
