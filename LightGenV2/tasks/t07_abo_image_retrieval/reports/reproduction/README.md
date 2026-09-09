# t07_abo_image_retrieval 复现说明入口

## 2026-09-09 新一轮图搜图（六次光传播）

环境：服务器 `/home/guest3/miniconda3/envs/xml/bin/python`；需要 PyTorch CUDA、transformers、Pillow、PyYAML、numpy、matplotlib。
使用 GPU 数字编号前设置 `export CUDA_DEVICE_ORDER=PCI_BUS_ID`，或直接以 GPU UUID 设置 `CUDA_VISIBLE_DEVICES`，避免编号与 nvidia-smi 不一致。
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

# 4. 可选的同结构训练对照：只加强训练集语义中心/KD监督，不新增推理头。
CUDA_VISIBLE_DEVICES=4 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run \
  --mode optical --epochs 40 \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/optical_top2_dc20_anchor.yaml \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/optical_top2_dc20_anchor_20260909

# 5. 固定 best 重评，不重新训练，不需要教师特征缓存。
CUDA_VISIBLE_DEVICES=1 python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run \
  --mode evaluate \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/optical_top2_dc20_20260909/best_checkpoint.pt \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/optical_top2_dc20_reeval_20260909
```

baseline 看 `baseline_report.json`，光学看 `history.json`/`status.json`/`final_report.json`；
最后的相位图为 `best_phase_overview.png`。run_manifest 记录源码 commit、环境、原命令和数据清单 SHA。
`features.pt` 是固定教师缓存，不是可部署的光学权重；部署/后续微调用 `best_checkpoint.pt`。
baseline 2048D 原长宽比的历史 Hit@1=95.2083%，本轮是否复现必须以新报告为准。
训练仅用 train；test 每 5 epoch 参与选模，明确不作为独立泛化估计。

## 继续优化：训练方法 / 电子增强的配对对照

两组都从 `optical_top2_dc20_anchor_20260909/best_checkpoint.pt` 开始（SHA 在配置中强制校验），
重新初始化优化器。固定旧 baseline、训练/测试名单、全 120 商品检索、64D 输出和光路。
只保存 best/last；80 epoch，每 5 epoch 完整测试选 best，不使用验证集。

```bash
# 先同步本次 Git commit；GPU UUID 按空闲显存选择，建议留出至少 10 GB。
python -m unittest discover -s LightGenV2/tasks/t07_abo_image_retrieval/tests -v

# 训练方法组；原推理结构不变。
python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_training.yaml

# 相同训练方法，再扩大电子残差卷积核 + 非线性读出。
python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_electronics.yaml

# 增强模型重评必须传其对应配置，不能用旧的默认线性头配置。
python -u -m LightGenV2.tasks.t07_abo_image_retrieval.run --mode evaluate \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_electronics.yaml \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/refine_electronics_20260909/best_checkpoint.pt \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/refine_electronics_reeval_20260909
```

训练 batch=20：十类各取两个不同训练商品的一张图，48 step/epoch（随机采样，非保证全覆盖）。
轻增强先做与测试一致的方形裁剪，再做 0.94–1.0 裁剪和 ±5% 亮度/对比度；不翻转。
KD 使用同一训练图片的干净视图缓存，明确属于增强一致性目标，不是重新执行教师。
5 epoch 预热 + 余弦学习率，KD 从 0.5 降至 0.1；监督对比/训练类中心监督保留。
电子增强仅在原两路残差内扩大卷积核至 9，末端读出增加 512 隐层的 GELU 修正，
无 attention/Transformer、无教师推理旁路。卷积新增系数、读出修正输出初始化为零，
初始函数保持旧权重行为（浮点运算误差除外）。真实 kernel、读出参数和活跃图见 architecture.json。
仍保留光 Router Top-2、同尺度融合、20%–30% 未调制训练分量、既有 CCD 噪声；像素偏移仍为零。

## 历史审计说明（保留）

[历史 baseline 方法审计](BASELINE_METHODS.md)保留了旧运行的模型、预处理及评估定义。

本目录集中保存 baseline 及主方法的可复现性证据；2026-09-08 建立入口时尚未复现，2026-09-09 的新一轮结果见上方说明与 [本轮证据](RUN_20260909.md)。
当前任务结构和已有结果见 [任务README](../../README.md)。不得因为存在本文件就声称已复现。

后续每个正式结果需在这里记录：

1. baseline定义，冻结/训练的参数，预处理、输出头与主方法差异。
2. 原始数据版本、train/test清单与SHA256、标签生成方法、指标与选模口径。
3. 源码commit、完整命令、依赖环境、模型及checkpoint的SHA256和获取位置。
4. 固定权重复评与从头重训分别报告；标明样本数、seed、运行ID、结果和误差。
5. 速度/能耗的硬件、计时边界、功率积分口径；未测的不得填估计值冒充实测。

原始日志及逐样本结果留在本任务runs，文档只引用。参考 [SALICON复现说明](../../../t03_saliency/reports/reproduction/README.md)。
