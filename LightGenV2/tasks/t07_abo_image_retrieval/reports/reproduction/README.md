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

## 商品图库损失与关系蒸馏（第二轮继续训练）

源模型 `refine_training_20260909/best_checkpoint.pt` 已完成80epoch，Hit@1=69.5833%，
同权重去光67.0833%，mAP@10=0.67616634。SHA256由两份新配置继承并强制核验。
对照的电子增强组完成80epoch，best60：Hit@1=67.0833%，去光66.6667%；不采用它作为续训起点。

```bash
# 每组60 epoch，沿用上述 Python 环境与数据/特征缓存；从仓库根执行。
python -m LightGenV2.tasks.t07_abo_image_retrieval.run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_gallery.yaml
python -m LightGenV2.tasks.t07_abo_image_retrieval.run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/refine_gallery_relation.yaml
```

两组推理图完全相同，保持原V卷积核3、L卷积核5、线性64D读出、六次光传播。
每5epoch重新编码1440张干净训练图，建立120个训练商品中心；每个训练batch再以0.5动量更新
访问过的图像特征，中心为逐图L2→按商品平均→L2。训练查询排除自身商品的整个中心，
故每次是119个有效候选、11个同类正商品；最终测试仍是原120候选、12正商品，不改协议。
损失为所有正商品概率之和的负对数，加0.1监督对比；固定教师类中心CE移除，
逐向量KD从0.1退火到0。关系组另加0.5 KL：同一训练查询在完整2048D教师空间中对训练商品
中心的相似度分布，监督学生64D空间的商品分布；教师/学生温度均0.1，自身商品两边都排除。
教师只用已有square缓存的train部分，不加载教师Transformer做学生推理。
所有memory均stop-gradient，查询分支可导；memory不属于部署模型，恢复训练时重建。

训练时显式恢复phase dropout，全图库刷新和test时关闭。更正此前说明：继承的
`FourLayerOpticalReplacement.set_student_train_mode()`本来就会恢复phase dropout，
因此旧训练并没有“测试后不恢复”的缺陷；新profile的显式调用只是重复保障，不改变该行为。
保留20%–30%未调制训练分量、CCD噪声、Router均衡。
仅保存best/last和最终报告、图；test每5epoch选best，仍属于test-selected结果。
训练memory定义另写入每个run的 `training_gallery_contract.json`。

## 2026-09-10：原目标低学习率续训 / 相位再加热

图库两组60epoch均已结束，Hit@1分别68.75%和68.5417%，未替换69.5833%的旧best。
本轮从 `refine_training_20260909/best_checkpoint.pt` 继续，恢复其原损失：0.5监督对比、
1.0训练教师类别中心CE、0.1逐向量KD（固定，不再退火）；无全图库loss或关系KL。
仅30epoch，batch=20，48step/epoch，EMA0.99，5epoch测试一次；best/last两份权重。
相同seed与增强，同一路径/几何/Top-2，无新网络。对照使用上一轮终点附近的学习率；
另一组仅将特征相位LR从0.0004提高到0.004，Router与电子LR不变。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/polish_low_lr.yaml
python -m LightGenV2.tasks.t07_abo_image_retrieval.run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/polish_phase_reheat.yaml
```

目标是原480-query Hit@1超过0.70，不改query、图库、标签、指标或评估预处理。
测试仍参与选模，小幅超过阈值不等于统计显著改进；完成后固定best再复评，报告命中张数和去光结果。

新增第三个有界对照 `polish_smoothing.yaml`，与low_lr仅相差anchor CE的label_smoothing=0.1，
推理不变。这基于旧best缓存的训练集诊断：1440训练图查询119个其他训练商品中心，
Hit@1=99.375%（1431/1440），不属于独立测试；未见商品仍为334/480。
不得把训练图的该诊断指标填入论文测试性能。

```bash
python -m LightGenV2.tasks.t07_abo_image_retrieval.run --mode optical \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/polish_smoothing.yaml
```

## 2026-09-10 完成结果与固定权重复评

本轮保留原推理结构、480张query、120个训练商品中心、Top-2、同尺度融合、训练直流/CCD噪声。
单seed，三个预设30epoch对照；每5epoch测试选EMA best，**不是独立无偏测试，也不是多seed统计**。

| 方案 | best epoch | Hit@1 | mAP@10 | 同权重去光Hit@1 |
| --- | ---: | ---: | ---: | ---: |
| 旧refine_training | 80 | 69.5833% | 0.67616634 | 67.0833% |
| polish_low_lr | 10 | 69.5833% | 0.68140021 | 67.0833% |
| polish_smoothing | 30 | 69.3750% | 0.67446313 | 67.5000% |
| phase_reheat最终恢复best，A100 | 10 | **70.0000%** | **0.67950967** | **67.0833%** |
| phase_reheat固定best，RTX4090复评两次一致 | 10 | **70.2083%** | **0.68003894** | **67.2917%** |

phase_reheat的A100训练周期best为70.0000%（336/480）；RTX4090固定权重复评为337/480。
两次4090逐query CSV完全一致；跨GPU仅query `4959f7d1e5955571` 的Top-1正确性由错变对。
两设备测试embedding平均余弦0.99979985、元素RMS差0.00250140；评估沿用bfloat16 autocast，
与微小数值差影响近邻排序相符，但没有进一步隔离到具体算子。保守主表记70.00%，不挑硬件较高结果充当额外训练提升。
相较旧best仅增加2～3张命中，不能因此宣称统计显著或已经接近95.2083%的冻结baseline。

训练源码：low_lr和phase_reheat为`40544b2b`；smoothing及固定复评为`74da80d0`。
后者只新增可选训练CE label_smoothing，phase_reheat配置未启用，推理图不变。
新best SHA256：`3674981c3499555077c7eadc3072a11e07583675000ec4c012616a96185867f5`。
续训起点为旧best SHA256 `2d588f40a1aa9f09336745b1e14a61c0cca76874f4c6d4f24a3d6163b978f20c`。
V1/V2/L1/L2 alpha分别约0.10270/0.10415/0.08809/0.08742；这是融合系数，不等于性能贡献。
同权重去光的A100和4090差值均为2.9167个百分点；不另训纯电网络。
三组训练及两次独立复评均已complete。最佳相位raw参数相对此轮起点的RMS变化约0.0502～0.1037，
确实发生更新；这不是包裹相位的弧度距离，也不把训练末轮live权重变化当作best变化。
best epoch的live训练Top-2选择计数：V=[478,473,488,481]，L=[495,481,489,455]；
份额均约23.7%～25.8%，本轮训练未见明显单专家集中。这不是EMA best在test上的使用率。

服务器唯一结果根目录：
`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/`。

- `polish_phase_reheat_20260910/`：best/last权重，训练history、配置、最终报告和`best_phase_overview.png`、`comparison.png/pdf`。
- `polish_phase_reheat_reeval_epoch10_20260910/`：首次4090固定复评，`final_report.json`、逐query CSV、检索特征。
- `polish_phase_reheat_reeval_repeat_20260910/`：第二次独立4090进程复评，相同权重SHA和指标。
- `polish_low_lr_20260910/`与`polish_smoothing_20260910/`：失败对照，保留审计证据，不替换best。

从仓库根目录运行（output必须使用不存在的新run目录；不需重训/教师特征缓存）：

```bash
CUDA_VISIBLE_DEVICES=GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/guest3/miniconda3/envs/xml/bin/python -m LightGenV2.tasks.t07_abo_image_retrieval.run \
  --mode evaluate \
  --config LightGenV2/tasks/t07_abo_image_retrieval/configs/polish_phase_reheat.yaml \
  --checkpoint LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/polish_phase_reheat_20260910/best_checkpoint.pt \
  --run-dir LightGenV2/tasks/t07_abo_image_retrieval/runs/simulation/phase_reheat_fixed_check_new
```

GPU UUID是本服务器4090；在其他机器需替换为当地空闲GPU。数据与冻结Qwen前端仍需按前文准备。

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
