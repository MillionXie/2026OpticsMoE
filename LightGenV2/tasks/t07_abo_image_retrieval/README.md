# T07 商品检索（图搜图）

- 当前数据集：`data/abo_similarity10_data`，来源 `abo_similarity10_data_only.zip`。
- 2026-09-09：两组光学40epoch与冻结baseline均完成。冻结原始输入2048D Hit@1=**95.2083%**，同光学输入64D=**92.9167%**；光学初版**60.8333%**，同结构加强语义训练版**65.4167%**（best25），同best去光**64.5833%**。当前光学性能仍明显落后，不能称为接近baseline。[本轮证据](reports/reproduction/RUN_20260909.md)。
- 历史冻结 Qwen3-VL-Embedding-2B：Hit@1 **95.2083%**，出处见 `reports/reproduction/BASELINE_METHODS.md`。这不是 T08 图搜文的 73.71%。
- 操作与复现唯一入口：[reports/reproduction/README.md](reports/reproduction/README.md)。

## 2026-09-09 后续优化（运行中，未替换正式最佳）

源码 `838ae656`，两组都从上述 65.4167% 的 ABO best 开始，80 epoch，
保持数据、图库、测试口径、光路、Top-2 与原始冻结 baseline 不变：

- `configs/refine_training.yaml` → `runs/simulation/refine_training_20260909`：
  每批十类、同类两个不同商品；轻增强；5 epoch 预热及余弦退火；KD 随训练由0.5降至0.1。
- `configs/refine_electronics.yaml` → `runs/simulation/refine_electronics_20260909`：
  相同训练，再将原电子残差的 V 卷积核3、L卷积核5扩大至9；保留原通道MLP。
  末端 `LN(384) → [Linear(384,64) + Linear(384,512)→GELU→Dropout(0.1)→Linear(512,64)] → L2`。
  读出总计255,360参数（原25,408）；它是最终融合特征上的读出，不是独立输入旁路。

不加 Transformer/attention，不增加捕获次数；初始化时新增卷积系数和读出修正输出均为零。
训练仍记录专家使用、alpha、相位变化，最后恢复 best 做同权重去光对照；只保留 best/last。
查看实时进度用对应 `status.json`、`history.json`；只有出现 `final_report.json` 且 status=complete
才表示包括去光与可视化在内的完整训练结束。运行中数字不要替代正式结果表。

## 数据与指标

10 类，200 个商品，每个商品 12 个视角。按商品划分 train/val/test = 120/40/40 个商品，即 1440/480/480 张图。
保留原划分，val 本轮不用；每 5 epoch 测试一次，按 test Hit@1（同分按 mAP@10）选 EMA best。
因此结果属于 **test-selected**，不是独立无偏测试估计。

测试查询：480 张未见商品图片。图库：120 个训练商品的多视角平均中心。
同类商品算相关，每张查询有 12 个相关候选；不使用标签提前筛选图库。
这是 **同类别相似商品检索，不是同一商品实例识别**。
报告 Hit@1/5/10、Precision、positive Recall、mAP@10、NDCG@10。
旧称 R@1 等同 Hit@1，不等同找回全部相关商品的 Recall。
报告里的旧字段 `category_route_accuracy` 只是额外的10类别中心诊断，不是4专家光 Router，也不用于缩小图库。

## 当前光电结构

输入图片 + 固定英文检索指令；不输入正确类别名称或商品标题。
Qwen 原生冻结 patch embedding/tokenizer/embed_tokens 保留；原生 V/L Transformer 堆栈由现有 T01 光电替代器接管。
本任务不增加新的 Transformer 或 attention，不另建复杂电子模型。

```text
图像 → 冻结 Qwen 视觉输入头
     → V 光 Router（4 探测区，选 Top-2）
     → V 专家光场传播 → CCD/电子重载 → V 全局相位传播
     → 与电子残差同尺度融合 → Qwen 模态连接/图文 token 组装
固定 prompt → tokenizer + 冻结 token embedding ─────────┘
     → L 光 Router → L 专家传播 → CCD/电子重载 → L 全局传播
     → 同尺度电子残差融合 → 电子检索读出 → L2 归一化 64D
     → 对 120 个商品中心做余弦排序
```

“V 三层、L 三层”在这里各自指 **Router + 专家 + 全局**，共六次传播/CCD 采集；
不是三层专家后再额外加 Router。Router 探测后 Top-2 选择/权重归一化仍是电子操作。
沿用 532nm、10cm、478 原生振幅场、17μm 振幅 SLM、8μm 相位 SLM 的几何。
参数/精确维度以每个 run 的 `architecture.json` 和 `config.yaml` 为准。

电子部分沿用 T01：Vision 输入 hidden=1024，投影到192后为两组 depthwise Conv2d 残差 MLP；
Language 输入 hidden=2048，投影到192后为两组 causal depthwise Conv1d 残差 MLP。
**继承的 CCD 读出确实包含帧均值缩放、clip/log1p、478→224 pooling**，不是仅线性强度，也不是这次额外添加的显示增强。
后续上硬件必须使用相同网络输入转换；若比较原始仿真/实测 PCC，须另在统一的原始强度域计算，不能把显示增强图冒充原始 CCD。

融合使用光/电 RMS 尺度匹配与 `(1-alpha)E + alpha O`，初始 alpha=0.10，范围 [0.01,0.95]。
alpha 是融合系数，不直接等于性能贡献百分比。
加载已审计的 Caltech warmstart 光电参数；重置光 Router 与融合系数，而不是从随机网络训练。
相位为全精度 `2π sigmoid(raw_phase)`，不做 8-bit 直通量化。
训练保留 20%～30% **强度比例**的相干未调制分量、CCD 噪声与专家均衡损失；位移扰动=0、k 空间约束关闭。
测试采用确定性理想传播，不能把它称为实测成绩或固定 20% 零级光下的成绩。

## 训练与公平对照

训练仅使用 train：类别监督对比损失 + 0.3 冻结教师 64D 余弦蒸馏 + Router 均衡等约束。
类别标签只进训练损失，推理仍是图搜图；无 title-only 支路。
每 epoch 60 个 P-K batch（4 类×3 图=12），40 epoch，EMA=0.99。
仅保存 best/last 两份模型 checkpoint，另有小型检索特征与最终相位 PNG。
最终自动复评同一个 best 在 `remove_optical` 下的结果（不另训纯电模型），保存相位/Router 参数改变量与 `comparison.png/pdf`。

冻结 baseline 不训练任何参数，重跑原始长宽比预处理及同光学 224×224 中心裁切预处理，
各报告 2048D 与 64D。全量 2048D 是大模型主 baseline，square 64D 是控制预处理/维度差异的辅助对照。
同一 prompt、划分、图库和指标函数贯穿所有版本。本轮不测速度/功耗。

当前较好候选：`configs/optical_top2_dc20_anchor.yaml`；除训练损失外与初版完全同图。
它增加训练图像教师特征的类别中心 CE（权重1），KD权重改为1，对比权重0.5；这些中心不进入推理。
运行为 `runs/simulation/optical_top2_dc20_anchor_20260909`，权重 `best_checkpoint.pt`，图 `comparison.png`/`best_phase_overview.png`。
best 的 V1/V2/L1/L2 alpha 约为0.1001/0.1010/0.0979/0.0977；去光下降0.83个百分点，不能解释成10%因果贡献。
best epoch的训练选择份额：Vision约25.07/25.00/26.32/23.61%；Language约50.00/17.43/15.90/16.67%。
这只是该epoch的live训练统计，并非EMA best在test上的专家统计；语言仍有固定首选专家，尚未完全解决集中问题。
