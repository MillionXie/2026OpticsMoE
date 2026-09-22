# 四任务跨模态终身学习（重新建立）

这个目录只包含老师要求的四项任务，不含 Kather2016、其他病理数据或 SONYC：

1. **EuroSAT RGB / SAR**：同一套 10 类地物标签，RGB 与 Sentinel-1 SAR 两种传感器域。
2. **CLEVR 图 / 文**：图中是否存在文本指定的颜色与形状，二分类匹配。
3. **Speech Commands 音 / 文**：原始 log-mel 语音在全部 8 个文字关键词候选中选出正确项。
4. **Physical Concepts 视频 / 文**：五种物理概念与 possible/impossible 组成 10 个文字候选，
   模型根据有序视频选择正确描述。

## 模型合同

```text
224×224 跨模态光场
  ├─ D2NN: 全孔径相位层
  └─ ours: router 相位 → CCD 路由能量 q → sqrt(q) 加权的专家相位槽
                 ↓
        角谱传播 → OEO → shared global phase → 角谱传播 → OEO
                 ↓
        全平面 CCD 强度 → 固定 28×28 pooling → Linear(784, C)
```

两种模型都使用真实角谱传播、相位调制、逐层 OEO、全 CCD 强度和相同的单层电子读出。
D2NN 使用全孔径相位；ours 从第一项任务起固定 16 个 224×224 专家槽位和完整传播画布，
任务按 4→8→12→16 激活。旧专家及旧任务线性头冻结；router、共享 global phase 和当前
任务线性头用当前数据与 replay 更新。光学路由窗口的能量归一化成 soft-routing 功率，
各专家入口振幅乘 `sqrt(q)`。

全 CCD 经固定 28×28 pooling 后只允许进入一个 `Linear(784, C)`。没有 LayerNorm、隐藏层、
GELU 或第二个 Linear。D2NN 按最终线性头的交叉熵直接训练，不用 CCD 辅助损失和训练后校准。
MoE 可以使用路由预热、负载平衡及固定光学层后的同一个单层线性头重拟合；这些属于 ours 的
训练策略，不增加电子层数。重拟合只有在验证分数不下降时才接受。固定 CCD 能量区如用于 MoE，
只提供训练梯度和诊断；当前正式配置已将这项辅助损失设为 0。checkpoint 选择与最终预测始终由
这一层 `Linear(784, C)` 完成。

## 当前数据审计

|任务|训练/验证/测试样本|类别|许可与状态|
|---|---:|---:|---|
|EuroSAT RGB/SAR|31996 / 10930 / 10858|10|完整配对、空间组隔离；未抽样；数据发布记录为 MIT|
|CLEVR 图文|420000 / 45000 / 45000 个问答|2|CC BY 4.0；来自全部 70000/7500/7500 张互斥标注图像|
|Speech Commands 音文|6263 / 843 / 867 条语音|8|CC BY 4.0；说话人互斥；每条语音面对全部 8 个文字候选|
|Physical Concepts 视频文|70400 / 14652 / 14948|10|CC BY 4.0；来自完整 25,000 个 quadruplet；五种官方 probe；无标签监督前端|

Speech 使用完整去重发布包；Physical 使用每种概念全部 5,000 个 quadruplet，不抽样。旧的
Speech 二分类与 continuity 二分类分别达到约 94% 和 99%，因任务饱和且前端直接用目标标签监督，
已从正式协议中移除。CLEVR 全量包及重新生成的 SHA256 manifest 已完成；早期
6000/750/750 问答包只保留作训练策略筛选，不进入正式矩阵。正式矩阵使用 `require_full=true`。
EuroSAT 发布许可不是 CC BY 4.0，这是对早期“仅 CC BY 4.0”偏好的明确例外，论文前需要老师确认。

## 早期四专家单任务可学性检查

|任务|D2NN 测试 balanced accuracy|MoE 测试 balanced accuracy|
|---|---:|---:|
|EuroSAT|80.72%|81.81%|
|Speech Commands|74.33%|79.66%|
|Physical Concepts|79.30%|81.23%|
|CLEVR 全量|76.76%|77.16%|

EuroSAT 与 Physical 的 MoE 均已在完整原始数据协议上超过各自 D2NN baseline。训练后的一层线性头重拟合
只在验证集不下降时接受：EuroSAT 拒绝重拟合并恢复联合训练权重，Physical 接受重拟合；测试集
只在选择完成后评估。CLEVR 预备包只用于选定去 dropout、去辅助 CCD 损失的训练策略，正式结果
以全量 420000/45000/45000 查询运行代替。

CLEVR 全量 D2NN 在第 17 轮按验证集选出 checkpoint，验证/测试 balanced accuracy 为
76.94%/76.76%；MoE 在第 33 轮选出 checkpoint，验证/测试为 77.29%/77.16%。MoE 的线性头
重拟合使验证分数下降，已按固定规则拒绝并恢复联合训练权重。早期四专家单任务均达到门槛，
且四项 MoE 测试分数均高于对应 D2NN。

## 三张正式矩阵

正式协议和最新进度见 [三张矩阵报告](reports/THREE_MATRIX_PLAN_20260922.md)，参数公平性见
[几何与参数报告](reports/PARAMETER_FAIRNESS_20260922.md)。四个任务按 EuroSAT → CLEVR →
Speech → Physical 顺序运行，完整数据、单层 `Linear(784, C)` 和 986×986 有效孔径保持一致。

1. **MoE full replay**：16 个固定槽位依次激活 4→8→12→16，旧专家冻结，每个旧任务保存
   512 条 replay，逐阶段测试已学任务。EuroSAT 正式测试 80.11%；CLEVR 阶段正在运行，
   后续 Speech 和 Physical 未完成。因此当前尚无完整 MoE 下三角测试矩阵。
2. **四个独立 D2NN 的纯推理完整矩阵**：行是源任务的固定光学权重，列接目标任务已训练的
   原始单层 Linear；不进行任何逐单元训练。16/16 单元已完成，新 CLEVR checkpoint 的
   对角线测试为 76.60%。完整矩阵和逐单元混淆矩阵见
   `reports/fair_986_inference_only_4x4_clevr12_s17/`。
3. **单一可重构 D2NN，无 replay**：顺序学习四任务并逐阶段测试，10/10 单元已完成。
   EuroSAT 从初始 79.92% 降至最后 52.62%，呈现遗忘。

完整 D2NN 矩阵中，Physical 光学权重接 EuroSAT 原始头仍达到 76.36%，Speech 光学权重
接 EuroSAT 头达到 72.56%。已核验四份独立相位 checkpoint 的哈希不同、光学参数在推理前后
不变、逐单元优化步数为零。固定 EuroSAT 原始 Linear 的相位对照得到：训练相位 79.92%、
全零相位 75.87%、均匀随机相位 11.27%。EuroSAT 对零相位和部分异源相位相对鲁棒；
不能以这张矩阵宣称所有异源光学权重都会失效。

此前固定 478×478 几何且对每格 Linear 微调的矩阵、四专家单任务分数，以及顺序 D2NN
replay 运行只保留作探索记录，不进入上述三张正式矩阵。联合训练 D2NN 不属于终身学习比较。
## 入口

先用 `prepare.py` 为现有数据建立带 SHA256 的小型引用包，再运行：

```bash
python -m LightGenV2.tasks.t13_four_modal_lifelong.run \
  --config LightGenV2/tasks/t13_four_modal_lifelong/configs/preliminary_s17.json \
  --eurosat DATA/t13/eurosat --clevr DATA/t13/clevr \
  --speech DATA/t13/speech --physical DATA/t13/physical \
  --out LightGenV2/tasks/t13_four_modal_lifelong/runs/ID --phase smoke
```

单任务正式入口在 `--phase train` 下使用 `--only single_task`（D2NN）或
`--only single_task_moe`（MoE），并用 `--single-task-name` 指定任务。
