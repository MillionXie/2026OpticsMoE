# 四任务跨模态终身学习（重新建立）

这个目录只包含老师要求的四项任务，不含 Kather2016、其他病理数据或 SONYC：

1. **EuroSAT RGB / SAR**：同一套 10 类地物标签，RGB 与 Sentinel-1 SAR 两种传感器域。
2. **CLEVR 图 / 文**：图中是否存在文本指定的颜色与形状，二分类匹配。
3. **Speech Commands 音 / 文**：原始 log-mel 语音在全部 8 个文字关键词候选中选出正确项。
4. **Physical Concepts 视频 / 文**：五种物理概念与 possible/impossible 组成 10 个文字候选，
   模型根据有序视频选择正确描述。

## 模型合同

两种模型都使用真实角谱传播、相位调制、逐层 OEO、全 CCD 强度和相同的单层电子读出。
D2NN 使用全孔径相位；ours 从第一项任务起固定 16 个 224×224 专家槽位和完整传播画布，
任务按 4→8→12→16 激活。旧专家及旧任务线性头冻结；router、共享 global phase 和当前
任务线性头用当前数据与 replay 更新。光学路由窗口的能量归一化成 soft-routing 功率，
各专家入口振幅乘 `sqrt(q)`。

全 CCD 经固定 28×28 pooling 后只允许进入一个 `Linear(784, C)`。没有 LayerNorm、隐藏层、
GELU 或第二个 Linear。D2NN 按最终线性头的交叉熵直接训练，不用 CCD 辅助损失和训练后校准。
MoE 可以使用路由预热、负载平衡及固定光学层后的同一个单层线性头重拟合；这些属于 ours 的
训练策略，不增加电子层数，且仅用验证集选择。固定 CCD 能量区如用于 MoE，只提供训练梯度，
不参与最终预测。

## 当前数据审计

|任务|训练/验证/测试样本|类别|许可与状态|
|---|---:|---:|---|
|EuroSAT RGB/SAR|6000 / 2000 / 2000|10|现有配对、空间隔离初步包；数据发布记录为 MIT|
|CLEVR 图文|6000 / 750 / 750 个问答|2|CC BY 4.0；来自 1000/125/125 张互斥图像|
|Speech Commands 音文|6263 / 843 / 867 条语音|8|CC BY 4.0；说话人互斥；每条语音面对全部 8 个文字候选|
|Physical Concepts 视频文|准备中：完整 25,000 个 quadruplet|10|CC BY 4.0；五种官方 probe；quadruplet 隔离；无标签监督前端|

Speech 使用完整去重发布包；Physical 使用每种概念全部 5,000 个 quadruplet，不抽样。旧的
Speech 二分类与 continuity 二分类分别达到约 94% 和 99%，因任务饱和且前端直接用目标标签监督，
已从正式协议中移除。CLEVR 全量 ZIP 仍在断点下载。正式矩阵须把 `require_full` 改为 true，
并使用全量重新生成的 manifest。
EuroSAT 发布许可不是 CC BY 4.0，这是对早期“仅 CC BY 4.0”偏好的明确例外，论文前需要老师确认。

## 实验顺序

先用标准 4 专家 518×518 几何分别运行四个任务的单任务 D2NN 与 MoE，验证数据和原始光路
可学性；它不冒充 16 槽终身模型的第一阶段成绩。目标是 D2NN
至少 65%，MoE 至少 70%；未通过的任务先调训练，不进入终身矩阵。通过后再固定顺序运行：

- independent D2NN transfer：每个任务单独训练一个光学骨干，冻结后在其余任务的全部训练样本上
  只拟合目标 MLP，得到 4×4 跨任务矩阵；
- sequential D2NN，无 replay；
- sequential D2NN，相同 replay；
- ours，固定 16 槽、旧专家冻结、相同 replay。

每个阶段按验证 balanced accuracy 选择 checkpoint，随后一次性评估已经学过的任务，保存
下三角矩阵、backward transfer 和 forgetting。没有联合训练 D2NN。

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
