# 冻结前端：监督尺度配对

用户已确认Qwen baseline冻结主干，光模型前端同样冻结。50轮baseline CC=.87483830已独立复评，
正式光候选CC=.86204960，目标.87尚未达到。前轮ASAM两组均停止，无新性能提升。

本轮不再调整网络或ASAM半径。保留原GT KL1+CC1.5+SIM.25−NSS.1、空间CC教师权重2和SAM.05，
额外添加相同系数1.5的真实标签CC监督；唯一配对变量是损失计算的空间尺度：

|配置后缀|额外CC损失|
|---|---|
|cc_fullgrid_20260913|原224×224密度上的1−CC|
|cc_pyramid_20260913|56×56和14×14平均池化密度上的1−CC，两个尺度等权平均|

先对224×224 logits做空间softmax，再池化概率密度；不是池化logits，不改变网络输出。
GT仅用于训练损失，常数GT在其对应尺度跳过。浮点32计算和单位均值缩放稳定CC分母。
fullgrid使用相同辅助函数，隔离数值实现差异；近常数图的稳定分母与原legacy CC不完全相同。
因此不能把它严格称为只把旧cc_weight改为3的逐位等价实现。

显著性研究讨论过联合CC/NSS/KLD的不同偏好，例如
[EML-NET原论文](https://www.sciencedirect.com/science/article/pii/S0262885620300196)。
**这里的密度金字塔是本项目待验证的训练假设，不是该论文的方法复现**：检验粗空间监督是否有助于定位，
不能预先保证提升，也不能将粗尺度分数作为最终分数。不引入该论文的多编码器网络。

## 固定合同

两组均从正式87ad候选开始，SHA256
`87ad4db51e3f58f9a41d6df09092439e5a09e81e93f88a3bfe2d3d008fafb29a`。
30轮预算，第21轮低学习率精修；学习率、EMA、路由均衡沿用extra_control；不做图像增强。
重新创建optimizer，不声称精确恢复旧优化器轨迹。仅best/last，完整train10000/test5000。
每1/5/末轮原分辨率公开测试选模，有选择偏差；不增加测试后处理或推理集成。
光router Top2、alpha>=.4、478 ROI、224专家、17µm、10cm、训练DC20–30%全部不变。
85,412参数读出头不变，额外推理参数0，Qwen patch/位置编码及原生主干均不解冻。

## 命令与停止条件

先检查可用GPU，测试并推送代码后启动，最多两张；不能占用他人GPU进程。

```bash
python -m pytest LightGenV2/tasks/t03_saliency/tests -q
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_cc_fullgrid_20260913.yaml
python -u -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all --config LightGenV2/tasks/t03_saliency/configs/moe_alpha40_cc_pyramid_20260913.yaml
```

产物为T03 `runs/simulation/moe_alpha40_cc_{fullgrid,pyramid}_20260913_seed42`。
首次真实更新核查相位梯度/改变量、冻结前端和输出形状；第5/10/15轮看全测试趋势、alpha、专家份额。
如训练改善而测试持续退步，停止并记录实际last轮数，不强行写成完成30轮。
发现NaN、相位不更新、alpha越界、专家只固定一对等则先诊断；不得以修改评价或增大电子网络绕过。
新候选必须重载best、独立全5000复算CC，并做同权重去光对照后才可替代正式候选。

本文件创建时尚未启动，不表示已经取得新的分数。
