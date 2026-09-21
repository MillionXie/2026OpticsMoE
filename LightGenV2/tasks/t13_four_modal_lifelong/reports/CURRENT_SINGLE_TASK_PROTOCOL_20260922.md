# 四模态单任务当前合同与进度（2026-09-22）

## 已固定的比较规则

- 最终电子读出严格为一个 `Linear(784, C)`，输入是完整 CCD 的 28×28 pooling；没有隐藏层。
- D2NN 使用两层光学相位、最终线性头交叉熵和统一 40 轮训练，不加 CCD 辅助损失、不做训练后校准。
- MoE 使用四个光学专家、物理 soft routing、专家预热和路由正则。可在固定最佳光学 checkpoint 后
  重拟合同一个线性头；这不增加电子层数，轮次只由验证集选择。
- 正式数据必须声明 `all_original_samples=true`。测试集不参与模型或超参数选择。

## 调整后的非饱和任务

|任务|完整划分|类别|输入与随机水平|
|---|---:|---:|---|
|Speech Commands|6263 / 843 / 867 条语音|8|原始 log-mel + 8 个固定文字候选；随机 12.5%|
|Physical Concepts|五种概念共 25,000 个 quadruplet|10|相邻帧差分 + 概念/可行性文字候选；随机 10%|

Speech 不再为每条音频只构造一个容易的负例。Physical 不再只做 continuity 二分类，也不再使用
用 possible/impossible 标签训练到接近满分的 CNN 前端。两项协议均记录
`label_supervised_frontend=false`。

## 已完成结果

Speech 的朴素 D2NN 40 轮已完成：validation balanced accuracy 73.42%，test 74.33%。任务难度满足
65%--95% 的目标区间。第一轮路由稳定的 MoE 为 validation 67.74%，test 66.88%，尚未达到 70%
准入线，因此不作为最终 ours；当前正在比较路由强度、相位 dropout、专家 specialization 和 80 轮训练。

旧 Physical continuity 二分类即使移除监督前端并限制为单层线性读出，D2NN 仍达 99.77% validation，
证明任务本身过易；该结果只作为废弃任务的诊断。五种概念的 10 候选完整包准备完成后才开始正式训练。
