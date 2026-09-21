# 四模态单任务当前合同与进度（2026-09-22）

## 已固定的比较规则

- 最终电子读出严格为一个 `Linear(784, C)`，输入是完整 CCD 的 28×28 pooling；没有隐藏层。
- D2NN 使用两层光学相位、最终线性头交叉熵和统一 40 轮训练，不加 CCD 辅助损失、不做训练后校准。
- MoE 使用四个光学专家、物理 soft routing、专家预热和路由正则。可在固定最佳光学 checkpoint 后
  重拟合同一个线性头；这不增加电子层数。只有重拟合后的验证分数不下降时才接受新权重。
- 辅助 CCD 窗口只可作为 MoE 的训练损失和诊断量。所有正式 checkpoint 均由最终单层
  `Linear(784, C)` 的验证分数选择，最终推理也只使用这一层的输出。
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

|任务|模型|验证 balanced accuracy|测试 balanced accuracy|状态|
|---|---|---:|---:|---|
|Speech 8 类|普通 D2NN，40 轮|73.42%|74.33%|通过 65% baseline 线|
|Speech 8 类|四专家 MoE，80 轮|79.64%|79.66%|通过 70% ours 线|
|EuroSAT 10 类|普通 D2NN，40 轮|81.54%|80.72%|完整数据，正式 baseline|
|CLEVR 二分类（当前预备包）|普通 D2NN，40 轮|74.80%|73.07%|仅用于调参，等待全量包|
|CLEVR 二分类（当前预备包）|四专家 MoE，40 轮|70.53%|69.20%|未作为最终 ours，已启动去 dropout/去辅助损失版本|
|Physical 10 类|普通 D2NN，8 轮 pilot|77.66%|77.66%|完整 25,000 quadruplet，已通过|
|Physical 10 类|四专家 MoE，8 轮 pilot|77.47%|77.57%|已通过；正在从第 8 轮续训至正式 40 轮|

Speech 的 MoE 最佳 checkpoint 是第 75 轮。固定光学层后重新初始化并拟合同一个 Linear 头会把验证分数
从 79.64% 降到 74.49%，因此按预先固定的验证规则拒绝该重拟合，并保留联合训练所得的一层 Linear。
这一步没有查看测试集；恢复后测试为 79.66%，比普通 D2NN 高 5.33 个百分点。

旧 Physical continuity 二分类即使移除监督前端并限制为单层线性读出，D2NN 仍达 99.77% validation，
证明任务本身过易；该结果只作为废弃任务的诊断。替代它的五种概念 10 类任务已使用全部
25,000 个官方 quadruplet 开始训练。8 轮 pilot 中 D2NN/MoE 的测试 balanced accuracy 分别为
77.66%/77.57%；两者正式 40 轮运行均已启动。
