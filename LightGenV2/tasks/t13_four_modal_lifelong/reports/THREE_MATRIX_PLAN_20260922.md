# 三张正式矩阵的定义、进度与单卡安排

任务顺序固定为 EuroSAT → CLEVR → Speech Commands → Physical Concepts。所有输出头都严格为一个
`Linear(784, C)`。

## 矩阵 1：ours，MoE + replay

这是 4×4 下三角终身学习矩阵。第 1/2/3/4 行分别在启用 4/8/12/16 个专家并完成当前任务后，
测试已经学过的 1/2/3/4 个任务。每个旧任务保留 512 个固定 replay 样本，覆盖所有已学旧任务；
保存 backward transfer 与 forgetting。

当前只有 EuroSAT 第一行是正式完成结果；CLEVR 阶段曾运行到主训练第 3/20 轮，但因进程终止，
将从完整的 EuroSAT stage checkpoint 严格重跑。有效进度为 1/4 行、1/10 个下三角单元。

## 矩阵 2：不可重构 D2NN + 目标 Linear 适配

四个任务各有一个独立训练好的 D2NN 光学骨干。每次固定一行的光学骨干，在每个目标任务的完整
训练集上只拟合一个目标 `Linear(784, C)`，由目标验证集选择 Linear checkpoint，再在目标测试集
评估。光学参数更新次数为 0，形成完整 4×4 矩阵。

四个源光学 checkpoint 和全部 16 个 Linear 适配单元均已完成。测试 balanced accuracy 如下，
行是固定光学骨干的来源任务，列是重新拟合单层 Linear 的目标任务：

|source optics / target Linear|EuroSAT|CLEVR|Speech|Physical|
|---|---:|---:|---:|---:|
|EuroSAT|81.27%|54.96%|50.91%|70.20%|
|CLEVR|78.73%|76.74%|47.38%|69.81%|
|Speech|79.00%|51.09%|70.63%|70.48%|
|Physical|78.68%|52.07%|48.17%|78.84%|

完整指标和逐单元类别混淆矩阵保存在
`reports/frozen_d2nn_linear_transfer_s17/matrix.json`。跨模态非对角单元在 CLEVR/Speech 上明显下降；
EuroSAT 与 Physical 的固定表示迁移较好，因此结论应表述为“固定光学骨干的跨模态兼容性不稳定”，
不能声称所有非对角单元都失效。此前生成的零适配 4×4 直接推理表是附加诊断，不计入本矩阵。

## 矩阵 3：可重构 D2NN，无 replay

始终维护唯一一份最新 D2NN 光学权重，依次训练四项任务，不保留旧样本。每个阶段结束后测试
所有已学任务，形成 4×4 下三角遗忘矩阵。

当前只有 EuroSAT 第一行正式完成。CLEVR 曾运行到第 13/20 轮，但 checkpoint 未保存 Adam 状态，
因此将从完整 EuroSAT stage checkpoint 重跑 CLEVR，避免不严格的优化器热重启。有效进度为
1/4 行、1/10 个下三角单元。

早期的 sequential D2NN replay 不是老师要求的三张矩阵，只保留日志，不再占用 GPU。

## 单 GPU 执行顺序与时间

1. 矩阵 2 已完成，进程已退出并释放 GPU。
2. 再跑矩阵 3：预计 8–10 GPU 小时；完成后释放 GPU。
3. 最后跑矩阵 1：预计 14–18 GPU 小时；完成后释放 GPU。

剩余约 22–28 GPU 小时。考虑服务器 I/O 与其他用户负载，预计墙钟时间约 1–1.5 天。任一时刻只
允许一个本项目训练或推理进程占用一张 GPU。
