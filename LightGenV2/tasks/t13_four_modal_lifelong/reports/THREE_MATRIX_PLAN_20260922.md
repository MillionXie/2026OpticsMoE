# 四模态终身学习三张正式矩阵

任务顺序：EuroSAT RGB/SAR → CLEVR 图文 → Speech Commands 音文 → Physical Concepts 视频文。
指标为完整测试集 balanced accuracy。最终电子读出始终只有一个 `Linear(784, C)`。

公平几何以最终 16 槽 MoE 为准：两模型均使用 1026×1026 传播画布和 986×986 有效全孔径；
MoE 光学参数 1,825,188，D2NN 光学参数 1,944,392（多 6.53%）。

## 矩阵 1：MoE + full replay

固定 16 个光学槽位，依次激活 4→8→12→16 个专家；旧专家冻结，每个已学旧任务固定回放
512 个样本。每阶段用所有已学任务的验证均值选择 checkpoint，再测试已学任务一次，形成
10 个下三角单元。EuroSAT 起点采用验证集选定的训练期电子 teacher 蒸馏 checkpoint；推理时
teacher 不存在，只有单层 Linear。

EuroSAT 第一格已完成：80.11%；相同几何 D2NN 为 79.92%，优势仅 0.19 个百分点，尚未达到
希望的明显优势。CLEVR 阶段正在运行；`replay_weight=1.0` 已完成 6/20 主训练轮次，当前最佳
第 1 轮验证 EuroSAT/CLEVR 为 76.70%/76.28%，均值 76.49%。并行的 `replay_weight=2.0`
候选已完成 3/20 轮，最佳第 1 轮验证均值 76.10%。这是验证指标，不能当作最终测试矩阵。
Speech 和 Physical 阶段尚未开始。完整矩阵进度仍为 1/10 个正式测试单元。

## 矩阵 2：独立 D2NN checkpoint 的纯推理 4×4

四个任务各自独立训练一份 986×986 D2NN 光学权重和原始单层 Linear。每格固定行任务的光学
权重，接列任务原本训练好的 Linear，直接在列任务的完整测试集推理；优化步数为 0，逐单元
不拟合电子头。第一行第一格复用矩阵 3 的 EuroSAT stage1 checkpoint，严格相等。

CLEVR 独立 D2NN 续训在第 10 轮达到验证 76.72%、测试 76.60%；全矩阵已用这个新的
checkpoint 复推。行是光学权重来源，列是测试任务：

|来源 / 测试|EuroSAT|CLEVR|Speech|Physical|
|---|---:|---:|---:|---:|
|EuroSAT|79.92%|49.97%|32.75%|50.67%|
|CLEVR|47.62%|76.60%|23.44%|10.04%|
|Speech|72.56%|49.97%|67.08%|43.90%|
|Physical|76.36%|50.26%|34.37%|80.30%|

16/16 单元完成；原始分数和逐单元类别混淆矩阵保存在
`reports/fair_986_inference_only_4x4_clevr12_s17/`。四份行结果均记录
`optimization_steps=0` 和 `source_optical_state_preserved=true`。四套光学相位 checkpoint 的哈希
不同，故 EuroSAT 列的 Speech/Physical 72.56%/76.36% 不是加载了相同光学权重。进一步固定
EuroSAT 原始 Linear，仅替换两层光学相位：原训练相位为 79.92%，全零相位仍为 75.87%，
均匀随机 [-π,π] 相位降为 11.27%。原始结果保存在 `reports/eurosat_optics_ablation_s17.json`。
这说明即使移除可训练相位，原输入特征、固定传播/OEO 光路与电子头仍可保持较高可分性；
强随机相位则会破坏它。因此不能以此矩阵宣称所有异源光学层都会使性能崩溃。

## 矩阵 3：单一可重构 D2NN，无 replay

始终只有一份最新光学权重，按相同顺序训练，每阶段测试所有已学任务。不回放旧样本。
这张 986×986 下三角矩阵已完成 10/10 单元：

|学习阶段 / 测试|EuroSAT|CLEVR|Speech|Physical|
|---|---:|---:|---:|---:|
|学完 EuroSAT|79.92%||||
|学完 CLEVR|63.95%|71.21%|||
|学完 Speech|52.42%|69.67%|58.84%||
|学完 Physical|52.62%|69.74%|52.47%|74.93%|

此矩阵显示连续训练后的遗忘。它与矩阵 2 的第一格均为 79.9159852841%。

## 后续操作

1. 两条 MoE CLEVR 候选按验证均值决定保留哪一条；随后推进 Speech、Physical，完成矩阵 1。
2. 完成后核对 10/16/10 个单元和 checkpoint 来源，运行服务器测试并更新 README。

联合训练 D2NN 和 sequential D2NN replay 均不属于这三张正式矩阵。
