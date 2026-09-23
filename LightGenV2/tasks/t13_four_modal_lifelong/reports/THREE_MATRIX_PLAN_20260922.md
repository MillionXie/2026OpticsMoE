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

EuroSAT 第一格为 80.11%；相同几何 D2NN 为 79.92%，优势仅 0.19 个百分点，尚未达到
希望的明显优势。CLEVR 阶段已完成 20 轮，按已学任务验证均值选择第 1 轮；仅在 CLEVR
验证分数从 76.28% 升到 76.49% 后接受单层 Linear 的校准。正式测试结果为：

|学习阶段 / 测试|EuroSAT|CLEVR|Speech|Physical|
|---|---:|---:|---:|---:|
|学完 EuroSAT|80.11%||||
|学完 CLEVR|76.63%|76.36%|||

CLEVR 阶段的 EuroSAT 回退 3.48 个百分点，少于无 replay D2NN 在同阶段的 15.97 个百分点；
CLEVR 当前任务分数也高于顺序 D2NN 的 71.21%。这只是前两阶段的结果，不能代替完整终身学习结论。
`replay_weight=2.0` 候选完成 10 轮后最佳验证均值为 76.10%，低于主链 76.49%，
已停止并释放 GPU。原始阶段结果为 `moe_replay_stage_1_eurosat_s17.json` 和
`moe_replay_stage_2_clevr_s17.json`，均记录旧专家与旧任务电子头未改变。

Speech 阶段已开始，Physical 尚未开始；完整矩阵进度为 3/10 个正式测试单元。

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
相位原始参数从 [-π,π] 均匀采样后降为 11.27%（模型再对原始参数做 sigmoid 相位映射）。
原始结果保存在 `reports/eurosat_optics_ablation_s17.json`。
这说明即使移除可训练相位，原输入特征、固定传播/OEO 光路与电子头仍可保持较高可分性；
强随机相位则会破坏它。因此不能以此矩阵宣称所有异源光学层都会使性能崩溃。四任务 D2NN
与 EuroSAT MoE 的完整验证相位消融见 `OPTICAL_PHASE_CONTRIBUTION_20260923.md`。

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

1. 继续 `replay_weight=1.0` MoE Speech 主链，再推进 Physical，完成矩阵 1。
2. 完成后核对 10/16/10 个单元和 checkpoint 来源，运行服务器测试并更新 README。

联合训练 D2NN 和 sequential D2NN replay 均不属于这三张正式矩阵。
