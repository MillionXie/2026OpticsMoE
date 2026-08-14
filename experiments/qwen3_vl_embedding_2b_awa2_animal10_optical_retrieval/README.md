# AwA2 10 类光学动物图像检索

本实验使用冻结的 `Qwen/Qwen3-VL-Embedding-2B` 教师和可训练的 Vision + Language
Optical MoE4 学生，在 Animals with Attributes 2（AwA2）上完成**类别级图像检索**。

AwA2 标注的是 50 个动物物种，没有“同一只动物”的个体 ID。因此这里的正确任务定义是：
给定一张 query 动物图像，从 gallery 中检索相同动物类别，而不是检索同一个动物个体。

官方页面给出的数据规模为 50 类、37,322 张图像，JPEG 图像压缩包约 13 GB。代码只使用
官方 `AwA2-data.zip`，支持自动下载、部分文件断点续传和已解压目录复用。数据来源与引用见
[AwA2 官方页面](https://cvml.ista.ac.at/AwA2/)。

## 固定划分与无泄漏约束

对每个类别独立使用 `seed=42` 的 SHA256 稳定排序：

1. 前 3 张作为 gallery；
2. 接下来的约 20% 作为 query/test；
3. 剩余图像作为 train；
4. 不创建 validation；
5. query 图像不进入 50 类预训练、10 类微调或 gallery；
6. 50 类和 10 类配置对同一类别产生完全相同的图像角色。

生成的 CSV manifest、SHA256、每类样本数和划分规则都会写入对应 `runs/`。修改类别、
随机种子、gallery 数量或 test 比例后，教师缓存会因 manifest identity 不一致而拒绝复用。

## 两阶段训练

阶段一使用全部 50 类的训练图像进行通用动物表征预训练；阶段二加载第 30 轮 EMA
checkpoint，在以下固定 10 类上继续训练 20 轮：

- `antelope`
- `grizzly+bear`
- `killer+whale`
- `beaver`
- `dalmatian`
- `persian+cat`
- `horse`
- `german+shepherd`
- `zebra`
- `dolphin`

这些类别同时覆盖陆生/水生、犬科/猫科和具有明显纹理的类别，既有容易区分的类别，也保留
合理的细粒度混淆。可直接在 `configs/awa2_10class_finetune.yaml` 修改；不要在 Python 中硬编码。

一键执行：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.run_two_stage

第一次会下载约 13 GB 数据并缓存教师 64 维 embedding，耗时会明显长于后续复跑。第二阶段
不会再次运行 Qwen：它会校验模型、instruction、pixel budget 和路径后，从 50 类缓存中直接
切出目标 10 类缓存。

## 输出

正式结果位于实验自己的目录内：

    experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/runs/

包括固定 manifest、教师缓存、训练日志、last/best/EMA checkpoint、Top-1、Top-3、MRR、
逐类准确率、检索结果 CSV、混淆矩阵、Teacher/Student 示例和失败案例。固定 epoch-50 EMA
是主结果；每轮观察 test 得到的 best-observed 仅作为显式标注 selection bias 的诊断结果。

## 依赖与测试

依赖沿用仓库 Qwen 光学实验环境。数据层和训练数学的单元测试：

    python -m pytest experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/tests -q

完整命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)，准确数据流见
[ARCHITECTURE.md](ARCHITECTURE.md)。
