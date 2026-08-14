# Caltech-101 两阶段光学图像检索

该独立实验复用 Grocery/AwA2 检索中已经验证的 Qwen3-VL-Embedding teacher、MoE4 光学 student、缓存、训练、评测和可视化流程，不修改已有实验。

## 任务定义

Caltech-101 提供的是物体类别标签，不提供同一物体实例的多视角 ID。因此本实验执行类别级图像检索：query 和 gallery 图像来自同一类别即为匹配。`BACKGROUND_Google` 不是物体类别，始终排除；第一阶段恰好使用其余 101 个类别。

固定随机种子 42 后，每个类别内部用稳定 SHA256 排序：

- 3 张图像固定为 gallery；
- 20% 图像固定为 query/test；
- 其余图像用于训练；
- train、gallery、query 在两阶段开始前即固定且互不重叠。

官方数据约 9,000 张图像，每类约 40–800 张、多数约 50 张，归档约 137.4 MB。程序可从 CaltechDATA 官方记录自动下载、校验 MD5、解压嵌套归档，并自动识别 `101_ObjectCategories`。

## 两阶段训练

第一阶段在全部 101 个物体类别上预训练 30 epoch。第二阶段从 epoch-30 EMA checkpoint 继续，在以下 10 类上训练 20 epoch（最终绝对 epoch 为 50）：

`airplanes`、`Motorbikes`、`Faces`、`Leopards`、`accordion`、`grand_piano`、`scorpion`、`sunflower`、`watch`、`yin_yang`。

第二阶段不会重新运行 Qwen：它按严格的 manifest identity 从第一阶段全量 teacher cache 中切出目标 10 类。

一键执行：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_two_stage

先检查命令和路径而不执行：

    python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_two_stage --dry-run

更细的分阶段命令见 `RUN_COMMANDS.md`。

## 输出

所有运行结果只写入本实验目录的 `runs/`：

- `runs/caltech101_101class_pretrain/`
- `runs/caltech101_10class_finetune_epoch50/`

保存固定最终 checkpoint、EMA checkpoint、训练历史、teacher/student 指标、逐 query 结果、混淆矩阵、检索示例、失败案例和两阶段比较 JSON。每 epoch 的 test 仅作为观察；accuracy-only 参考使用固定 epoch-50 EMA，论文路由分析主结果使用 epoch-56 最终折中版，另行标记的 best-observed test 具有选择偏差。

## 物理结构

Vision 与 Language 均使用同一套 response-preserving Optical MoE4：2×2 专家、Top-2、每专家一层 224×224 phase-only mask、global phase、5 cm 传播和 CCD/OEO。详细数据流与损失见 `ARCHITECTURE.md`。
