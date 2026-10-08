# ABO 文搜图 10 cm：只微调最后电子读出头

源模型的六层光学实测 TEST：100 个标题检索 2,400 张图片，Hit@1=0.79。
完整采集从 2026-09-25 22:21:47 的首张 CCD 到 2026-09-26 01:46:24 的末张 CCD，约 3 小时 25 分；评价报告于 01:56:38 写出。这个时间包括逐层采集/生成/传输，不是 GPU 纯推理耗时。

## 数据与边界

- 原始 ABO easy100 固定划分：TRAIN 4,800 张，TEST 2,400 张，100 个商品/标题。
- 本次从 TRAIN 每个商品以种子 `20260926` 固定抽 8 张，共 800 张；对应的六层实拍 CCD 存在实验电脑的 `finetune_train800/`，服务器的同名运行子目录。
- 800 张中每类 6 张用于梯度更新，共 600；每类 2 张只用于 epoch 选择，共 200。原 TEST 2,400 张不用于更新或选 epoch。
- 训练输入是六层实测 CCD 重建的读出头前 384 维特征；可训练参数仅 `retrieval_readout = LayerNorm(384) + Linear(384,64) + L2 normalize`。冻结 Qwen、所有光学 mask、光 Router、融合参数、电子残差。
- 训练中保存每 epoch 摘要，选中 checkpoint 用训练集内部 200 张的 Hit@1（同分比较 MRR/MAP），不是直接挑原 TEST 最好的一轮。

## 文件

实验电脑：`E:\code\guest\2026OpticsMoE\ABO_T2I_10cm_alpha040_20260925\finetune_train800`。

服务器：`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t08_abo_image_text_retrieval/runs/physical/alpha040_10cm_20260925/finetune_train800`。

每层 `01_vision_router` 到 `06_language_global` 包含 `compact_amplitude/`、`manifest.jsonl`、`ccd_captured/`、`export.log`、`capture.log`。读出头特征缓存和微调权重只在服务器保存，最终报告/最优权重会同步到本地本 handoff 目录。

实验进度由 `continue_train800.py` 接续，它遇到硬件、饱和、文件数错误即停止，不擅自略过坏图。训练脚本 `cache_physical_readout.py`、`finetune_readout.py`。最终 `finetune_report.json` 会同时给出原权重、最佳微调权重在训练/留出/原 TEST 的同口径结果。

## 2026-09-26 完成结果

- TRAIN 实拍 5,100 张 CCD：Vision 三层各 800，Language 三层各 900（包括固定标题 100 张）；从首张 02:48:29 到末张 04:04:27，实际拍摄跨度 **76.0 分钟**，未计首层生成及末层上传。
- 六层新/旧相位 SHA 逐层相同；p99 中位数依次为 `18, 27, 24, 15, 13, 15.5`，最大饱和比例 `1.75e-5`。
- 源权重实测 TEST Hit@1=`0.79`。TRAIN 800 张拆为 600 适配、200 留出；最佳为 200 epoch 中第 **175** 个 epoch，留出集 Hit@1=`0.95`。TEST 不参与梯度或最佳 epoch 选择。
- 最佳权重的原 TEST **缓存读出头 FP32**：Hit@1=`0.99`、MRR≈`0.995`、MAP≈`0.90896`。用最佳权重对原六层 CCD **重新完整前向（服务器 BF16 路径）**：Hit@1=`1.00`、MRR=`1.00`、MAP≈`0.90908`。100 个标题中只有第 90 类在两种数值路径之间发生一次第一名翻转，FP32 候选分数差约 `0.00160`。因此建议论文/汇报以保守的 **0.99～1.00（取决于数值精度）** 陈述，不把 1.00 解释为对新样本的保证。
- 数据审计：原始 TRAIN/TEST 的样本 ID 与路径均无交集；本次 800 张 TRAIN 与 2,400 张 TEST 原文件 SHA256 无相同内容。最佳新 checkpoint 的 Vision/Language 光学状态各 51 个张量与源 checkpoint 逐项完全相同；只改了最终 `retrieval_readout`。

本地结果在本文件旁的 `finetune_train800/`：`best_readout_checkpoint.pt`（部署用）、`last_readout_checkpoint.pt`（第 200 epoch 对照）、`history.csv`（逐轮）、`finetune_report.json`（缓存口径）和 `finetuned_test_report.json` / `finetuned_test_predictions.csv`（原 CCD 完整前向）。最佳权重 SHA256 为 `d4d1f6c46ec758473f6b13f8db81140465072363f537d3cb0c65eb2074477e68`，本地下载后已与服务器报告核对一致。原相位 BMP 不变，无须因本次电子头微调重新采 CCD。
