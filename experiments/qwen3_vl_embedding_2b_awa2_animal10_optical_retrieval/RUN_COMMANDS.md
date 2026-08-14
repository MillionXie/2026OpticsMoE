# AwA2 检索运行命令

以下命令均在仓库根目录 `2026OpticsMoE` 执行。

## 0. 只检查命令和路径，不下载/训练

    python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.run_two_stage --dry-run

## 1. 一键完成 50 类预训练和 10 类微调（推荐）

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval.run_two_stage

一键脚本执行：官方 AwA2 自动下载/解压 -> 固定清单 -> 50 类教师缓存 -> 30 epoch
预训练 -> 从 50 类缓存无前向切出 10 类教师缓存 -> 从 epoch-30 EMA 继续 20 epoch -> 固定 epoch-50 EMA 评测 ->
best-observed 诊断评测 -> 可视化。

## 2. 分步运行

### 数据准备（首次约 13 GB）

    python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/configs/awa2_50class_pretrain.yaml --phase prepare_data

下载中断后重新执行同一条命令，会从 `data/AwA2-data.zip` 已有字节继续。若已手动下载，
也可以把官方目录放成以下任意一种结构：

    data/AwA2/JPEGImages/<class>/*.jpg
    data/AwA2/Animals_with_Attributes2/JPEGImages/<class>/*.jpg

### 50 类教师缓存与训练

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/configs/awa2_50class_pretrain.yaml --phase cache_teacher_embeddings

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/configs/awa2_50class_pretrain.yaml --phase train

### 10 类教师缓存与微调

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/configs/awa2_10class_finetune.yaml --phase cache_teacher_embeddings

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/configs/awa2_10class_finetune.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/runs/awa2_50class_pretrain/ema_last_checkpoint.pt

### 固定 epoch-50 EMA 评测

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/configs/awa2_10class_finetune.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/runs/awa2_10class_finetune_epoch50/ema_last_checkpoint.pt

### 可视化

    python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/configs/awa2_10class_finetune.yaml --phase visualize

## 3. Smoke（需要 AwA2 已下载）

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/configs/smoke_10class.yaml --phase all

## 4. 单元测试

    python -m pytest experiments/qwen3_vl_embedding_2b_awa2_animal10_optical_retrieval/tests -q
