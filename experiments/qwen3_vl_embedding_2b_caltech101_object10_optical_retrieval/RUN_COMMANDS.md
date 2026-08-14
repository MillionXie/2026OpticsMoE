# Caltech-101 命令

以下命令均在仓库根目录 `2026OpticsMoE` 执行。

## 一键两阶段训练（推荐）

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_two_stage

检查将要执行的命令：

    python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_two_stage --dry-run

首次运行会自动下载约 137.4 MB 的官方数据。也可手动放置为以下任一形式：

    data/Caltech101/101_ObjectCategories/<class>/*.jpg
    data/Caltech101/caltech-101/101_ObjectCategories/<class>/*.jpg

## 分阶段执行

准备并检查 101 类固定划分：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/configs/caltech101_101class_pretrain.yaml --phase prepare_data

完整 101 类预训练（含 teacher cache、训练和最终评测）：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/configs/caltech101_101class_pretrain.yaml --phase all

从全 101 类缓存切出目标 10 类缓存：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/configs/caltech101_10class_finetune.yaml --phase cache_teacher_embeddings

从 epoch-30 EMA checkpoint 续训目标 10 类：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/configs/caltech101_10class_finetune.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/runs/caltech101_101class_pretrain/ema_last_checkpoint.pt

固定 epoch-50 EMA 权重评测：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/configs/caltech101_10class_finetune.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/runs/caltech101_10class_finetune_epoch50/ema_last_checkpoint.pt

生成检索可视化：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/configs/caltech101_10class_finetune.yaml --phase visualize

## 论文分析数据

从固定 epoch-50 EMA checkpoint 导出全部 embedding、routing、detector feature 和物理相位：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.export_paper_analysis --config experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/configs/caltech101_10class_finetune.yaml --checkpoint experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/runs/caltech101_10class_finetune_epoch50/ema_last_checkpoint.pt

本地或服务器重绘 Nature 风格论文图：

    python document/17_caltech101_retrieval/plot_caltech101_paper_figures.py

## 未见目标微调类别检索

使用十类微调后的固定 epoch-50 EMA 模型，评测另一组十个未参与目标微调、但参与过 101 类预训练的类别：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_unseen_eval

该结果衡量 target-fine-tuning held-out transfer，不应表述为 unseen-to-pretraining zero-shot。输出位于：

    experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/runs/caltech101_10class_unseen_eval

## Router 均衡后期微调

从固定 epoch-50 EMA 权重保守地继续训练五个 epoch。原 93.88% 模型不会被覆盖：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_router_rebalance

只查看完整命令而不执行：

    python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_router_rebalance --dry-run

新结果保存到：

    experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/runs/caltech101_10class_router_rebalance_epoch55

其中 `routing_analysis/routing_summary.csv` 是均衡前后对比的主要数据。`lambda_router_hard_load_balance=0.05` 温和约束实际 Top-2 选择频率；旧配置中的该权重为 0，因此旧实验行为不变。

更直接的训练数据驱动 gate-bias 校准版本：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_router_bias_rebalance

它只更新现有 router gate bias，不新增推理参数，不使用类别或 test 标签；对应输出为：

    experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/runs/caltech101_10class_router_bias_rebalance_epoch53

历史对照中，原始 epoch-50 EMA 保留为 accuracy-only 上限，`0.002` gate-bias 版和强均衡版仅作为消融；论文正文统一使用下述 epoch-56 最终折中版。完整数字见：

    document/17_caltech101_retrieval/ROUTER_REBALANCE_RESULTS.md

最终论文折中版（一次中等 gate-bias 校准，然后固定 router 恢复五轮）：

    CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.run_router_rebalance_final

论文主模型为 `runs/caltech101_10class_router_rebalance_final_epoch56/last_checkpoint.pt`。
它的 Target-10 Top-1 为 92.66%，Vision 选择率为 78.6/24.1/48.5/48.8%，
Language 为 9.3/43.7/100/47.0%。原 epoch-50 EMA 仅保留作 accuracy-only 参考。

Held-out 类别保持性与 checkpoint interpolation：

    python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval.unseen_transfer_tradeoff --device 3

注意：75:25 插值是探索性点，不能写成预先确定的无偏主结果。

## Smoke test

Smoke 配置不下载数据，复用已解压的数据树，每类只取少量图像并训练一步：

    CUDA_VISIBLE_DEVICES=4 python -m experiments.qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval --config experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/configs/smoke_10class.yaml --phase all

单元测试：

    python -m pytest experiments/qwen3_vl_embedding_2b_caltech101_object10_optical_retrieval/tests -q
