# Commands

Run every command from the repository root.

## Recommended: optical Vision + optical Language

The main new run disables DeepStack, injects the final visual tokens once, and
uses independent Vision/Language optical cores. It enables response-amplitude
preservation and evaluates the explicitly selection-biased best-observed-test
checkpoint while also retaining last and best-train-loss checkpoints.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_vision_language_moe4_shared.yaml --phase all

Position-specific expert ablation (196 independent four-expert banks per
optical stack):

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_vision_language_moe4_nonshared.yaml --phase all

One-step optical-language smoke run:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_vision_language_moe4_smoke.yaml --phase all

Evaluate the saved best-observed-test checkpoint selected by the config:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_vision_language_moe4_shared.yaml --phase evaluate

The checkpoint is written to
`checkpoints/best_observed_test_top1_checkpoint.pt`. This result is intentionally
labelled selection-biased because the test set is observed every epoch.

## 1. Verify configuration and dataset

    python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4.yaml --phase prepare_data

The resolved architecture must report:

    expert_group_size: [66, 66]
    active_panel_size: [950, 950]
    canvas_size: 990
    max_tokens: 196

## 2. Cache frozen teacher embeddings

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4.yaml --phase cache_teacher_embeddings

Cache path:

    experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/cache/grocery10_tokenwise_196px_teacher_embeddings.pt

## 3. Train the main global-second-plane experiment

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4.yaml --phase train

Resume:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4.yaml --phase train --resume-checkpoint experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/runs/qwen3_vl_embedding_2b_grocery10_tokenwise_moe4_global/checkpoints/last_checkpoint.pt

## 4. Evaluate

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4.yaml --phase evaluate

Evaluate a specific checkpoint:

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4.yaml --phase evaluate --checkpoint experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/runs/qwen3_vl_embedding_2b_grocery10_tokenwise_moe4_global/checkpoints/last_checkpoint.pt

## 5. One-command full run

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4.yaml --phase all

## 6. Second expert-plane ablation

This reuses the first router decision; it never computes another router output.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4_second_expert.yaml --phase all

## 7. Direct 8-µm hardware geometry

This configuration fits the 950×950 active panel directly on the current SLMs.
It is a separate training run, not a post-training resize of the 16-µm mask.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4_8um_hardware_geometry.yaml --phase all

## 8. Smoke run

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python -m experiments.qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4 --config experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/configs/grocery10_tokenwise_moe4_smoke.yaml --phase all

## 9. Unit tests

    python -m pytest experiments/qwen3_vl_embedding_2b_grocery10_tokenwise_optical_moe4/tests -q

If CUDA memory is insufficient, lower `pk_skus_per_batch` while keeping
`pk_images_per_sku >= 2`. Do not increase processor pixels unless the resulting
visual token count still fits the configured token panel.
