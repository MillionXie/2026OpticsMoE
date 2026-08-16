# Commands

Run from `/DATA/DATA1/guest3/2026OpticsMoE`.

## Smoke test

    python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --phase smoke

## Download CIFAR-100 and CIFAR-10

    python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase prepare_data

## One formal 400 × 400, 20-stage forward/backward batch

    CUDA_VISIBLE_DEVICES=3 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase formal_smoke

## Shared CIFAR-100 SupCon pretraining

    CUDA_VISIBLE_DEVICES=3 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase pretrain

## No fine-tuning prototype baseline

    CUDA_VISIBLE_DEVICES=2 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase no_finetune

## BP fine-tuning, all three seeds

    CUDA_VISIBLE_DEVICES=2 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase finetune --method bp

## Fixed pretrained feedback, all three seeds

    CUDA_VISIBLE_DEVICES=2 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase finetune --method fa_pretrained

## Fixed random feedback, all three seeds

    CUDA_VISIBLE_DEVICES=2 python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase finetune --method fa_random

## Aggregate task and matched-endpoint results

    python -m experiments.d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400 --config experiments/d2nn_cifar100_cifar10_fixed_feedback_contrastive_20stage400/configs/main.yaml --phase compare

For a single diagnostic seed, append `--seed 1234` to a fine-tuning command.
