# Commands

Run every command from `/DATA/DATA1/guest3/2026OpticsMoE`.

## 0. Fast implementation smoke test

    python -m experiments.d2nn_cifar100c10_fixed_feedback_20stage400 --phase smoke

## 1. Download and verify CIFAR-100 and CIFAR-100-C

    python -m experiments.d2nn_cifar100c10_fixed_feedback_20stage400 --config experiments/d2nn_cifar100c10_fixed_feedback_20stage400/configs/main.yaml --phase prepare_data

Data are stored under `data/cifar100_fixed_feedback`. The downloader resumes a
partial CIFAR-100-C archive when `wget` is available.

## 2. BP group: common pretraining followed by matched BP fine-tuning

    PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar100c10_fixed_feedback_20stage400/commands/02_run_bp.sh

This is the first formal command to run. It creates the only shared pretrained
checkpoint, then runs the three configured downstream seeds.

The shell scripts resolve the selected `nvidia-smi` index to its GPU UUID before
launch. This avoids CUDA enumeration differing from the physical index. Change
`PHYSICAL_GPU_INDEX=3` to a GPU that is actually idle.

## 3. Fixed pretrained feedback group

    PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar100c10_fixed_feedback_20stage400/commands/03_run_fa_pretrained.sh

## 4. Fixed random feedback group

    PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar100c10_fixed_feedback_20stage400/commands/04_run_fa_random.sh

## 5. No-fine-tuning group

    PHYSICAL_GPU_INDEX=3 bash experiments/d2nn_cifar100c10_fixed_feedback_20stage400/commands/05_run_no_finetune.sh

## 6. Compare every completed method with matched BP endpoints

    python -m experiments.d2nn_cifar100c10_fixed_feedback_20stage400 --config experiments/d2nn_cifar100c10_fixed_feedback_20stage400/configs/main.yaml --phase compare

The comparison validates batch-order hashes, then computes parameter drift,
phasor drift, operator coherence, and endpoint update cosine relative to BP.

## Single-seed diagnostic

Append `--seed 1234` to a `run` or `finetune` command to run only that matched
seed. Do not compare different methods with different seeds.
