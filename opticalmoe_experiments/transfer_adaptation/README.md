# Transfer Adaptation

This experiment family studies new dataset transfer for an already trained dataset-switching OpticalMoE.

The source backbone is a trained dataset-switching OpticalMoE for MNIST, Fashion-MNIST, and EMNIST-letters. The target datasets are USPS and KMNIST.

Only one new target optical prompt is trained: `prompt_usps` or `prompt_kmnist`. The 9 expert phase layers, global FC phase mask, source prompts, source detector/readout heads, and target detector/readout head are frozen. The default target readout is `optical_only`, a fixed detector-energy readout with no trainable electronic layer.

Before training, place `source_best.pt` and `source_config.yaml` in:

`opticalmoe_experiments/transfer_adaptation/pretrained_backbones/dataset_switching_moe_mnist_fashion_emnist_letters/`

`source_architecture_report.json` is optional; if present, it is copied into each run directory.

The central evaluation is target prompt swap: target data and target readout are fixed while the prompt is swapped between the learned target prompt and each source prompt. Source retention re-evaluates MNIST, Fashion-MNIST, and EMNIST-letters after transfer to verify that adding the new prompt does not damage previous tasks.

This directory does not include ordinary D2NN, from-scratch training, or patch-adapter baselines.

