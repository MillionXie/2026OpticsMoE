# OpticalMoE

Initial PyTorch framework for a diffractive optical neural network classifier.

This version implements a single optical classifier:

- angular spectrum propagation with `torch.fft`
- trainable phase-only diffractive layers
- a reserved physical prompt plane using `IdentityPrompt`
- fixed detector arrays
- configurable electronic readout
- MNIST, FashionMNIST, and KMNIST dataloaders
- CSV logging, checkpoints, and visualizations

The original single-expert path is still available through `scripts/train.py`.
The large-canvas OpticalMoE bank path is available through `scripts/run_optical_moe.py`.
See `OPTICAL_MOE_USAGE.md` for the recommended YAML-driven workflow.

## Recommended Dataset Order

Start with MNIST to verify the optical training pipeline.

Use FashionMNIST next. It is still grayscale, small, and 10-class like MNIST, but the images are clothes and shoes rather than digits, so it is a better next step without making the task too hard.

Use KMNIST after that if you want another 10-class grayscale dataset with different character shapes.

## One-Line Commands
Run OpticalMoE YAML configs:
```bash
python scripts/run_optical_moe.py --config configs/optical_moe_eval_mnist_migrated.yaml
python scripts/run_optical_moe.py --config configs/optical_moe_train_mnist_left_debug.yaml
python scripts/run_optical_moe.py --config configs/optical_moe_finetune_mnist_left_comp.yaml
python scripts/run_optical_moe.py --config configs/optical_moe_eval_mixed_assembled.yaml
python scripts/run_optical_moe.py --config configs/optical_moe_eval_task_mnist_left.yaml
python scripts/run_optical_moe.py --config configs/optical_moe_eval_task_fashion_right.yaml
```

Run grating_alignment_test:
```bash
python scripts/prompt_grating_alignment_test.py --run_name prompt_grating_alignment_v1 --output_dir runs/prompt_grating_alignment_v1
```

Run MNIST:

```bash
python scripts/train.py --config configs/mnist_donn.yaml --run_name mnist_001
```

Run FashionMNIST:

```bash
python scripts/train.py --config configs/fashionmnist_donn.yaml --run_name fashionmnist_001
```

Run the small smoke config:

```bash
python scripts/train.py --config configs/mnist_donn_smoke.yaml --run_name smoke_mnist
```

Resume training and change learning rate:

```bash
python scripts/train.py --config configs/mnist_donn.yaml --run_name mnist_001 --resume runs/mnist_001/last.pt --lr 0.001
```

Resume training with a fresh optimizer state:

```bash
python scripts/train.py --config configs/mnist_donn.yaml --run_name mnist_001 --resume runs/mnist_001/last.pt --lr 0.001 --reset_optimizer
```

Run tests:

```bash
python -m pytest -q
```

## Large-Canvas OpticalMoE Commands

Run a smoke eval in the large 800 x 1600 bank geometry:

```bash
python scripts/run_optical_moe.py --config configs/optical_moe_bank.yaml
```

Train one side from scratch in the bank geometry:

```bash
python scripts/run_optical_moe.py --config configs/optical_moe_train_mnist_left_debug.yaml
```

Fine-tune only the entrance compensation prompt:

```bash
python scripts/run_optical_moe.py --config configs/optical_moe_finetune_mnist_left_comp.yaml
```

Evaluate paired 10-class left/right detector summation:

```bash
python scripts/run_optical_moe.py --config configs/optical_moe_eval_mixed_assembled.yaml
```

Evaluate a checkpoint already trained by `run_optical_moe.py`:

```bash
python scripts/run_optical_moe.py --config configs/optical_moe_bank.yaml --mode eval --dataset mnist --target_side left --moe_ckpt runs/mnist_left_bank_train/best.pt --run_name mnist_left_bank_eval_loaded
```

Assemble left/right experts that were trained separately by `run_optical_moe.py`:

```bash
python scripts/run_optical_moe.py --config configs/optical_moe_bank.yaml --mode eval --dataset mixed_mnist_fashion --readout_mode paired_sum_global --left_moe_ckpt runs/mnist_left_bank_train/best.pt --right_moe_ckpt runs/fashion_right_bank_train/best.pt --run_name optical_moe_assembled_eval
```

Run the grating alignment diagnostic:

```bash
python scripts/prompt_grating_alignment_test.py --run_name prompt_grating_alignment_v1 --output_dir runs/prompt_grating_alignment_v1
```

## Changing Epochs

`training.epochs` means the total target epoch.

If a run has already reached epoch 20 and you want it to continue to epoch 80, edit the config:

```yaml
training:
  epochs: 80
```

Then resume:

```bash
python scripts/train.py --config configs/mnist_donn.yaml --run_name mnist_001 --resume runs/mnist_001/last.pt
```

## Changing Batch Size

The simulation uses a 600 x 600 complex field and multiple FFT propagations, so memory use is much higher than a normal MNIST classifier.

Suggested starting points:

```text
6 GB GPU: batch_size 2 or 4
8 GB GPU: batch_size 4 or 8
12 GB GPU: batch_size 8 or 16
24 GB GPU: batch_size 16 or 32
```

If CUDA out of memory occurs, lower `dataset.batch_size` in the YAML config and resume from `last.pt`.

## Run Outputs

Each run writes to:

```text
runs/<run_name>/
  config.yaml
  metrics.csv
  summary.json
  best.pt
  last.pt
  detector_layout.png
  detector_energy_bar.png
  confusion_matrix.png
  phases/
  light_fields/
  sample_outputs/
```

`last.pt` is the most recent checkpoint. `best.pt` is the checkpoint with the best validation accuracy.

## Physical Defaults

- wavelength: 532 nm
- pixel size: 8 um
- input image size: 200 x 200
- padding: 200 pixels on each side
- simulation grid: 600 x 600
- phase layers: 5
- default propagation distance: 5 cm for every segment

For 5 phase layers, the optical path has 7 propagation segments:

```text
input -> prompt
prompt -> phase layer 1
phase layer 1 -> phase layer 2
phase layer 2 -> phase layer 3
phase layer 3 -> phase layer 4
phase layer 4 -> phase layer 5
phase layer 5 -> detector
```

The prompt plane always exists. No-prompt experiments use `IdentityPrompt`.

## Config Notes

`readout.type` can be:

```text
optical_only
linear
mlp
```

`optical_only` is the most physically constrained readout. `linear` and `mlp` add electronic post-processing and are usually easier to optimize.

`dataset.name` can be:

```text
mnist
fashionmnist
kmnist
```
