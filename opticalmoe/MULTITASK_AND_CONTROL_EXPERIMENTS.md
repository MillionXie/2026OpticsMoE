# Multitask Runtime And Six-Layer Control

## Why Batch Size 1 Is Still Slow

The four-expert model propagates a complex64 field on a 700 x 700 canvas.
Every task forward pass contains eight free-space propagation segments, and
each segment uses a two-dimensional FFT and inverse FFT.

Multitask training processes one MNIST batch and one FashionMNIST batch before
one backward step. With batch size 1 and a 90/10 training split, a literal full
epoch is about 54,000 paired updates. Reducing the batch size lowers GPU memory
usage but increases the number of optimizer updates, so it can make an epoch
slower.

The multitask config now uses:

```yaml
training:
  multitask:
    steps_per_epoch: 500
    print_freq: 25
  evaluation:
    max_val_batches: 100
    max_test_batches: 500
```

`steps_per_epoch` defines a research epoch as a fixed number of paired updates.
The dataloaders remain shuffled, so different samples are seen over successive
epochs. Set it to `null` only when a literal full-dataset epoch is required.

Validation and test files record sample counts. Set `max_val_batches` and
`max_test_batches` to `null` for final full-dataset evaluation.

Check the first terminal lines:

```text
device: cuda
updates per epoch: 500 (natural full-dataset value: ...)
```

If the device is `cpu`, the 700 x 700 FFT model will be extremely slow.

## Multitask Metrics

`multitask_metrics.csv` contains joint sample-weighted metrics:

- `joint_train_loss`
- `joint_train_acc`
- `joint_val_loss`
- `joint_val_acc`
- update and sample counts
- train, validation, and complete epoch durations
- per-task train and validation columns

`task_metrics.csv` is a long-format table with one row per task and epoch:

- task name
- task train loss and accuracy
- task validation loss and accuracy
- sample counts

Final correct-prompt test results are split into:

```text
task_test_metrics.csv
joint_test_metrics.csv
```

For the two-task MNIST + FashionMNIST config, the existing
`task_switching_eval.csv` contains all four correct and mismatched
dataset/prompt combinations.

For the three-task MNIST + FashionMNIST + EMNIST-letters experiment, use:

```text
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion_emnist.yaml --run_name four_expert_multitask_mnist_fashion_emnist
```

This configuration uses `emnist` with `split: letters`, so EMNIST is a
26-class English-letter task. MNIST and FashionMNIST keep 10-class heads.
The propagation, expert masks, global FC mask, and final CCD intensity field
are shared. Each task interprets that common CCD plane through its own
detector-region map and optional electronic readout. With three tasks,
`task_switching_eval.csv` contains 9 combinations: each evaluation dataset is
tested with each task prompt while keeping that dataset's own readout head.

The three-task config now exposes task loss weights:

```yaml
training:
  multitask:
    loss_weights:
      mnist: 1.0
      fashionmnist: 1.0
      emnist: 3.0
```

The terminal still prints raw per-task cross-entropy losses. The weight only
changes the combined loss used for backpropagation. EMNIST also has a
configurable electronic head; the default three-task config uses `mlp` with
`layernorm` for EMNIST and keeps MNIST/FashionMNIST as `optical_only`.

## Per-Epoch Visualization Outputs

The four-expert single-task and multitask scripts now save full process
visualizations at `visualization.save_interval_epochs`, usually every five
epochs.

Single-task output:

```text
runs/<run_name>/
  phases/epoch_0005/
    expert_phase_layers.png
    global_fc_phase.png
  prompt/epoch_0005/
    prompt_phase.png
    prompt_amplitude_map.png
  light_fields/epoch_0005/
    overview.png
    00_input_amplitude.png
    01_after_input_to_prompt.png
    02_after_prompt.png
    ...
    detector_plane.png
  sample_outputs/epoch_0005/
    expert_prompt_bars.png
    sample_predictions.png
    sample_predictions.json
```

Multitask output:

```text
runs/<run_name>/
  phases/epoch_0005/
    shared_expert_phase_layers.png
    global_fc_phase.png
  prompt/epoch_0005/mnist/
    prompt_phase.png
    prompt_amplitude_map.png
    prompt_amplitude_bar.png
  prompt/epoch_0005/fashionmnist/
    prompt_phase.png
    prompt_amplitude_map.png
    prompt_amplitude_bar.png
  light_fields/epoch_0005/mnist/
    overview.png
    00_input_amplitude.png
    01_after_input_to_prompt.png
    02_after_prompt.png
    ...
    10_detector_plane.png
    diagnostics.json
  light_fields/epoch_0005/fashionmnist/
    ...
  sample_outputs/epoch_0005/mnist/
    prompt_energy_detector_bars.png
    sample_predictions.png
    sample_predictions.json
  sample_outputs/epoch_0005/fashionmnist/
    ...
```

The shared phase masks are saved once per epoch. Task-specific prompt and light
field outputs are saved separately because MNIST and FashionMNIST use different
task prompt amplitudes and phase biases.

## Six-Layer Control Experiment

The control experiment is independent of both four-expert training scripts.

It has:

- the same 700 x 700 canvas;
- the same centered 200 x 200 input;
- the same identity physical prompt plane location, but no prompt modulation;
- no expert apertures, expert branches, or blocked gaps;
- six phase-only masks affecting the full canvas;
- the same eight propagation segments and propagation distances;
- the same detector and optical-only readout;
- AdamW with learning rate 0.003;
- the same backward progressive training concept.

Six full 700 x 700 trainable masks would contain 2,940,000 phase parameters,
which is much larger than the four-expert model. For a fairer comparison, each
control mask has a 464 x 464 trainable phase grid. The unit-magnitude complex
phase is periodically interpolated to the full 700 x 700 canvas.

Parameter comparison:

```text
four-expert optical trainable parameters: 1,290,008
six-layer control optical parameters:     1,291,776
relative difference:                      about 0.14%
```

MNIST smoke test:

```text
python scripts/train_six_layer_control.py --config configs/six_layer_control_mnist.yaml --run_name six_layer_control_mnist_smoke --epochs 1 --smoke_test
```

MNIST formal training:

```text
python scripts/train_six_layer_control.py --config configs/six_layer_control_mnist.yaml --run_name six_layer_control_mnist
```

FashionMNIST formal training:

```text
python scripts/train_six_layer_control.py --config configs/six_layer_control_fashionmnist.yaml --run_name six_layer_control_fashionmnist
```

The control output contains:

```text
config.yaml
control_architecture_report.json
control_architecture_report.md
metrics.csv
summary.json
best.pt
last.pt
initial_state/
light_fields/
  epoch_0005/
    overview.png
    phase_masks.png
    sample_predictions.png
    sample_predictions.json
confusion_matrix.png
```

For a valid comparison, use the same dataset split seed, batch size, optimizer,
learning rate, detector layout, evaluation sample limits, and epoch/stage
schedule in the MoE and control configurations.
