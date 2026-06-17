# Four-Expert OpticalMoE Training Notes

Date: 2026-06-08

## 1. Two Independent Experiments

The repository now keeps single-task and multitask training as two separate
experimental programs.

### Single-task, multiple experts

- Model: `FourExpertMoEClassifierV2`
- Script: `scripts/train_four_expert_moe_v2.py`
- Configs: `configs/four_expert_moe_mnist.yaml`,
  `configs/four_expert_moe_fashionmnist.yaml`,
  `configs/four_expert_moe_emnist.yaml`, and
  `configs/four_expert_moe_cifar10.yaml`
- One dataset is trained at a time.
- The four optical experts, global FC phase, and one trainable prompt are
  optimized for that dataset.

MNIST example:

```text
python scripts/train_four_expert_moe_v2.py --config configs/four_expert_moe_mnist.yaml --run_name four_expert_mnist
```

FashionMNIST example:

```text
python scripts/train_four_expert_moe_v2.py --config configs/four_expert_moe_fashionmnist.yaml --run_name four_expert_fashion
```

### Multitask joint training

- Model: `FourExpertMultitaskMoEClassifier`
- Script: `scripts/train_four_expert_multitask_moe.py`
- Two-task config: `configs/four_expert_moe_multitask_mnist_fashion.yaml`
- Three-task config:
  `configs/four_expert_moe_multitask_mnist_fashion_emnist.yaml`
- All tasks share the same five expert layers, global FC phase, propagation,
  and final CCD intensity field.
- Each task then applies its own fixed detector-region map and optional
  electronic readout to that common CCD intensity field. These task heads are
  post-detection interpretations of the same optical output plane, not
  separate optical backbones.
- Each task owns an independent set of four prompt amplitude parameters and
  four optional scalar prompt phase biases.
- One update uses one batch from each configured task, combines their losses,
  backpropagates once, and updates the shared optical backbone plus every
  task-specific prompt.

The multitask loss can be weighted from the YAML config:

```yaml
training:
  multitask:
    loss_reduction: mean
    loss_weights:
      mnist: 1.0
      fashionmnist: 1.0
      emnist: 3.0
```

With `loss_reduction: mean`, the optimization loss is:

```text
total_loss = sum(weight_task * loss_task) / sum(weight_task)
```

The per-task losses printed in the terminal remain the raw cross-entropy
losses. The weights only change the gradient contribution to the shared optical
backbone and task-specific prompt/readout heads.

Run:

```text
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion.yaml --run_name four_expert_multitask_mnist_fashion
```

Three-task MNIST + FashionMNIST + EMNIST-letters run:

```text
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion_emnist.yaml --run_name four_expert_multitask_mnist_fashion_emnist
```

The three-task config uses EMNIST `split: letters`. MNIST and FashionMNIST use
10-class heads, while EMNIST uses a separate 26-class detector/readout head.
EMNIST's original labels `1..26` are remapped to `0..25` for cross entropy.
The shared optical propagation, expert masks, global FC mask, and CCD plane
are unchanged.

### Paper-Style Small-Data Protocol

The dataset blocks can enable:

```yaml
sampling_protocol:
  enabled: true
  total_size: 2000
  train_test_ratio: [4, 1]
```

The current configs use `total_size: 2000` for MNIST and FashionMNIST, and
`total_size: 5200` for EMNIST letters. Sampling is class-balanced and
deterministic. The 4:1 ratio first creates a train pool and a held-out test
set. The normal `val_split` is then carved from the train pool for checkpoint
selection.

For EMNIST letters, 5200 means 200 samples per letter class. The 4:1 split
therefore gives 4160 train-pool samples and 1040 test samples before the
validation split.

For a quick pipeline check:

```text
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion_emnist.yaml --run_name four_expert_multitask_three_task_smoke --epochs 1 --smoke_test
```

The two scripts deliberately do not auto-detect each other's configuration.
This prevents multitask task-bank logic from changing the behavior of the
single-task experiment.

## 2. Shared Physical Architecture

Both experiments use the verified four-expert geometry:

- 700 x 700 simulation canvas
- centered 200 x 200 amplitude input
- four fixed 200 x 200 expert apertures
- four 300 x 300 microlens prompt cells
- prompt and expert centers are co-located
- blocked transmission outside the prompt cells
- local thin-lens phase plus local grating phase
- five trainable local expert phase layers
- one trainable full-canvas global FC phase mask
- one common detector plane

Changing the task does not replace the optical backbone. In the multitask
experiment, task switching selects a different prompt amplitude/phase-bias
before the shared experts. After the shared global FC mask and propagation,
the model obtains one common CCD intensity image. The selected task then uses
its own detector-region map and readout rule on that image.

This division also leaves a clean extension point for future tasks. A
classification task can sum task-specific detector regions, while an imaging
or generation task can consume the full CCD intensity image and apply its own
image-domain post-processing head.

## 3. Initial Optical State

Before the first optimizer step, both training programs run a fixed validation
batch through the initialized model. The results are saved under:

```text
runs/<run_name>/initial_state/
```

The single-task program writes the images directly in that directory. The
multitask program writes one subdirectory per task:

```text
runs/<run_name>/initial_state/mnist/
runs/<run_name>/initial_state/fashionmnist/
runs/<run_name>/initial_state/emnist/
```

The saved state includes the input, prompt plane, expert entrance, five expert
layers, global FC plane, detector plane, phase maps, prompt amplitudes, expert
energies, detector energies, and `initial_diagnostics.json`. This gives an
epoch-0000 reference even when normal visualizations are saved every five
epochs.

## 4. Optical and Electronic Nonlinearity

Every run saves:

```text
runs/<run_name>/architecture_report.json
runs/<run_name>/architecture_report.md
```

Free-space propagation and phase-only modulation are linear in the complex
field. The detector computes intensity `|U|^2`, which is the required optical
detection nonlinearity.

The default readout for pure optical baselines is `optical_only`. In this mode
there is no electronic ReLU, GELU, or MLP; normalized detector energies are
scaled and used as logits.

The four-expert scripts also support optional electronic post-processing:

```yaml
readout:
  type: optical_only | linear | mlp
  input_norm: none | layernorm
  norm_affine: true
  hidden_dim: 128
  hidden_layers: 2
  activation: relu | gelu | tanh | silu
  dropout: 0.0
```

For multitask runs, each task can override these values in its own `head`
block. The current three-task config keeps MNIST/FashionMNIST as
`optical_only`, but uses `mlp + layernorm` for EMNIST letters because the
26-class letter task was otherwise staying near random accuracy. If a strict
pure-optical comparison is required, set EMNIST back to:

```yaml
readout_type: optical_only
input_norm: none
```

The architecture report records input normalization, MLP depth, activation,
dropout, electronic parameter count, and whether an electronic nonlinear
readout is enabled.

## 5. Optimizer

All current four-expert configs use:

```yaml
optimizer:
  type: adamw
  lr: 0.003
  weight_decay: 0.0
```

The training scripts also support `adam` and `sgd`, but AdamW at learning rate
0.003 is the standard setting for these experiments. The resolved setting is
printed at startup and saved in the run summary and architecture report.

## 6. Task-Specific Prompt Parameters

For each multitask task, the prompt bank stores:

- `amplitude_logits`: four trainable values
- `amplitude = sigmoid(amplitude_logits)`
- `power = amplitude^2`
- normalized power across the four experts
- optional four trainable scalar phase biases

The fixed microlens and grating geometry is shared. Only these task controls
change when switching from MNIST to FashionMNIST.

Prompt histories are saved in:

```text
runs/<run_name>/task_prompt_amplitude_history.csv
runs/<run_name>/task_prompt_power_history.csv
runs/<run_name>/task_prompt_amplitude_history.png
runs/<run_name>/task_prompt_power_history.png
```

Expert energy histories are saved in
`task_expert_energy_history.csv` and
`task_expert_energy_ratio_history.png`.

## 7. Task-Switching Evaluation

After multitask training, every evaluation dataset is tested with every task
prompt. The three-task experiment therefore produces nine combinations.
During a mismatched-prompt evaluation, the evaluation dataset keeps its own
detector/readout head. Only the optical prompt is changed. This is necessary
because MNIST/FashionMNIST have 10 outputs while EMNIST letters has 26.

The results are written to:

```text
runs/<run_name>/task_switching_eval.csv
```

This comparison measures whether selecting a task-specific optical prompt
meaningfully changes the behavior of the same shared optical backbone.

## 8. Progressive Training

The single-task and multitask programs have separate progressive schedule
classes. Both can progressively unfreeze expert layers in forward or backward
order. In multitask training, all task prompt amplitudes remain trainable when
`train_task_prompts_always: true`, while the shared expert layers follow the
configured stage schedule.

The exact trainable parameter names for every stage are saved in
`trainable_parameters_by_stage.json`.

## 9. Main Output Files

Single-task metrics:

```text
runs/<run_name>/metrics.csv
runs/<run_name>/prompt_amplitude_history.csv
runs/<run_name>/expert_energy_history.csv
runs/<run_name>/detector_energy_history.csv
```

Multitask metrics:

```text
runs/<run_name>/multitask_metrics.csv
runs/<run_name>/task_val_metrics.csv
runs/<run_name>/task_prompt_amplitude_history.csv
runs/<run_name>/task_prompt_power_history.csv
runs/<run_name>/task_switching_eval.csv
```

Both experiments save `best.pt`, `last.pt`, `summary.json`, initial optical
fields, and architecture reports.

## 10. Known Limitations

- CIFAR10 is supported through grayscale amplitude encoding, but it is harder
  than MNIST-like datasets because color information is discarded.
- Multitask training supports different class counts through task-specific
  detector/readout heads. The current three-task config uses 10/10/26 outputs.
- Task switching currently uses a known task name to select trainable prompt
  amplitudes and phase biases. It is not an input-dependent learned router.
- The blocked prompt gap is intentional. Making it transparent would create an
  uncontrolled unmodulated optical path and weaken the interpretation of
  prompt amplitudes as routing controls.

## 11. Change Summary

- Added pre-training epoch-0000 optical diagnostics.
- Added explicit optical/electronic architecture reports.
- Standardized four-expert configs on AdamW with learning rate 0.003.
- Kept the existing single-task model and script single-task only.
- Added a separate multitask model with a task prompt bank and task-specific
  detector/readout heads.
- Added a separate multitask progressive unfreezing schedule.
- Added a separate multitask training/evaluation script.
- Added MNIST + FashionMNIST and MNIST + FashionMNIST + EMNIST-letters
  joint-training configurations.
- Added correct-prompt and mismatched-prompt evaluation; mismatched evaluation
  swaps only the prompt and keeps the evaluation dataset's own head.
- Added phase bypass dropout for train-time phase-layer regularization.

## 12. Phase Bypass Dropout

Phase bypass dropout is a train-time regularizer for optical phase masks. It
does not drop activations and does not rescale amplitudes. Instead, during
training only, selected phase pixels or phase blocks temporarily use identity
modulation:

```text
keep * exp(i * phase) + (1 - keep) * 1
```

The stored trainable phase parameters are never modified. During validation,
testing, and `model.eval()`, the phase layer always uses the normal
`exp(i * phase)` modulation.

YAML block:

```yaml
regularization:
  phase_dropout:
    enabled: true
    mode: block_phase_bypass
    expert_p: 0.05
    global_fc_p: 0.0
    block_size: 8
    batch_shared: true
    apply_to_experts: true
    apply_to_global_fc: false
    start_epoch: 10
```

Recommended first comparison:

```text
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion_emnist.yaml --run_name baseline_no_phase_dropout
python scripts/train_four_expert_multitask_moe.py --config configs/multitask_5000_5000_13000_phase_bypass_dropout.yaml --run_name phase_bypass_dropout_005
```

Each run writes `phase_dropout_summary.json`. The multitask run also writes
`multitask_loader_summary.json`, which records per-task sample counts, batch
sizes, loader steps, effective updates per epoch, repeat factors, and reset
counts.
