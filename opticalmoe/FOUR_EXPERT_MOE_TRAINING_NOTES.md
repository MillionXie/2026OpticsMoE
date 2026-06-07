# Four-Expert OpticalMoE Training Notes

Date: 2026-06-07

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
- Config: `configs/four_expert_moe_multitask_mnist_fashion.yaml`
- MNIST and FashionMNIST share the same five expert layers, global FC phase,
  detector, and optional electronic readout.
- Each task owns an independent set of four prompt amplitude parameters and
  four optional scalar prompt phase biases.
- One update uses one MNIST batch and one FashionMNIST batch, combines their
  losses, backpropagates once, and updates the shared optical backbone and both
  task prompts.

Run:

```text
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion.yaml --run_name four_expert_multitask_mnist_fashion
```

For a quick pipeline check:

```text
python scripts/train_four_expert_multitask_moe.py --config configs/four_expert_moe_multitask_mnist_fashion.yaml --run_name four_expert_multitask_smoke --epochs 1 --smoke_test
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
entry before the shared experts.

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

The default readout is `optical_only`. In this mode there is no electronic
ReLU, GELU, or MLP; normalized detector energies are scaled and used as logits.
If `readout.type` is explicitly changed to `linear` or `mlp`, the architecture
report records the electronic parameter count and, for an MLP, its activation
and hidden dimension.

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

After multitask training, the model is evaluated in four combinations:

- MNIST data with the MNIST prompt
- MNIST data with the FashionMNIST prompt
- FashionMNIST data with the FashionMNIST prompt
- FashionMNIST data with the MNIST prompt

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
- Multitask training currently requires every task to use the same number of
  detector classes. MNIST and FashionMNIST both use ten detector indices, even
  though the meanings of their labels differ.
- There is no task-specific electronic detector head.
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
- Added a separate multitask model with a task prompt bank.
- Added a separate multitask progressive unfreezing schedule.
- Added a separate multitask training/evaluation script.
- Added MNIST + FashionMNIST joint-training configuration.
- Added correct-prompt and mismatched-prompt evaluation.

