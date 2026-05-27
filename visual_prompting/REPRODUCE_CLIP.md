# Reproduce CLIP Visual Prompting

This note describes the minimal CLIP visual prompting path for CIFAR10 first, with EuroSAT also supported.

## Environment Setup

Create or activate your Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The CLIP path also needs OpenAI CLIP. If the CLIP repository is placed next to this repository:

```text
2026OpticsMoE/
  CLIP/
  visual_prompting/
```

install it into the same environment with:

```bash
pip install -e ../CLIP
```

If you do not have a local CLIP checkout, install it from GitHub:

```bash
pip install git+https://github.com/openai/CLIP.git
```

If you use conda on Windows, for example:

```powershell
conda activate RFL
pip install -r requirements.txt
pip install -e ../CLIP
```

## Dataset Path Convention

Pass `--root` as the parent directory where torchvision should find or download the dataset.

For CIFAR10:

```text
<root>/
  cifar-10-batches-py/
```

For CIFAR100:

```text
<root>/
  cifar-100-python/
```

For EuroSAT, torchvision manages the downloaded folder under the provided root.

The script currently supports:

```text
cifar10
cifar100
eurosat
```

## Smoke Test

Run this first. It uses a tiny subset, forces 1 epoch, checks trainable parameters, saves a checkpoint, and prints train/eval accuracy.

```bash
python main_clip.py --dataset cifar10 --root ./data --smoke_test
```

Expected checks in stdout:

```text
CLIP trainable parameters: 0
Visual prompt trainable parameters: <non-zero>
Smoke test checkpoint saved: ...
Smoke test train/eval Acc@1: train=..., eval=...
```

## Train A Visual Prompt

CIFAR10:

```bash
python main_clip.py --dataset cifar10 --root ./data
```

If the default `--batch_size 256` causes CUDA OOM, start with:

```bash
python main_clip.py --dataset cifar10 --root ./data --batch_size 64 --num_workers 0
```

or:

```bash
python main_clip.py --dataset cifar10 --root ./data --batch_size 32 --num_workers 0
```

EuroSAT:

```bash
python main_clip.py --dataset eurosat --root ./data
```

## Evaluate A Saved Prompt

Use the `model_best.pth.tar` or `checkpoint.pth.tar` saved under `--model_dir`.

```bash
python main_clip.py \
  --dataset cifar10 \
  --root ./data \
  --evaluate \
  --resume ./save/models/<run_name>/model_best.pth.tar
```

On Windows PowerShell:

```powershell
python main_clip.py `
  --dataset cifar10 `
  --root ./data `
  --evaluate `
  --resume ./save/models/<run_name>/model_best.pth.tar
```

## Expected Output Files

By default, checkpoints are saved under:

```text
./save/models/<run_name>/
  checkpoint.pth.tar
  model_best.pth.tar
```

The checkpoint stores only the visual prompt state dict plus optimizer metadata. The CLIP model is frozen and is not saved.

## Code Changes Made For Reproducibility

- Added `--smoke_test` to `main_clip.py`.
- Added CIFAR10 and EuroSAT dataset loading while keeping CIFAR100 support.
- Froze all CLIP parameters with `requires_grad = False`.
- Added a trainable-parameter check:
  - CLIP must have zero trainable parameters.
  - the visual prompt must have non-zero trainable parameters.
- Made prompt tensors follow the input device instead of hardcoding `.cuda()`.
- Guarded `wandb.run.finish()` so runs without `--use_wandb` do not crash.
