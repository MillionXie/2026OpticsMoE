# Commands

Run these commands from the repository root. Commands intentionally do not use
line-continuation backslashes.

## One-time ImageNet authorization

Open `https://huggingface.co/datasets/ILSVRC/imagenet-1k`, accept its access
conditions, create a read token, then authenticate the server account:

```bash
hf auth login
```

Do not put the token in a JSON config or Git.

## Download/reuse and validate data

```bash
python -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k.json --phase prepare_data
```

## Build frozen CLIP cache

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k.json --phase clip_cache
```

## Formal single-GPU training

```bash
CUDA_VISIBLE_DEVICES=5 python -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k.json --phase train
```

## Formal 8-GPU H200 training

Build the shared CLIP cache once before launching DDP.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k.json --phase train
```

## Final best-checkpoint evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k.json --phase evaluate
```

## Complete run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k.json --phase all
```

## Smoke run

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k_smoke.json --phase all
```

## Resume a paused run

The run directory is kept inside this experiment. Resume without editing the
formal config:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k.json --phase train --resume-checkpoint experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/runs/optical_mlp_mixer_moe9_imagenet1k_clip_distill/checkpoints/pause_after_epoch_0001.pt
```

For DDP, append the same `--resume-checkpoint` argument to the `torchrun`
training command. Only rank zero writes checkpoints; no weight merge is needed.

## Tests

```bash
python -m compileall experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill
```

```bash
pytest experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/tests -q
```

CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 torchrun --standalone --nproc_per_node=6 -m experiments.optical_mlp_mixer_moe9_imagenet1k_clip_distill --config experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/configs/imagenet1k.json --phase train --resume-checkpoint experiments/optical_mlp_mixer_moe9_imagenet1k_clip_distill/runs/optical_mlp_mixer_moe9_imagenet1k_clip_distill/checkpoints/last.pt