# Commands

Run these commands from:

```powershell
cd opticalmoe/d2nn_baseline_mnist256
```

## Smoke Test

```powershell
python train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_smoke --epochs 1 --smoke_test --device cuda
```

Use CPU if CUDA is unavailable:

```powershell
python train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_smoke_cpu --epochs 1 --smoke_test --device cpu
```

## Formal Training

```powershell
python train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_canvas400_5layer_seed7 --device cuda
```

## Formal Training With 200 Epochs

```powershell
python train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_canvas400_5layer_200epoch_seed7 --epochs 200 --device cuda
```

## Disable Visualization

```powershell
python train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_no_viz --device cuda --disable_visualization
```

## Evaluate Best Checkpoint

```powershell
python evaluate.py --run_dir runs/d2nn_mnist256_canvas400_5layer_seed7 --checkpoint best.pt --device cuda
```

## Notes

- This is a plain D2NN baseline, not MoE.
- It uses full MNIST train/test splits unless `--smoke_test` is passed.
- Input images are resized to `256 x 256` and centered on a `400 x 400` canvas.
- `readout.dropout` is electronic readout dropout.
- `regularization.phase_dropout` is optical phase-layer dropout.

