# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 直接运行，不需要进入子目录。

Smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_baseline_mnist256/train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_smoke --epochs 1 --smoke_test --device cuda
```

CPU smoke：

```bash
python opticalmoe/d2nn_baseline_mnist256/train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_smoke_cpu --epochs 1 --smoke_test --device cpu
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_baseline_mnist256/train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_canvas400_5layer_seed7 --device cuda
```

200 epochs：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_baseline_mnist256/train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_canvas400_5layer_200epoch_seed7 --epochs 200 --device cuda
```

关闭可视化：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_baseline_mnist256/train_d2nn_mnist256.py --config config.yaml --run_name d2nn_mnist256_no_viz --device cuda --disable_visualization
```

评估 best checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_baseline_mnist256/evaluate.py --run_dir opticalmoe/d2nn_baseline_mnist256/runs/d2nn_mnist256_canvas400_5layer_seed7 --checkpoint best.pt --device cuda
```

