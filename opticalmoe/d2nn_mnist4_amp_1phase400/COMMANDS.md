# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 直接运行。

Smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_mnist4_amp_1phase400/train_d2nn_mnist256.py --config configs/config.yaml --smoke_test --epochs 1
```

正式训练：

```bash
CUDA_VISIBLE_DEVICES=3 python opticalmoe/d2nn_mnist4_amp_1phase400/train_d2nn_mnist256.py --config configs/config.yaml
```

评估：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_mnist4_amp_1phase400/evaluate.py --run_dir opticalmoe/d2nn_mnist4_amp_1phase400/runs/d2nn_mnist4_amp_1phase400_template_sigmoid_zero_seed7 --checkpoint best.pt
```

