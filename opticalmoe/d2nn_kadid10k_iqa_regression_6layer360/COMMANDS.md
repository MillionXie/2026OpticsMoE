# Commands

以下命令均从仓库根目录 `2026OpticsMoE` 直接运行，不需要进入本实验文件夹。

准备或自动定位 KADID-10k：

```bash
python opticalmoe/d2nn_kadid10k_iqa_regression_6layer360/train.py --config configs/kadid10k_iqa_regression.yaml --phase prepare_data
```

Smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_kadid10k_iqa_regression_6layer360/train.py --config configs/kadid10k_iqa_regression_smoke.yaml --phase all --smoke-test
```

完整训练与测试：

```bash
CUDA_VISIBLE_DEVICES=3 python opticalmoe/d2nn_kadid10k_iqa_regression_6layer360/train.py --config configs/kadid10k_iqa_regression.yaml --phase all
```

仅训练：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_kadid10k_iqa_regression_6layer360/train.py --config configs/kadid10k_iqa_regression.yaml --phase train
```

从最佳 checkpoint 重新测试：

```bash
CUDA_VISIBLE_DEVICES=0 python opticalmoe/d2nn_kadid10k_iqa_regression_6layer360/train.py --config configs/kadid10k_iqa_regression.yaml --phase test
```
