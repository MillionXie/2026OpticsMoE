# D2NN Inverse Design with CLIP Features

## Quick Start

1. Install dependencies: `pip install torch torchvision clip matplotlib pyyaml numpy`
2. Extract CLIP features (if not provided): `python extract_clip_features.py`
3. Train Adapt version: `python train_d2nn_adapt_grid.py --epochs 100`
4. Train Traditional baseline: `python train_d2nn_mnist256.py --epochs 100`

## Key Files

- `train_d2nn_adapt_grid.py` — CLIP + Transformer → phase mask → optical propagation
- `train_d2nn_mnist256.py` — Traditional D2NN (direct phase optimization)
- `mask_generator.py` — Transformer-based phase mask generator
- `model_adapt.py` — Optical backbone (fixed, no trainable params)
- `model.py` — D2NN classifier with trainable phase params
- `optics.py` — Angular spectrum propagation + detector
- `config.yaml` — All hyperparameters

## Configuration (config.yaml)

- 1 layer, 256×256 phase mask, 400×400 canvas
- Wavelength: 532nm, Pixel: 8μm, Distance: 5cm
- Detector: 10×32×32 grid regions
- Optimizer: AdamW, lr=0.001
- CLIP: RN50 (1024-dim features)

## Data

CLIP features are pre-extracted for MNIST and stored in `data_mnist_clip_features/`.
Copy them to `./data/mnist_clip_features/` before running.
