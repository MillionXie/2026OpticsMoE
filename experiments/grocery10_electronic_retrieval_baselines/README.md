# Grocery-10 electronic retrieval baselines

This experiment provides three electronic CNN baselines for the exact fixed
10-SKU retrieval protocol used by
`qwen3_vl_embedding_2b_grocery10_optical_retrieval`.

The selected SKUs, official train/test split, iconic gallery, light
augmentation, 64-dimensional embedding, cosine retrieval, PK sampling and
evaluation metrics are unchanged. These are retrieval models, not ten-class
classification heads.

## Baselines

1. **ResNet-18** is the conventional reference CNN. Residual learning was
   introduced to make deeper networks easier to optimize in
   [He et al., CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
2. **EfficientNet-B0** is the main parameter-efficient comparison. EfficientNet
   jointly scales depth, width and resolution and transfers well with fewer
   parameters
   ([Tan and Le, ICML 2019](https://research.google/pubs/efficientnet-rethinking-model-scaling-for-convolutional-neural-networks/)).
3. **MobileNetV3-Small** represents a compact electronic deployment baseline,
   designed using hardware-aware search
   ([Howard et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf)).

All formal configs use TorchVision ImageNet-1K weights. With only 306 natural
training queries, random initialization primarily measures small-data
optimization failure rather than a useful electronic upper bound. Smoke
configs use random weights only to avoid network downloads during CI.

## Model

```text
RGB image [B,3,224,224]
→ ImageNet-pretrained electronic CNN
→ global backbone feature [B,F]
→ LayerNorm(F)
→ Linear(F,64)
→ L2 normalization
→ retrieval embedding [B,64]
```

No ReLU, sigmoid, softmax or ten-class classifier is applied after the
64-dimensional projection.

The three feature dimensions are:

- ResNet-18: `F=512`
- EfficientNet-B0: `F=1280`
- MobileNetV3-Small: `F=576`

The backbone and retrieval projection are both fine-tuned. Differential
learning rates protect pretrained visual features:

```text
backbone LR   = 1e-5
projection LR = 3e-4
```

## Objective and selection

Each batch contains 10 SKUs × 3 natural images plus the 10 matching iconic
gallery images. Training uses:

```text
L = supervised_contrastive_loss + 0.25 * gallery_cross_entropy
```

Every natural query therefore sees same-SKU positives and nine wrong-SKU
gallery negatives. Checkpoints are selected by minimum training objective;
test metrics are logged but never used to choose weights.

## Outputs

Each run writes:

```text
resolved_config.json
environment.json
dataset.json
model.json
train_log.csv
best_train_loss_checkpoint.pt
last_checkpoint.pt
metrics/test_metrics.json
metrics/retrieval_results.csv
metrics/per_sku_metrics.csv
metrics/comparison_with_optical.json
figures/training_curves.png
figures/confusion_matrix.png
```
