# Architecture and comparison policy

## Fixed data contract

The experiment calls the same Grocery subset loader as the optical retrieval
experiment. The replacement 10-SKU list contains 306 training queries, 260
official test queries and one iconic gallery image per SKU. The persisted
manifest digest is included in every final metrics file and the comparison
tool refuses to compare different digests.

## Retrieval protocol

The CNN classifier layer is removed. Its pooled feature is mapped by one
signed linear layer to 64 dimensions and L2-normalized. Both online query and
gallery images are encoded by the same model. Deployment uses cosine
similarity against the normalized mean gallery prototype.

## Fairness notes

- The optical Student uses a frozen Qwen embedding Teacher; therefore an
  ImageNet-pretrained electronic baseline is more meaningful than a
  random-initialized CNN on 306 images.
- All models receive RGB 224×224 images and the same packaging-safe
  augmentation.
- No classification CE is introduced.
- The electronic models do not use Qwen embeddings or KD.
- Results must report model parameters because ResNet-18 is larger than the
  optical Student, while EfficientNet-B0 is much closer in scale.
