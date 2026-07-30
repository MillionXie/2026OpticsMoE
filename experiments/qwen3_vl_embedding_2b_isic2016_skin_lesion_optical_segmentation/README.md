# ISBI 2016 / ISIC Task 1 Optical Skin-Lesion Segmentation

This experiment performs **binary lesion-boundary segmentation**, not
melanoma classification. It uses the official ISBI 2016 Task 1 split:

- 900 training dermoscopic images and binary masks;
- 379 test dermoscopic images and binary masks;
- no validation split and no random repartition of the official test set.

The official dataset is CC0 and is downloaded automatically from the ISIC
challenge archive into the repository-wide `data/ISIC2016` directory.

## Two controlled experiments

Both configurations build the same model:

```text
RGB image 224x224
-> frozen Qwen3-VL-Embedding-2B patch/position stem
-> 3-stage Optical MoE16 (Top-4 routing)
-> global phase -> 10 cm propagation -> CCD
-> 224x224 detector readout
-> Fccd + alpha * Linear(LayerNorm(Fccd))
-> restored token grid [B,224,14,14]
-> lightweight electronic segmentation head
-> lesion mask logits [B,1,224,224]
```

`isic2016_scratch.yaml` jointly trains every task-specific optical,
recombination and segmentation-head parameter from random initialization.
“End-to-end” in this project does **not** unfreeze Qwen: all native Qwen
parameters remain frozen.

`isic2016_coco_duts_pretrained.yaml` loads the optical core, CCD residual
recombiner and segmentation head from the completed COCO feature
distillation followed by DUTS saliency pretraining. It first calibrates the
head for five epochs, then jointly fine-tunes the task-specific modules.

The test split may be printed every epoch for observation, but checkpoint
selection is always minimum training loss. The test results never select a
checkpoint, threshold or hyperparameter.

## Loss and metrics

Training uses:

```text
BCEWithLogits + Dice + 0.75 SoftIoU + 0.25 boundary Dice
+ 0.03 router balance
```

Final metrics include mean Jaccard/IoU (the challenge primary metric), Dice,
MAE, pixel accuracy, sensitivity and specificity. Masks use nearest-neighbor
resize and remain binary. Image and mask receive exactly the same crop,
flip and rotation.

Outputs are isolated under this experiment's `runs/` directory. Each run
stores resolved settings, dataset manifests, model/parameter reports,
training history, best/last checkpoints, test predictions, phase masks and
qualitative lesion-mask panels.

## Citation

Gutman et al., *Skin Lesion Analysis toward Melanoma Detection: A Challenge
at the International Symposium on Biomedical Imaging (ISBI) 2016, hosted by
the International Skin Imaging Collaboration (ISIC)*, arXiv:1605.01397.
