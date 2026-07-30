# FSS-1000 fine-tuning from COCO/DUTS Optical MoE16 pretraining

This experiment transfers the completed three-stage optical saliency model to
the official class-disjoint FSS-1000 split. It is intentionally separate from
the original one-stage, from-scratch FSS baseline.

## What is restored

The source `duts_student_best_train_loss.pt` is loaded strictly and restores:

1. all three Optical MoE16 expert planes and the global phase;
2. the CCD `LayerNorm + Linear(224,224)` residual recombiner;
3. the lightweight segmentation head.

The source optimizer is **not** restored. This is domain fine-tuning, not an
attempt to resume the DUTS optimizer trajectory.

## Fine-tuning schedule

- Epochs 1–3: freeze the Qwen stem, optical core, and CCD recombiner; adapt the
  already trained segmentation head to FSS masks.
- Epochs 4–50: jointly update the optical core, recombiner, and segmentation
  head with differential learning rates.
- The official FSS test classes remain disjoint from train classes.
- There is no validation split. The best checkpoint is selected by minimum
  training loss; test metrics are observations and never affect optimization
  or checkpoint selection.

The mask loss is BCE + Dice + 0.75 SoftIoU + 0.25 Boundary. Router balance is
also logged and weighted by 0.03.

Outputs are stored inside this experiment:

```text
runs/fss1000_saliency_coco_duts_pretrained_moe16/
```

Important outputs include the initial DUTS-to-FSS transfer metric, epoch
history, final test metrics, prediction CSV, phase masks, and per-sample
prediction/CCD/feature visualizations.

