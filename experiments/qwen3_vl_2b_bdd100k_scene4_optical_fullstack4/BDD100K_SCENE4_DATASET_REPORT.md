# BDD100K Scene-4 Dataset Report

## Task definition

This experiment reads the BDD100K image-level `attributes.scene` field and builds four classes:

| BDD100K source label | Scene-4 class |
|---|---|
| `highway` | `highway` |
| `city street` | `city_street` |
| `residential` | `residential` |
| `parking lot` | `other` |
| `tunnel` | `other` |
| `gas stations` | `other` |
| `undefined` or empty | ignored |

BDD100K train is used as the experiment source train split. BDD100K val is used as the final test split. Ten percent of source train is stratified into validation.

## Raw label statistics

The following counts were audited from `det_v2_train_release.json` and `det_v2_val_release.json` on 2026-07-07.

### BDD100K train

| Source label | Samples |
|---|---:|
| city street | 43,516 |
| highway | 17,379 |
| residential | 8,074 |
| parking lot | 377 |
| tunnel | 129 |
| gas stations | 27 |
| undefined | 361 |
| **Total records** | **69,863** |

After mapping, the retained Scene-4 source-train counts are:

| Scene-4 class | Samples |
|---|---:|
| highway | 17,379 |
| city_street | 43,516 |
| residential | 8,074 |
| other | 533 |
| **Total retained** | **69,502** |

### BDD100K val / experiment test

| Source label | Samples |
|---|---:|
| city street | 6,112 |
| highway | 2,499 |
| residential | 1,253 |
| parking lot | 49 |
| tunnel | 27 |
| gas stations | 7 |
| undefined | 53 |
| **Total records** | **10,000** |

After mapping, the Scene-4 test counts are:

| Scene-4 class | Samples |
|---|---:|
| highway | 2,499 |
| city_street | 6,112 |
| residential | 1,253 |
| other | 83 |
| **Total retained** | **9,947** |

### Effective student train / validation split

With `validation_fraction=0.1`, the deterministic stratified split produces:

| Scene-4 class | Student train | Validation | Natural test |
|---|---:|---:|---:|
| highway | 15,641 | 1,738 | 2,499 |
| city_street | 39,164 | 4,352 | 6,112 |
| residential | 7,267 | 807 | 1,253 |
| other | 480 | 53 | 83 |
| **Total** | **62,552** | **6,950** | **9,947** |

## Imbalance and training policy

The source-train majority/minority ratio is approximately `81.64:1` (`city_street` versus `other`). Overall accuracy alone is therefore not sufficient. The experiment saves top-2 accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1, predictions, and confusion matrices. Top-5 is retained for compatibility with the parent experiment but is always 100% for a four-class problem and must not be used as evidence.

The main config uses `train_samples_per_class_per_epoch=1000` and `oversample_minority_classes=true`. Each teacher-MLP and student epoch therefore sees 1,000 samples per class. Large classes use a rotating window; the small `other` class is sampled with replacement. Validation and test remain in their natural distributions.

This oversampling improves optimization but cannot create new visual diversity. The `other` result must be interpreted with its support count and per-class recall.

## Prepared layout

```text
data/bdd100k_scene4/
  train/
    highway/
    city_street/
    residential/
    other/
  test/
    highway/
    city_street/
    residential/
    other/
  scene4_manifest.json
  scene4_dataset_report.md
```

Preparation reuses existing BDD100K raw images and prefers symbolic links, then hard links, with file copy only as a fallback.
