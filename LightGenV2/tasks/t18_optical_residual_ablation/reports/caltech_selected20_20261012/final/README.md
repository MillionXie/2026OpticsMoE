# Screened Caltech20 regularized experiment: final results

All eight fresh arms (depth2/4/12/16, rho0/.3, seed17) completed100 epochs.
All four CPU pair audits passed before any selection lock or test. Each first
minimum validation EMA balanced NLL checkpoint then received exactly one test.
The final CPU audit rechecked saved predictions, labels, IDs, argmax, confusion
counts, supports, macro recall/F1 and NLL; it performed no optical inference.
All training/test children and both supervisors exited; actual GPU0/1/2/4 release
was verified. No extra candidate, category change, epoch extension or test retry.

## Selected EMA results

Train and validation numbers refer to the same selected EMA, not last weights.
Macro recall averages per-class recall equally; accuracy weights each photo.
| Depth | rho | Selected epoch | Train accuracy | Val accuracy | Val macro recall | Test accuracy | Test macro recall | Train-val gap | Test balanced NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.0 | 100 | 53.57% | 45.58% | 44.49% | 56.98% | 45.48% | 7.99pp | 2.0225 |
| 2 | 0.3 | 100 | 57.80% | 51.59% | 49.92% | 62.55% | 51.68% | 6.21pp | 1.8638 |
| 4 | 0.0 | 99 | 61.81% | 49.82% | 47.65% | 64.10% | 50.16% | 11.98pp | 1.8652 |
| 4 | 0.3 | 100 | 76.90% | 61.84% | 59.26% | 72.04% | 56.31% | 15.06pp | 1.5654 |
| 12 | 0.0 | 100 | 69.02% | 54.77% | 51.95% | 64.87% | 50.94% | 14.25pp | 1.7606 |
| 12 | 0.3 | 91 | 80.90% | 60.78% | 57.94% | 73.26% | 57.84% | 20.13pp | 1.6110 |
| 16 | 0.0 | 99 | 66.18% | 53.00% | 50.28% | 63.13% | 49.76% | 13.18pp | 1.7791 |
| 16 | 0.3 | 98 | 82.94% | 59.72% | 56.73% | 72.04% | 57.53% | 23.23pp | 1.5684 |

The highest validation accuracy is L4rho.3 at61.84%; every70% validation gate fails.
L12rho.3 has the highest observed once-test accuracy73.26% and macro recall57.84%.
L4rho.3 and L16rho.3 have72.04% accuracy, with56.31% and57.53% macro recall.
Test balanced NLL instead favors L4rho.3 (1.5654); L16rho.3 (1.5684) is also
lower than L12rho.3 (1.6110). There is no one depth winning every metric.
This does not establish a70% balanced-category result or independent generalization.
The test has2064 photos with unequal support: Faces315, Motorbikes678 and
airplanes680 together occupy81.06%. Small classes have8-20 test photos.
Raw test accuracy above validation accuracy can reflect that class mixture;
the macro comparison is more informative about performance across all20 classes.

## Paired residual differences (rho.3 minus rho0)

| Depth | Val accuracy gain | Val macro recall gain | Test accuracy gain | Test macro recall gain |
|---|---:|---:|---:|---:|
| 2 | 6.01pp | 5.44pp | 5.57pp | 6.20pp |
| 4 | 12.01pp | 11.61pp | 7.95pp | 6.15pp |
| 12 | 6.01pp | 5.99pp | 8.38pp | 6.90pp |
| 16 | 6.71pp | 6.45pp | 8.91pp | 7.77pp |

All four pairs improve both validation/test accuracy and macro recall; balanced
NLL also decreases in every pair. Initialization, sample order, augmentation and
MixUp schedules match within each pair, and configuration differs only in rho.
These are observed single-seed paired differences, not cross-seed uncertainty
estimates or proof that a particular residual mechanism causes generalization.

## Convergence and generalization

| Depth | rho | Last10 vs preceding10 val balanced NLL improvement | E90 to E100 train balanced NLL improvement | Fit seconds | Mean epoch seconds |
|---|---:|---:|---:|---:|---:|
| 2 | 0.0 | 2.005% | 2.684% | 1566.28 | 15.57 |
| 2 | 0.3 | 3.040% | 5.071% | 1618.39 | 16.10 |
| 4 | 0.0 | 3.377% | 6.197% | 1864.87 | 18.27 |
| 4 | 0.3 | 5.450% | 11.719% | 1877.00 | 18.45 |
| 12 | 0.0 | 4.782% | 10.749% | 17035.04 | 169.58 |
| 12 | 0.3 | 2.292% | 15.486% | 17144.27 | 170.64 |
| 16 | 0.0 | 4.902% | 11.184% | 20816.57 | 207.20 |
| 16 | 0.3 | 4.333% | 24.758% | 21689.96 | 215.91 |

All eight validation balanced NLL window averages still improve by2.005-5.450%.
Seven selected epochs are98-100; L12rho.3 selects91. These budgets do not establish
sufficient convergence. No extension was performed after observing the results.
L2 train accuracy itself is only53.57/57.80%; a small gap does not establish good
fitting, severe overfitting or a proven capacity limit. L4 still improves in both
train and validation, with residual selected gap15.06pp.
L12/L16 residual training improves much faster late than validation, with selected
accuracy gaps20.13/23.23pp (macro gaps22.17/24.81pp). Their curves show substantial
generalization gaps and late overfitting signs, alongside continuing validation
NLL improvement. This is not evidence that all poor performance is overfitting.
L16rho.3 trains to82.94% at its selected epoch yet validates at59.72%, below
L4rho.3 and L12rho.3. More layers therefore do not ensure higher validation
performance under this seed, data and100-epoch optimization budget. Optimization
difficulty and generalization can both limit depth; no mechanism experiment or
multi-seed causal conclusion was performed. L12 vs L16 test macro difference is
only0.31pp and should not be interpreted as a stable ranking.

Class screening and regularization both changed relative to the earlier subset;
the old/new gap difference is not a controlled estimate of regularization benefit.

## Scope, license and reproducibility

Once-only validation-guided fixed-thumbnail screening froze20 of47 eligible
categories before optical training. All3719 selected photos were retained:
train1372/val283/test2064. Prior50+20 train/val roles were preserved and do not
enter the new test. The screening proxy is not optical-model accuracy. Validation
screening and previously observed old Caltech20 test make this exploratory
development; subject/photographer independence is not established. No claims of
full101-class performance, independent task selection or guaranteed precision.

Common settings: label smoothing.08, phase smoothness.05, affine20deg/5pixels/
10% scale, MixUp alpha.2, seed17, lr.002, EMA.95, effective macro16. L2/L4micro4;
L12/L16micro1. The optical ASM/OEO nine-expert graph and fixed20 windows are
retained, with no CNN/Linear/pretraining; rho affects expert/global paths only.
Every recorded MixUp hash, including the12-sample macro tail, was independently
regenerated. All phase gradients, source/data identity and pair checks passed.
All70 actual source files match original training Git blobs at
`a477d916ce76d5ac6ab109b3c892b0b4361410ce`; later report commits do not change them.

Official CaltechDATA retained author record identifies CC BY4.0 for
[Caltech101](https://data.caltech.edu/records/mzrjq-6wc02), DOI10.22002/D1.20086,
by Fei-Fei Li, Marco Andreetto, Marc'Aurelio Ranzato and Pietro Perona.
We changed category selection, split, image preparation and augmentation.
The new online API request returned403; the preserved official author record is
the license evidence, not a fabricated successful live API response.

ManifestSHA `91f7aa36a97713e300699367904c8866551996de199b9630c0545bb24dc5f60e`.
CacheSHA `a81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1`.
Server run: `LightGenV2/tasks/t18_optical_residual_ablation/runs/simulation/
caltech_selected20_s17_L2L4L12L16_reg100_4gpu_20261012`.
Original data, best/last checkpoints, predictions and prior tasks remain preserved.

## Inspected artifacts

- [Complete train/val curves with one test star per selected EMA](learning_curves_all.png), [SVG](learning_curves_all.svg), [PDF](learning_curves_all.pdf).
- [Selected split metrics](selected_summary.png), [SVG](selected_summary.svg), [PDF](selected_summary.pdf).
- Test confusion matrices: [L2](test_confusion_L2.png), [L4](test_confusion_L4.png), [L12](test_confusion_L12.png), [L16](test_confusion_L16.png); corresponding SVG/PDF files are included. Colors are row-normalized recall, annotations are actual counts.
- [Final summary](final_summary.csv), [800 epoch rows](curve_data.csv), [class support](per_class_support.csv), [all9600 confusion entries](confusion_matrices.csv), [CPU evidence](final_audit.json), [selection lock](test_global_selection_lock.json), [GPU release](release_receipt.json).

All six PNGs were visually inspected. All six PDFs were rendered with Poppler
and inspected, and SVG renderings/XML were checked. Matplotlib SVG source keeps
train dash patterns and alpha; the local PyMuPDF SVG renderer ignores some styles,
so those renderer differences were cross-checked against PDF/PNG and SVG source.
Axes, legends,20 class names, counts, selected single-test stars and single-seed
scope are present. No error bars imply unmeasured seed variability.
