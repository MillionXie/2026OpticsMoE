# Training sample count and the separate larger-data option

Human asked why there are only1246 training images, permitted the three phase-grid
diagnostics promptly, and prefers the already fixed final network size. If current
data is unsuitable, change to a suitable dataset rather than automatically shrinking
the final architecture. No new dataset training or test is started by this document.

## Actual source count, 2026-10-14

Read the existing author's archive member headers only (no image decode or dataset
arrays) and the frozen revised20 manifest. Original archive has9144JPG across102
folders including background. Selected20 folders contain1937 images in total;
1246train/261val/430test-count metadata, identical to existing frozen cache identity.
The official [Caltech description](https://data.caltech.edu/records/mzrjq-6wc02)
also states roughly40-800 images/category and most categories about50 images.
It is a small, unbalanced dataset;20 selected categories do not provide10,000 photos.

| Class | All source photos | Train | Val | Test count metadata |
|---|---:|---:|---:|---:|
| Leopards |200|100|20|80|
| bonsai |128|89|19|20|
| car_side |123|86|18|19|
| crab |73|51|10|12|
| crayfish |70|49|10|11|
| grand_piano |99|69|14|16|
| helicopter |88|61|13|14|
| joshua_tree |64|44|10|10|
| kangaroo |86|60|12|14|
| ketch |114|79|17|18|
| laptop |81|60|12|9|
| lotus |66|46|10|10|
| minaret |76|53|11|12|
| revolver |82|57|12|13|
| schooner |63|44|10|9|
| stop_sign |64|44|10|10|
| trilobite |86|60|12|14|
| umbrella |75|52|11|12|
| watch |239|100|20|119|
| yin_yang |60|42|10|8|

Preparation uses approximately70/15/remainder with100train-group/20val-group caps,
minimum8test groups and exact duplicate grouping. Previously train/val-assigned
images remain in their original roles. Only Leopards/watch hit100training groups;
uncapping them cannot turn these20 categories into a10,000-image training task.
All1937 source photos in these classes are represented across the frozen splits.
Never move the old seen test into fitting, rescreen more categories by their scores,
overwrite the existing split or claim augmentation is new independent data.

Small validation supports make per-class recall coarse (one crab sample=10pp).
This limits precision but does not explain away the measured27-29pp aggregate
generalization gap. Data scarcity/imbalance is a plausible factor, not a proven
sole cause. Capacity comparison preserves this data to avoid changing two factors.

## Larger balanced option retaining20 output windows and full-resolution phases

Primary candidate to examine is CIFAR-100 using its **official20 coarse labels**,
not a validation-picked set of20 fine categories. The
[author's specification](https://cave.cs.toronto.edu/kriz/cifar.html) defines100fine
classes,500train/100test images per fine class, grouped into20superclasses with
five fine classes each. Thus all official train images give50,000photos,2500 per
coarse class, and10,000officialtest photos. The original images are RGB32x32.
This preserves20 fixed output detector windows and can retain every existing
full-resolution main phase and router parameter; the label semantics/data change.

Proposed new frozen data identity: all20officialcoarse categories, all50,000official
training images retained for train/validation; nominal450fine-train/50fine-val per
class ->45,000train/5,000val. Exact original-pixel duplicate connected groups need
inspection before declaring these exact counts and no cross-role leakage.
Official test remains separate, never selects categories/augmentation/profile/PT.
Author archive/metadata checksums and preparation manifest/SHA required. Existing
Caltech artifacts, pixels, roles and tests remain untouched. No extra class screening.

Limitations:32x32 has less spatial detail than Caltech; upsampling cannot create
detail. Coarse categories and their within-class variation define a different
task, so its accuracy cannot be called a controlled improvement on Caltech. Bigger
data does not guarantee75% or cure optical optimization. Use fixed input encoding
and full-size graph, first check sample input/physical contract and actual timing.
45,000train is~36times the current epoch size; do not silently promise150epochs in
four hours. Declare a fresh bounded profile from actual throughput/optimizer-step
budget before launching this separate data experiment. Capacity training runs on
the frozen Caltech data first; no fourth GPU job or simultaneous new-data search.

This option is a concrete preparation, not an already downloaded/verified dataset,
trained result, approved full suite, or automatic choice of an easier benchmark.
Final network size remains the human-specified architecture. Any coarse-grid gain
is mechanistic evidence for discussing regularization/extra data, not automatic
permission to replace that architecture.
