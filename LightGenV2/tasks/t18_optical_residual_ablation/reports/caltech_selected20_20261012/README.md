# Caltech20 screened regularized development task

User requested20 classes, better accuracy/generalization, replace8 with12, four
GPUs. New depths2/4/12/16, rho0/.3, fresh common100; original eight fits preserved.

Actual one-pass screening has frozen20 of47 eligible categories (>=60 photos):
Faces, Leopards, Motorbikes, airplanes, bonsai, crab, crayfish, grand_piano,
helicopter, kangaroo, ketch, laptop, lotus, minaret, revolver, schooner,
trilobite, umbrella, watch, yin_yang. Fixed thumbnails train-only centroids scored on validation
improve proxy macro recall29.79% to59.59% after removing difficult classes.
This is a screening proxy, NOT optical-model performance or an independent score.
No optical/test score or preference for rho participates; choices freeze before fit.

3719 retained photos, train1372/val283/test2064; variable class support, all photos
of the selected categories retained. BOTH previous50 and20 train/val roles pinned,
none enter new test, duplicate original/cached pixels never cross roles.
ManifestSHA91f7aa36a97713e300699367904c8866551996de199b9630c0545bb24dc5f60e;
cacheSHAa81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1.
Data at /DATA/DATA1/guest3/t18_caltech_selected20_20261012; no arrays/PT in Git.
Class screening uses validation and previous Caltech20 test was observed: later
held-out results remain exploratory development, not independent task selection.

Common regularization: MixUp alpha.2, label smoothing.08, phase smoothness.05,
affine20deg/5pixels/10%scale. Seed17, lr.002, EMA.95, capture.2 and full optical
ASM/OEO/9experts/20windows retained. Training graphs have no CNN/Linear/pretrain.
12 layers have6 expert/global cycles,61 phase parameters,2,649,088 scalar parameters.
Training-only MixUp happens on fixed encoded amplitude; inference is unchanged.
Class soft-target weights and sampleweighted macro-tail accumulation are checked.

Windows14 CPU checks passed,2 nativeTorch checks skipped; Linux13 CPU checks passed
including12 graph contract and mixed soft-target gradients. Fresh GPU forward and
every phase gradient thresholds remain probability<=1e-6/relative<=.02/cosine>=.999;
production uses only the largest common passed microbatch per pair. Four GPU slots
0/1/2/4, Torch3GiB+800MiB, no displaced projects or environment changes.

Fit queue is train/val only, common100/36,000sec per arm; no automatic test, restart,
extra candidates, or category changes. All eight CPU saved-artifact audits and
selection/global locks precede each selectedEMA once-only test. Final plotting
must show macro/NLL/support and gap, no promised accuracy or residual advantage.
Protocol: ../../CALTECH_SELECTED_PROTOCOL_20261012.md.
