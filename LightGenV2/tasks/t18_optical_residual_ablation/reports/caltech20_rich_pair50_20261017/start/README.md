# Confirmed return to the quoted richest20: fresh paired50, 2026-10-17

Human explicitly confirmed the quoted20 classes/roles and explicitly selected **only no-residual alpha>.85; residual alpha>.7** after the comparison limitation was explained. This is the current task. The other original20/60+20 experiment remains preserved and is not the default result for this run. No scheduler automation is used.

|Identity|TRAIN/VAL/TEST|Budget|Alpha constraints|
|---|---|---|---|
|Quoted rich20 historical table|3141/383/411|controls100; residual2:100+40head,4:100+20head,6:120|both>.7|
|Last reported original20 table|1372/283/2064|controls60; residual80 plusL2ridge|both>.7|
|Current confirmed richest20|3141/383/411|fresh both50; then freeze controls50|rho0>.85,rho.3>.7|

Both old tables have20 labels, **only11 overlap**, not the same task. The quoted historical residual TEST80.5353/81.7518/80.2920 was already not monotonic in depth; only its no-residual TEST75.1825/77.8589/78.3455 increased monotonically. Do not claim6-layer loss of advantage first appeared in the recent task. Old100 baseline had19700Adam versus original20 baseline60*86=5160; latest residual80*86=6880. Current rich50 has9850Adam per arm,197/E/tail5. Different training data, updates and extra-head budgets make direct before/after performance attribution invalid. Reporting should have kept these identities explicit.

Current classes: Faces, Leopards, Motorbikes, airplanes, bonsai, brain, buddha, butterfly, car_side, chandelier, ewer, grand_piano, hawksbill, helicopter, kangaroo, ketch, menorah, starfish, trilobite, watch.

Rich-only relative to original20: brain, buddha, butterfly, car_side, chandelier, ewer, hawksbill, menorah, starfish.
Original20-only: crab, crayfish, laptop, lotus, minaret, revolver, schooner, umbrella, yin_yang.

## Actual source and start evidence

Run **caltech20_rich20_current246_s17_pair50_asymmetricAlpha_20261017**. Kernel209 sourcecd75fef5f3b845273e655fe2f3a045dfdf588626, unchanged original cache38b719b8ad935a0170d059d0e4fffa22f6f22640f979603f7f2543fc2d3f2445/manifestd4921172a9cc8aa427b89e03772733837be64a3fdaaf826d91050e44df2c459c both actually hashed on server before launch. No prepare/re-split. All six freshseed17, effectivealpha initial.9 both (raw logits necessarily differ by bounds; all other initial tensors match). rho0 learned .8501-.95;rho.3 .7001-.95. **Difference includes alpha constraints, not pure optical-rho causal evidence.**

CPU own prooffceda3143970a3d14578ffb582b98601af8229808592c3a7bc66b2691fe3ce2e passed:12TRAIN/blackforward, exact independent ABO forward/gradients for both bounds, shape/seed/nonalpha initialization/20labels/NumPylinear;0optimizer/TEST. Supervisor4187057 actual launched. Initial actual four producer PIDs/physicalGPUs: L6rho03=4188728/GPU0, L6rho0=4188794/GPU1, L4rho03=4188860/GPU3, L4rho0=4188938/GPU4; L2pair queued. Each four producer own gate actually passed; IDs are historical and must be matched to live commands/UUIDs at later checks. No failure at this start snapshot. A100 physical6 prohibited;2/5 unused in this four4090 schedule. No foreign process changes. No completion/target-success claim from early metrics.

One router; E/G repeat1/2/3, four224experts/global478/RGB100/8bit/OEO. Identical electronic module shapes in both arms:9weight biasfree per-stage residual + learned scalar alpha and terminalLinear256to20=5140. Total electronics5160/5180/5200; optical479364/908552/1337740. No network swapped after capture, no inference teacher. Phase and electronic weights jointly trained in baseline50, not random unused electronics. ABO detachedRMS normalization retained; alpha is normalized feature mixing, not literal optical power percentage. Current explicit profile/own checkpoints are authoritative, not aggregate reportHEAD.

## Prepared follow-up, not already running

Separate CPU-only public training teacher212 sourcec8b7ae952cd06d8392dea36d57a396ea16681c61 actually finished: TRAIN100/VAL98.9556, selected teacher head22/100steps, teacher TEST0/studentmodifiedfalse/GPUadded0. Its own source/data/IDs/head arithmetic sciencea7e07a10c57036a2e84ac2298a16c2509d45f8500db6225e154bbc175f411fee and selectedheadb1d02a921413f5c8a7118878aee585f17156875c8e65581d0ebd7f701aacdf61, producer/supervisor actual exits verified. These are **teacher**, not student accuracy/epochs. The baseline50 is teacher-free.

Residual-only20 review segment217 sourcee80751ae4a44a45fc53d4d38c4a3ef0d9811a0d3 prepared with hard admission: sixfull50+ownSCI+onceTEST/frozencontrols50+allactualexit+teacherSCI before real CPU restoration/production. It cannot start early or replace actual50 with old60/100. Synthetic math admission is a separate unique proof; no future actual-training or target-success claim. After first50, extra residual training/teacher knowledge plus alpha-constraint asymmetry must be disclosed. FirstminimumVAL EMA balancedNLL, not maximumTESTepoch. No blind extra-head fitting or arbitrary new classes. All old sources/PT/data/claims/failures/plots protected; private JSON/IDs/PT/full logs not published.
