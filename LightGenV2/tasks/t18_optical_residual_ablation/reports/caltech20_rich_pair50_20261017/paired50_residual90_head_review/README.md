# Quoted richest20: frozen paired50 and residual continuation90 review

Created 2026-10-17T23:57:35.035401+08:00. Actual production/science/tests and CPU/GPU exits have all been verified. Completion is distinct from achievement of all performance goals.

Fixed quoted richest20, TRAIN3141/VAL383/TEST411; no classes, roles, input images or evaluation metrics moved. Single router then E/G repeated1/2/3. Optical phases479364/908552/1337740; intermediate electronic residuals and finalLinear256to20=5140 retained in both arms, total electronics5160/5180/5200. Learned feature-fusion alpha>.8501 nores and>.7001 residual, same initialeffective .9 and ABO detached per-sample RMS. These asymmetric constraints, extra residual budget and public pretrained training-only teacher are not a pure equal-budget optical-residual causal comparison. No inference changes, A100 or lab actions.

## Every measured selected model

Selection within every complete prefix is firstminimum validation EMA balancedNLL, not maximum test accuracy. Actual training epochs and selected epoch differ. Units%, gap in percentage points.

| Depth | Model | Actual/extra | Selected E | TRAIN | VAL | TEST | TEST macro | TEST gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | paired50_residual | 50/0 | 45 | 81.4072 | 74.6736 | 74.6959 | 64.6750 | +1.9465 |
| 2 | frozen50_control | 50/0 | 45 | 77.9051 | 71.0183 | 72.7494 | 61.6355 | +0.0000 |
| 2 | calibratedKD70_candidate | 70/20 | 70 | 84.1452 | 75.1958 | 75.9124 | 64.8912 | +3.1630 |
| 2 | same_head70_anchor1 | 70/20 | 70 | 97.2939 | 78.8512 | 78.8321 | 67.1512 | +6.0827 |
| 2 | same_head70_anchor10 | 70/20 | 70 | 92.8367 | 77.5457 | 77.8589 | 66.3919 | +5.1095 |
| 2 | depthwiseKD90_candidate | 90/40 | 89 | 85.4505 | 76.5013 | 77.6156 | 65.4922 | +4.8662 |
| 4 | paired50_residual | 50/0 | 45 | 84.9093 | 74.4125 | 77.1290 | 65.9487 | +3.1630 |
| 4 | frozen50_control | 50/0 | 45 | 79.7835 | 70.7572 | 73.9659 | 63.2485 | +0.0000 |
| 4 | calibratedKD70_candidate | 70/20 | 66 | 88.3477 | 75.4569 | 78.5888 | 68.8632 | +4.6229 |
| 4 | same_head70_anchor1 | 70/20 | 66 | 88.3477 | 75.4569 | 78.5888 | 68.8632 | +4.6229 |
| 4 | same_head70_anchor10 | 70/20 | 66 | 93.9191 | 77.0235 | 78.8321 | 68.0835 | +4.8662 |
| 4 | depthwiseKD90_candidate | 90/40 | 66 | 88.3477 | 75.4569 | 78.5888 | 68.8632 | +4.6229 |
| 6 | paired50_residual | 50/0 | 47 | 88.8571 | 75.9791 | 77.3723 | 66.4233 | +2.4331 |
| 6 | frozen50_control | 50/0 | 45 | 83.7631 | 73.3681 | 74.9392 | 62.8733 | +0.0000 |
| 6 | calibratedKD70_candidate | 70/20 | 66 | 91.5314 | 78.3290 | 80.0487 | 69.8042 | +5.1095 |
| 6 | same_head70_anchor1 | 70/20 | 66 | 91.5314 | 78.3290 | 80.0487 | 69.8042 | +5.1095 |
| 6 | same_head70_anchor10 | 70/20 | 66 | 95.6383 | 78.3290 | 79.8054 | 69.4790 | +4.8662 |
| 6 | depthwiseKD90_candidate | 90/40 | 66 | 91.5314 | 78.3290 | 80.0487 | 69.8042 | +5.1095 |

## Preservation and target assessment

Predeclared sequential strategy acceptance order is paired50, calibrated70, weak-anchor samehead70 (completed first), strong-anchor samehead70, then joint90. A new selected strategy must improve validation balancedNLL and preserve the CURRENT accepted residual TEST score, otherwise retain the incumbent. This adds a performance-preservation filter; no epoch is selected by highest TEST. Actual/extra counts refer to optical+joint stages; head iterations are separate in CSV, never extra optical epochs. Predeclared strategy acceptance preserves the previously accepted residual test score: retain the incumbent when a new strategy lowers it, and disclose the full negative candidate above. This is test-informed strategy acceptance, never an epoch search by test. Controls remain50 and are never retrained/retested. Each new selected parameter digest gets one test; unchanged parent parameters reuse actual saved results. Prior dataset selection/test observations and current feedback mean development evidence, not untouched independent generalization. Exact cross-role IDs/pixels disjointness does not establish no near-duplicates or subject leakage. Full per-class train/val/test supports/recalls are in class_recall.csv.

Accepted by this predeclared preservation criterion:
- Depth2: actual70, selectedE70, variantsame_head70_anchor1, headLBFGSattempt200, selectedSHAa823f324c9d8e363a8df0a6d72dcd91156f96c910cfb29fb2050dd84e90018fc, TRAIN97.2939/VAL78.8512/TEST78.8321%, macro67.1512%, gap+6.0827pp;10pp unmet.
- Depth4: actual70, selectedE66, variantsame_head70_anchor10, headLBFGSattempt200, selectedSHA82c74302c6ceda4d3eb70830f50ddb7bbf5e8b83aa1e55a55012140db4009ad8, TRAIN93.9191/VAL77.0235/TEST78.8321%, macro68.0835%, gap+4.8662pp;10pp unmet.
- Depth6: actual70, selectedE66, variantcalibratedKD70_candidate, headLBFGSattempt0, selectedSHA19faae8dd788835b07d450ee7e48e0bacbc124924664f1b52f341c6bcae9a2f5, TRAIN91.5314/VAL78.3290/TEST80.0487%, macro69.8042%, gap+5.1095pp;10pp unmet.

Four-layer TEST>80: False; six-layer TEST>four-layer: True.

## Evidence-driven recipe and resources

Stage70 measured CE/KD mean cosines ~.45-.51 with scale cap.1 active almost all steps. Under a hard admission requiring positive mean cosine>.1, mean KD/CE norm<.20 and oldcap-active>.5 in both groups, stage90 raises cap to.5 but retains the <=25% CE gradient norm budget. TeacherT16/studentT4 calibrated from TRAIN logits; correct/conf.5 gate and exact same MixUp pairing. Teacher is pretrainedResNet18 plus a frozen trained20-class head, only during training; teacher head100CPUupdates/selected22 is not student epochs. Prepared earlier T4/T4 strategy217 never entered real admission/training.

Stage90 restores LAST70 rawEMA/Adam13790 with moments intact; epoch71-90 adds3940 to17730total; scheduler20 reset with declared per-depth rates. L2 MixUp.1/LS.02/featureDropout.05/phaseLR.0005/electronicLR.0002 to ease shallow fitting. L4/L6 retain .3/.05/.1, rates.0003/.0001. All retain TV.02/capture.05/EMA.99/WD.01 and cosine20floor2e-5. Two complete macro CE/KD gradients, one clip/Adam/EMA; no state replay, new head or changed inference. Guard3600seconds protects incomplete failure, does not make incomplete training successful.

Human explicitly permits low-util/shared nonA100 cards: six-layer GPU0, four-layer GPU1, two-layer GPU2; spare5. Frozen old resource modules unchanged via own scoped context. Own guard independent positive/negative proof accepts max4onepercard, rejects A100/duplicates/unknown UUID and >4. No foreign process touched.

## Own scientific and test identities

- paired50 actual root Git `cd75fef5f3b845273e655fe2f3a045dfdf588626`; science `5fd2b94db0c05d02b9427776bd18b34e0eeb1e1a10c5cb5d71f0cda44a1f5d63`; saved TEST audit `5ff4700bbd47a35bd27acbccb1e62de6177e9e0f747d6d7de7d3c474a656f489`. Raw producer exits0 and actual CPU/supervisor exits verified. Report HEAD may differ from training source.
- calibrated70 actual root Git `76122d8bb6b367a78677a04726184dc1987be76e`; science `665fcab3a43b7b959fb3200b2b2af4db7a913f7a231bd69222116ed105c6de00`; saved TEST audit `2eb559a454e105a88d4dcb99ae0f9f14a6fb029afe20fb3b13cc80a9cdeb3724`. Raw producer exits0 and actual CPU/supervisor exits verified. Report HEAD may differ from training source.
- depthwise90 actual root Git `fd8336d28f25923a67f360140a70e4da8f424098`; science `82913c14db061153dc3fdb98d2922a718b4895c94d647b684206eb40e02fccbe`; saved TEST audit `237b0db07ef8d21fc445f382803f3ee73d4cdac13d45d9df0fd9889c6d18f661`. Raw producer exits0 and actual CPU/supervisor exits verified. Report HEAD may differ from training source.

Every own source/config/prefix/rawEMA/initialshape/all12/21/30gradients/Adam197steps-tail5/completeorder-affineMix-flip-mask/LR and savedID-label-argmax-CM-support-macroNLL/firstminimum was independently checked; zero additional auditor optical forwards/test images. Per-case configs explicitly replayed by own auditor. Full raw arrays, IDs/PT/gradient traces/logs remain private. No old audit/cpu/producer/test helper rerun.

## Last segment gradient diagnostics

- L6_rho03: {"groups": ["phase", "electronic"], "mean_cosine": [0.4693958027563464, 0.44291833182585183], "negative_fraction": [0.0012690355329949238, 0.003045685279187817], "mean_scale": [0.4052592111390429, 0.409578703374999], "maximum_KD_CE_norm_fraction": [0.24999999999969982, 0.24999999999993042]}; endpoint TRAIN92.8685/VAL78.5901, separate from selected-model metrics above.
- L4_rho03: {"groups": ["phase", "electronic"], "mean_cosine": [0.4936656063455949, 0.4780278525092065], "negative_fraction": [0.0002538071065989848, 0.0010152284263959391], "mean_scale": [0.42635778992258094, 0.43193675669080916], "maximum_KD_CE_norm_fraction": [0.24999999999950973, 0.24999999999992725]}; endpoint TRAIN89.7485/VAL77.2846, separate from selected-model metrics above.
- L2_rho03: {"groups": ["phase", "electronic"], "mean_cosine": [0.517672548105957, 0.5288845919898174], "negative_fraction": [0.0, 0.0002538071065989848], "mean_scale": [0.4357678856355187, 0.4629669168873083], "maximum_KD_CE_norm_fraction": [0.2499999999994706, 0.2499999999999236]}; endpoint TRAIN85.5142/VAL76.5013, separate from selected-model metrics above.

## CPU-only same5140 head branches

Saved TRAIN feature diagnosis showed conditional head gradient norms.226/.251/.232 and correlated design condition5.5e6/1.4e7/1.3e7. One cached-feature CPU fullbatch optimization of the SAME head was therefore admitted; all optical, middle electronic and alpha tensors were frozen and their digests checked. No GPU/optical/teacher-backbone forwards and no new inference parameters. TeacherT16/studentT4, .98inverse-countCE/LS.02 + .02KL plus anchor. One final LBFGS candidate, never bestclosure. The measured weak-anchor candidates reached~97% TRAIN with~76-79% VAL, supporting one fixed stronger anchor10/N instead1/N. No other settings changed, no blind scan or class movement. Firstminimum validationNLL gate was preserved; no validation scalar calibration. Both branches and all rejected candidate metrics below are retained.

The original factor10 entry266 failed before real optimization with missing cached-loader alias; all three failure files c0ebfc3ca9dcc399f0743809ff71e56acf26b82cefbc32ece4bf1f56c67a2516 and rawrc1 are preserved. New270 explicitly binds four pure loader/probability/logit/core-digest functions and adds real TRAIN/VAL feature-path admission; objective and thresholds unchanged. Original266 remains FAILED, not relabeled passed.
Each CPU solve attempted at most200 LBFGS iterations/maxeval300; counters and final gradients are reported, not assumed fully stationary. Cached TEST features get one new linear evaluation per new selected digest, not a new optical TEST forward. Unchanged model reuses prior result; this remains test-informed development.
- anchor1/L2_rho03: candidate TRAIN97.2939/VAL78.8512, bNLL1.27855693; accepted by VAL=True; iterations200, finalgradientInf4.5178597e-05; science `90d06704a73c06066936182305cbf11ead222f6647c99dcd336b6043400fe84a`/TESTaudit `e5c58b362164147a9f75c91ef56417e1e369fbe5fad2741559fabd5629e4d187`; selectedSHA `a823f324c9d8e363a8df0a6d72dcd91156f96c910cfb29fb2050dd84e90018fc`; core digest `874ef4f38b30a755fec4d04b7a2468a743e9d71ccffa30d59859e8e35f8679da` unchanged.
- anchor1/L4_rho03: candidate TRAIN97.3894/VAL76.7624, bNLL1.31991723; accepted by VAL=False; iterations200, finalgradientInf5.1002696e-05; science `ca527446e25f51fe3dfd36bea4a0d96e338c910d70b29e2d99f14075377a7d95`/TESTaudit `350809d4390025a952ad9fc978c268b4e1b06950e20aa8eab7fc83d2d5cfd5ad`; selectedSHA `5f6db596d61b2ed2c7726d45f300cdc58aa21f75856ea835106d6fac54765340`; core digest `5e13fb2c57c87adb2d9a5c81985cedfddf7268b53406ee2734a6551677b97dc0` unchanged.
- anchor1/L6_rho03: candidate TRAIN97.4530/VAL78.0679, bNLL1.29111044; accepted by VAL=False; iterations200, finalgradientInf4.9258023e-05; science `d71c933d2d3c118e293334637c75f372b6b1fa2f9aad0ef9d3ac408f1ee351f5`/TESTaudit `c003f55fab1c6822c71b1fff9a850e0a7c6507b03302bcf2afb933751111b0c1`; selectedSHA `fbfb8a5af9d461e7e49f71385d929d9cfc05e6dedec9f5ce81465459fc5a75c9`; core digest `ab58b12017ad48bce74c1973590d9e0882f3c5a74644f34d32e0ce559d3f219b` unchanged.
- anchor10/L2_rho03: candidate TRAIN92.8367/VAL77.5457, bNLL1.17292036; accepted by VAL=True; iterations200, finalgradientInf1.17504e-07; science `dd2696608ee1376cbf9269d1057c899e92d2ec34dad6bb8455c62b63f35977b8`/TESTaudit `e94673872783f5f89ceddb243586228d69926f155ddc81014faef559761563de`; selectedSHA `b87f0f57ee218e6aec07ee9a4c92860b330268840cfa30d1cc94483ee9ad966d`; core digest `874ef4f38b30a755fec4d04b7a2468a743e9d71ccffa30d59859e8e35f8679da` unchanged.
- anchor10/L4_rho03: candidate TRAIN93.9191/VAL77.0235, bNLL1.18345032; accepted by VAL=True; iterations200, finalgradientInf5.6854681e-07; science `9f2eb50dc509a4e29c34b8e7f1480af362fc70c5b00a52c5e4bfae9ccc8750ee`/TESTaudit `f3050412ba829d979d101a2951e9b0ccf1d00035518a0ea2f1bc3a71cadc54e6`; selectedSHA `82c74302c6ceda4d3eb70830f50ddb7bbf5e8b83aa1e55a55012140db4009ad8`; core digest `5e13fb2c57c87adb2d9a5c81985cedfddf7268b53406ee2734a6551677b97dc0` unchanged.
- anchor10/L6_rho03: candidate TRAIN95.6383/VAL78.3290, bNLL1.20145623; accepted by VAL=True; iterations200, finalgradientInf3.625964e-07; science `64968fdaae96af0b0354c8e3f39e61dee20e4acba4ed419ead6e7578e797fb4d`/TESTaudit `ba1a31103eb0941b876be481b68ebbcb405bfa24e0298739cc99f80b63080436`; selectedSHA `93be5e0242ccd8bdf5a786a55e6ee3ea0b4b0037b20d2b6d770ad96b79ee2d36`; core digest `ab58b12017ad48bce74c1973590d9e0882f3c5a74644f34d32e0ce559d3f219b` unchanged.
