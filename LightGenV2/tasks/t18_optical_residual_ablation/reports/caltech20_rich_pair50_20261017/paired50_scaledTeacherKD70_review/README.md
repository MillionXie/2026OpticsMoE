# Quoted richest20: paired50 and residual-only scale-calibrated KD20 review

Report created 2026-10-17T23:18:49.107301+08:00. All six paired50 models, their own scientific audit and selected-model tests actually completed. Three residuals then restored full LAST50 and completed20 more; own scientific/test receipts verified and all CPU/GPU processes exited. This is a review segment, not a claim that all human accuracy/gap targets were met.

## Task identity and comparison contract

Human explicitly restored the quoted richest20 classes and roles, TRAIN3141/VAL383/TEST411. This differs from the previous original20 TRAIN1372/VAL283/TEST2064. The two tasks share11 labels;9 differ. Old quoted residual budgets were100optical+40head /100optical+20head /120optical, controls100, so their old scores are historical references rather than new50 results. No category or image-role movement in this run. Current classes: Faces, Leopards, Motorbikes, airplanes, bonsai, brain, buddha, butterfly, car_side, chandelier, ewer, grand_piano, hawksbill, helicopter, kangaroo, ketch, menorah, starfish, trilobite, watch.

One optical router then expert/global repeated1/2/3; same deployed intermediate9-weight electronic residuals and terminalLinear256to20=5140. Optical phase counts479364/908552/1337740; electronic5160/5180/5200. Both arms retain electronics. Learned alpha constrained>.8501 for no optical residual and>.7001 for optical residual, both effective initialalpha.9. ABO detached per-sample RMS normalization retained; alpha is a feature-fusion coefficient, not literal optical power. Different bounds/derivatives plus extra residual training/pretraining mean these comparisons are not pure equal-budget optical-rho causal evidence.

## Complete measured models

All metrics below refer to each row's same selected model. Actual epochs are distinct from selected epochs. Selection is firstminimum VAL EMA balancedNLL within the complete prefix, not maximum TEST accuracy. Accuracy units%, difference units percentage points.

| Depth | Model | Actual/extra epochs | Selected E | TRAIN | VAL | TEST | TEST macro | TEST gap vs frozen50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | paired50_residual | 50/0 | 45 | 81.4072 | 74.6736 | 74.6959 | 64.6750 | +1.9465 |
| 2 | frozen50_control | 50/0 | 45 | 77.9051 | 71.0183 | 72.7494 | 61.6355 | +0.0000 |
| 2 | KD70_residual_candidate | 70/20 | 70 | 84.1452 | 75.1958 | 75.9124 | 64.8912 | +3.1630 |
| 4 | paired50_residual | 50/0 | 45 | 84.9093 | 74.4125 | 77.1290 | 65.9487 | +3.1630 |
| 4 | frozen50_control | 50/0 | 45 | 79.7835 | 70.7572 | 73.9659 | 63.2485 | +0.0000 |
| 4 | KD70_residual_candidate | 70/20 | 66 | 88.3477 | 75.4569 | 78.5888 | 68.8632 | +4.6229 |
| 6 | paired50_residual | 50/0 | 47 | 88.8571 | 75.9791 | 77.3723 | 66.4233 | +2.4331 |
| 6 | frozen50_control | 50/0 | 45 | 83.7631 | 73.3681 | 74.9392 | 62.8733 | +0.0000 |
| 6 | KD70_residual_candidate | 70/20 | 66 | 91.5314 | 78.3290 | 80.0487 | 69.8042 | +5.1095 |

## Acceptance and limitations

Before new candidate tests, strategy acceptance was declared: preserve each residual paired50 TEST accuracy; if the KD candidate lowers it, retain the paired50 incumbent and publish the full negative candidate. This is TEST-informed strategy acceptance, not selecting the highest TEST epoch. No-residual controls remain exactly50 and are never retrained or retested during residual optimization. Public teacher is training-only and never enters deployment. Teacher head trained100 CPU updates/selected22, TRAIN100%/VAL98.9556%, TEST0; these are not student metrics/epochs. TRAIN logit-scale ratio~4 motivated teacherT16/studentT4; independent own synthetic proof rejects mistaken T4 teacher targets. Old prepared217 T4/T4 was never real admitted/trained and remains preserved. Dataset selection and prior TEST observations, current once-per-selected-parameter tests and strategy feedback make all reported TEST values development evidence, not untouched independent generalization. Exact ID/pixel disjointness does not prove absence of near-duplicates or subject overlap. Full per-class supports/recalls are in class_recall.csv.

Accepted candidate by this predeclared criterion:
- Depth2: actual70, selectedE70, TRAIN84.1452/VAL75.1958/TEST75.9124%, gap+3.1630pp;10pp target unmet.
- Depth4: actual70, selectedE66, TRAIN88.3477/VAL75.4569/TEST78.5888%, gap+4.6229pp;10pp target unmet.
- Depth6: actual70, selectedE66, TRAIN91.5314/VAL78.3290/TEST80.0487%, gap+5.1095pp;10pp target unmet.

## Actual gradient review

- L2_rho03: phase/electronic mean CE-KD cosine [0.5119663936001688, 0.5116219997175057]; negative-batch fractions [0.0002538071065989848, 0.0]; mean adaptive scales [0.09999007049132608, 0.09998414658571457]; maximum KD/CE norm fractions [0.24999999999924083, 0.24999999999993344]. These are measured training-gradient summaries, not independent evidence of causal generalization.
- L4_rho03: phase/electronic mean CE-KD cosine [0.49679964847115454, 0.4789865927813662]; negative-batch fractions [0.0, 0.0]; mean adaptive scales [0.09997440947044486, 0.09996750568618253]; maximum KD/CE norm fractions [0.24999999999954792, 0.24999999999993033]. These are measured training-gradient summaries, not independent evidence of causal generalization.
- L6_rho03: phase/electronic mean CE-KD cosine [0.483027173860373, 0.454834455278443]; negative-batch fractions [0.0007614213197969543, 0.0012690355329949238]; mean adaptive scales [0.09997429591463167, 0.09995748694056951]; maximum KD/CE norm fractions [0.24999999999943964, 0.24999999999993086]. These are measured training-gradient summaries, not independent evidence of causal generalization.

Four-layer accepted TEST>80: False. Six-layer accepted TEST exceeds four-layer: True.

## Identity and saved evidence

Baseline training root Git `cd75fef5f3b845273e655fe2f3a045dfdf588626`; residual KD root Git `76122d8bb6b367a78677a04726184dc1987be76e`. Per-arm actual Git identities are in depth_comparison.csv; later child HEAD can include report/preparation commits while frozen training source bytes remain identical. Report commit differs from training source identity.

Paired50 science `5fd2b94db0c05d02b9427776bd18b34e0eeb1e1a10c5cb5d71f0cda44a1f5d63`; paired50 saved TEST audit `5ff4700bbd47a35bd27acbccb1e62de6177e9e0f747d6d7de7d3c474a656f489`.
Residual KD science `665fcab3a43b7b959fb3200b2b2af4db7a913f7a231bd69222116ed105c6de00`; residual KD saved TEST audit `2eb559a454e105a88d4dcb99ae0f9f14a6fb029afe20fb3b13cc80a9cdeb3724`.

Source/config/rawEMA/checkpoint/initialization/Adam197steps-per-epoch/tail5/allorders-affineMix-flips-masks/LR/budget/savedID-label-argmax-CM-support-NLL/firstminimum and newKD targets/gradient-budget/singleAdam boundary were checked by their own exclusive auditors. Readonly collectors rechecked receipt hashes and actual exits without new optical forwards/audits. Baseline9850Adam each; residual13790total,3940new. Every selected new parameter evaluated once; unchanged digest reuses baseline result. Raw arrays, IDs, PT, gradient traces and complete logs remain private.
