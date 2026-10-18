# Caltech richest20: alpha .45 restart / floor .4 night review

This snapshot records saved numerical results and actual process exits. The requested 2026-10-18 Beijing 09:00 delivery deadline was missed. None of the newly audited candidates surpassed the retained models. The desired 10 percentage point gaps have NOT been reached.

Data are unchanged: 20 fixed classes, TRAIN3141 / VAL383 / TEST411. Cache SHA256 `38b719b8ad935a0170d059d0e4fffa22f6f22640f979603f7f2543fc2d3f2445`. Manifest SHA256 `d4921172a9cc8aa427b89e03772733837be64a3fdaaf826d91050e44df2c459c`. Router occurs once; E/G repeats 1, 2, 3 times for 2/4/6 layers. The existing 5140-parameter linear readout remains present in every model. All-electronic parameter totals are 5160/5180/5200. No new inference head, backbone or data split was introduced.

## Retained models

These are explicitly the PREVIOUS retained models, not fresh gains from this night. Controls completed 50 epochs, selected E45, and were never retrained or tested again in this night. Residual models had extra optimization, teacher knowledge and different alpha constraints; this is not an equal-budget causal optical-residual ablation.

| Depth | Residual actual / selected optical epoch; head iterations | Residual train / val / test (%) | Frozen control train / val / test (%) | Test difference |
|---|---|---|---|---|
| 2 | 70 / 70; head 200 | 97.2939 / 78.8512 / 78.8321 | 77.9051 / 71.0183 / 72.7494 | +6.0827 pp |
| 4 | 80 / 80; head 178 | 94.8106 / 77.2846 / 79.8054 | 79.7835 / 70.7572 / 73.9659 | +5.8394 pp |
| 6 | 80 / 80; head 184 | 95.3200 / 78.3290 / 80.7786 | 83.7631 / 73.3681 / 74.9392 | +5.8394 pp |

Full macro/NLL and all 20 class recalls remain in the CSV files. L6 exceeds L4 in retained overall TEST accuracy; L4 remains below 80%, and no depth meets +10pp. On 411 TEST images, +41 correct is only 9.9757pp; at least +42 is required. Retained residuals need 17/18/18 more correct predictions to meet the respective +10pp targets.

## Actual new attempts and diagnosis

Three inherited-encoder alpha-.45 warm runs completed 20 new epochs each (3 electronic-only + 17 joint). Their original audit entry had a missing math import; the failure is preserved, and separate source-identified independent saved reviews passed without any model forward. Original failure files were not replaced with passed markers.

Three fresh sigmoid-alpha models then completed 50 epochs each. All started at .45 with allowed effective alpha above .4001, but strong augmentation/noise/dropout left clean TRAIN around 70–77%, so this stage was underfitting rather than solely overfitting. Sigmoid expert-stage coefficients approached the lower bound.

Three additional fresh models completed 50 epochs each using direct bounded alpha coordinates, first-10-epoch .45→.65 curriculum and milder regularization. Independent projection/Adam/EMA/8-bit numerical checks passed. Endpoint TRAIN rose to 77–83%, VAL to 73–77%, and TEST to 74–78%, still below the prior models. Learned alpha returned towards .70–.95; merely lowering the floor did not establish that more electronic mixing is better. Direct-alpha state keys contain physical coefficients and MUST use the ContainedMoE/Alpha40MoE loader in caltech20_rich_projected_alpha_contained, not a sigmoid-logit loader. These candidates remain archived, not deployed.

Finally six frozen-encoder fits optimized ONLY the existing 5140 head, max200 LBFGS iterations each, from each full-E50 EMA endpoint. All other parameters stayed bitwise identical. TRAIN/VAL optical features were extracted on the same physical GPU as their endpoint evaluations. TEST used saved features only, no new optical TEST forwards or TEST images. Every fit and cached TEST result passed its own independent objective/gradient/prediction audit. These fits improve some endpoint scores but still do not beat retained models. Stronger clean head fitting can reintroduce a large TRAIN/VAL gap, so a single-head bottleneck does not fully explain the result.

Every fresh-E50 endpoint and same-head iteration count is recorded in trial_comparison.csv, not hidden behind a retained-reference score. Six joint-run epoch histories are exported without interpolation in epoch_metrics.csv. The original failed device-anchor/import entries are preserved; no automatic rerun or threshold relaxation was used.

## Interpretation and identity

The user explicitly authorized TEST-based checkpoint choice. These are TEST-informed development results, not independent unseen generalization. TRAIN/VAL/TEST class proportions differ; macro and per-class support are reported. Exact ID/pixel separation does not establish subject or near-duplicate independence. No cross-seed stability is claimed. Frozen controls, old accepted models, raw histories, data, failures and claims are protected.

The six fresh joint runs and six head-fitting pipelines were checked against their original zero producer/audit/test returncodes and actual process exits. The three warm fits also have raw fit returncode zero and separate completed independent saved reviews; their original audit/controller failure remains recorded as a failure. Earlier failed admission/entry attempts are not counted as successful model runs. Final global project GPU count is zero; A100 project count is zero. No automation was created or resumed. Source leaves were published through Git; raw JSON, checkpoints, features, sample IDs and full logs remain private.

Current source-review commit: `825641a76c1a3110ef82a396594545046146bdfc`; source closure 337. Source HEAD is not substituted for each run’s actual training commit. All selected model paths, exact SHA256, loaders and original source commits are in accepted_models.csv. Original scores and model bytes were not overwritten.
