# Rich Caltech20 foreground review, 2026-10-18 approximately11:00 CST

This is a completed-evidence progress snapshot, not the end of human-authorized optimization. No scheduler is used. Current5GPU children on0/1/2/3/4 run three class-prototype candidates and matched stronger-view L4/L6. All five actual numeric gates passed. Previous350/359/368 work completed; original failures retained.

## Best accepted completed results

|Depth|Residual actual / selected E|Residual TRAIN / VAL / TEST %|Frozen control actual / selected E|Control TRAIN / VAL / TEST %|TEST gap pp|
|---|---|---|---|---|---|
|2|80 / 75|97.7077 / 79.8956 / 79.8054|50 / 45|77.9051 / 71.0183 / 72.7494|+7.0560|
|4|90 / 90|95.3836 / 78.8512 / 81.2652|50 / 45|79.7835 / 70.7572 / 73.9659|+7.2993|
|6|80 / 80|95.3200 / 78.3290 / 80.7786|50 / 45|83.7631 / 73.3681 / 74.9392|+5.8394|

L4 is above80%, but L6 does not yet exceed updated L4. No depth reaches10pp. Actual budget includes completed joint epochs after the selected epoch; selected epoch is not an early stopping claim. Prior CPU head fits are inherited, not counted as optical epochs. Accepted-model CSV specifies exact PT SHA and correct alpha-coordinate loader: do not mix direct and legacy sigmoid states. Six-layer incumbent remains the earliest original model at this score.

## Diagnosed evidence and decisions

- Fixed balanced TRAIN100/VAL100 probes (not full-set scores) show abrupt alpha reset from~.94 to.45 destroys accuracy. Preserved coadapted weights/alpha instead. Electronics remains9trainable kernel coefficients per stage plus alpha; same5140 readout, not a large CNN.
- Independent RMS derivative checks preserve bitwise clean inference. Correct derivative changes stage-electronic CE gradient norm by2.146/9.039/4.447 at depths2/4/6, with substantial direction change in deeper models. Adam is scale-adaptive, so these are not equivalent update-magnitude multipliers.
- Full-RMS+clean/augmented dualview improves L2 TEST78.8321 ->79.8054 and L4 79.8054 ->81.2652. Matched L4 detached-RMS control80.7786; no sole-cause attribution or multi-seed significance claimed.
- Rare categories often TRAIN95–100% but VAL25–60%; memorization remains. Pooled within-class feature covariance effective rank2/4/6=9.70/6.97/4.83 is a concentration diagnostic, not proof of information loss.
- Same5140 covariance-regularized heads pass their own fit/test independent science but reduce TEST to77.3723/79.8054/79.3187. Not adopted. Original source364 numeric failure remains failed; source368 exact class-common logit centering passes unchanged3e-6 and its own fresh science.
- Instance teacher feature supervision finishes actual10 extra epochs: L4 best81.0219 and L6 retains80.7786. Weaker/equal local results do not replace global better incumbents.
- Current source373 class-prototype supervision uses TRAIN-only teacher feature centers; all20 centers classify correctly through each retained readout. Targets are fixed, MixUp matched, gradient strength calibrated on TRAIN, no new inference module. Source378 adds matched stronger crop/input-block/feature dropout for L4/L6. These are ongoing, not accepted results.

## Identity and completed receipts

350 training source `bde4feb1544e4fb13b16c4bfcf90e0bab585319d`; all five producer rawrc0/actualexit and own science receipts.
- L2 fullRMS SCI `d2140314eca15268e371f5ee96ef4ed136d0e0fed7a75f2eefaff805e8cd8e19`
- L4 fullRMS SCI `538d23bfa15ced14d4a59f2c16a8af07d775bf0a31a5dc08794e245082fe374a`
- L4 semantic SCI `77ea66f96b9df405b74b8709e4dc2073717a18a86f5353f9fa4e2835da673f88`
- L6 semantic SCI `4033ae4563c2d255a6aeb2e00be3514a3c218b3f9849bf45bac8bf48b17270b0`
- 368 fit SCI `52c944ebc3266d2c748a333cff2cb18f012bd994376b5c1decddf47174ecdff9`; cached TEST SCI `91940de1e8e1b0d1c6aa9e082717231dfdee64aafc85e2241c68dab1619acce8`.
- Ongoing373 source `e25634955a41d87a86c21e3c8ffc1b92fc507481`; CPU `e51cb1ac64f60fb9f53a829000808d9e6327f229f3f8db38c27a7da0a3ccf690`.
- Ongoing378 source `95928378f1cf6f88da70067c6eefcbd190406c46`; CPU `46b52f54b9d31a52c8bc93de9b134833d937a556f142454db4e3b2bdb623e51a`.

## Limits and protection

Fixed richest20/split TRAIN3141/VAL383/TEST411; no roles/classes/images moved. Whole inference stays one router +E/G x1/2/3, same electronic head/branches. Controls are unchanged actual50 and never retested. Residuals receive extra training, pretrained teacher knowledge and different alpha constraints; this is TEST-informed development, not independent unseen generalization or a same-budget pure optical-residual causal comparison. Macro and all20 recalls are preserved in CSV because class supports differ. No proof of near-duplicate/subject independence is added.

No A100/native/lab/foreign process manipulation. Max5 own children one/card0–4 with2GiB+800MiB policy. Actual process exit releases each GPU. Source-only Git; raw JSON, per-sample IDs/features/PT/logs remain private. Shared task README and other windows paths unchanged. Report commit differs from training identity.
