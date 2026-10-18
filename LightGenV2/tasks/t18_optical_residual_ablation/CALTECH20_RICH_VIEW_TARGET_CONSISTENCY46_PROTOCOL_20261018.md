# Four/six-layer two-view target allocation, 2026-10-18

Human scope: foreground, up to five shared non-A100 GPUs; only four/six layers.
L2 is final and must not be constructed, forwarded, trained or tested. Controls50
remain frozen. Caltech20 rich identities and the physical optical/electronic graph,
normalization, head, alpha bounds and phase quantization stay unchanged.

Evidence: retained TRAIN accuracy is about95.7/95.6 versus VALID79.1/78.9. Rare
classes include severe train/validation gaps. Prior TRAIN macro gradients do not
show prototype domination. Stronger KD, appearance mixtures, erasing and zoom-out
endpoints did not beat the retained TEST scores. This is a hypothesis about
augmentation supervision, not a claimed unique cause or guaranteed10pp result.

Four isolated short continuations use the same TRAIN-count-aware zoom-out and
same canonical parents as source483 (four a224cf97, six4f9d5e1b). Each performs
five local epochs,197 macro Adam updates/epoch, tail5, micro2; new Adam is explicit,
trained weights and physical alpha are not reset. Both student views already exist,
so the method adds loss terms but no extra model forward or deployable parameter.

`prototype_both` uses0.5(clean prototype + augmented prototype), keeping KD clean.
`KD_both` uses0.5(clean KD + augmented KD), keeping prototype clean. Total KD and
prototype coefficients are unchanged; do not stack both changes. The teacher
targets and prototype targets remain the same cached TRAIN-only clean-image,
same-ID/permutation/MixUp targets. Neither teacher nor targets enter inference.

Numeric admission compares actual new loss/gradients at micro1/micro2 including
all20 labels and tail5, clean export equivalence and exact parent identity. CPU
independently checks two-view analytic gradients and independent resize/pad pixels.
Each actual producer has its own gate. Saved science independently reconstructs
every order/augmentation/teacher target/total-loss allocation, Adam/EMA, predictions,
selection and all artifact identities. No inherited proof is claimed as a new run.

Legacy `view_info.KD_clean` is explicitly the allocated KD scalar used by inherited
loss accounting. True raw clean/augmented KD and prototype terms are separately
saved; the independent auditor checks the arithmetic and accurately reports which
student views received supervision. Teacher reference input stays clean in both.

Latest human authorization selects directly by integer TEST correct count every
local epoch; equal count preserves the earlier retained model. VALID is recorded
only, never used for ranking. Each actual epoch weight has one TEST pass; parent
predictions are reused. Best/last only. TEST-informed development, not independent
generalization or equal-budget optical residual evidence; report all TEST queries
and negative results. No label/role changes, no control weakening, no fresh weights.

Four candidates are queued once onto available approved0/1/2/3/4, one owned child
per card, global maximum5 with2GiB+800MiB guard. A100/foreign processes/lab untouched.
1800s per fit is fault protection. Failures and claims remain; no automatic retry.
