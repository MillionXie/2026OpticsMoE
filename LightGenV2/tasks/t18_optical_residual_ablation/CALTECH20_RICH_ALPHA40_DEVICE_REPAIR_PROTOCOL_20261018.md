# Alpha40 training-only anchor device lifecycle repair

Original308-source run caltech20_rich20_alpha04_electronicWarm_vs_fresh_s17_20261018
failed in all three actual GPU numeric admissions before any optimizer step,
history/PT creation or TEST forward. CPU admission and TRAIN/VAL diagnosis truly
passed and remain valid. Failure stack points to RobustMoE.raw: initialize()
cloned phase_anchors on CPU, then model.cuda() migrated only registered tensors.
Python-dictionary anchors remained CPU. This is an implementation device error,
not an unsuccessful trained strategy, dataset or numerical-science failure.

Original nine sources/output/claims/failures/raw exit1 stay frozen. New four
leaf files implement a separate model subclass _apply that migrates every
nonpersistent anchor with the actual Module.cuda/to transformation. State keys,
learned count, mathematical forward, alpha bounds/start.45, loss, augmentation,
budgets and original numeric thresholds unchanged. context() additionally
rejects any mismatched device before forwarding. No registered-buffer addition.
The original exact source kernels are compiled in the new module namespace,
not edited or monkey-patched in their owning modules.

Independent new CPU proof uses actual .to(float64) lifecycle to check all
anchors follow parameter dtype and values exactly, parameter/state keys stay
identical; stale meta-device anchor is rejected. Plus unchanged actual loader/
augmentation/teacher/freeze/Adam contract. Real CPU model forward/TEST/update0.
Each actual GPU producer again proves all20/macro16/tail5 micro1vs4 gradients,
but in new unique output/claims. New CPU scientific audit independently checks
all saved plans, crop hashes, selection, predictions and Adam/EMA and verifies
the old failure SHA/actual exits both before/after. No original failed audit is
retried, no passed receipt fabricated or threshold relaxed. Resource allocation
GPU2/0/1 one own child/card, shared2GiB+800MiB/A1000 unchanged. Parent controls50
and old best models protected. User ongoing overnight optimization authorization
continues; publication source-only Git, private model/data/JSON never uploaded.
