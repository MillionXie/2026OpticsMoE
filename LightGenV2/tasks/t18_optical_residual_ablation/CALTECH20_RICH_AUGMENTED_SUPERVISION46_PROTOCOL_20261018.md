# Augmented-view supervised weight80%, four/six layers

Continue the authorized foreground optimization; no timers. L2 is final, all50-epoch
controls frozen. Same Caltech20 rich3141/383/411 identities and same optical/electronic
graph, head, physical alpha/quantization. No new inference parameters/teacher.

Observation: retained TRAIN95.7/95.6 versus VALID79.1/78.9, large rare-class gaps.
Appearance/erase/zoomout endpoint and direct TEST bank did not beat retained337/333.
Current two-view-target trials have not improved initial TEST scores so far; those
runs remain protected and complete independently. This new hypothesis changes one
semantic factor: use20%clean+80%augmented supervised classification instead of50/50.
Total CE coefficient remains1. Cached teacherKD/prototype stay clean-only, their
coefficients and JS/anchors/phase jitter unchanged. Same TRAIN-count-aware zoom-out.
This tests reduced reliance on clean-image fitting; not a unique-cause claim or a
guarantee. Do not silently combine with both-view teacher/prototype changes.

Two candidates, retained canonical4a224cf97/6 4f9d5e1b, no fresh weights or alpha
reset. Each five local epochs/197macro updates/tail5/newAdam explicit. Micro4 is
admitted by source497 actual TRAIN benchmark:0.90→0.50seconds/macro, gradient
relative1.73e-6, probabilities identical, peak1.61GB; micro8 rejected by unchanged
2GiB policy. Local3090 timing is not a full4090 epoch guarantee. Each actual producer
must still pass its own micro1vs4, all20/tail5 gate and clean export equivalence.

Independent CPU analytic view-gradient and pixel/teacher/prototype/source checks;
own saved science verifies every loss scalar with20/80 weighting, all orders,
teacher targets, masks, phase jitter, Adam/EMA, saved head/CM/NLL and parentSHA.
Evaluate each actual epoch weight once on TEST; maximum integer correct count,
ties preserve earlier canonical. VALID only reports, no ranking or tie-break.
Best/last only, reuse parent saved predictions. TEST-informed development and
additional residual budget, not independent generalization or equal-budget effect.

Queue only these two candidates on authorized available0/1/2/3/4, one owned job
per GPU, global<=5,2GiB+800MiB. A100/foreign processes/lab unaffected. Queue while
previous jobs finish; do not restart or edit them.1800fit seconds fault protection,
preserve failure/claims, no automatic retries. Stop this strategy after short segment
and assess actual gain rather than extend a negative configuration indefinitely.
