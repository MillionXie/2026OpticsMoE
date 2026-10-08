# ABO RGB late-branch physical-CCD probes

## Fixed optical contract

The starting model is alpha39 SHA-256
`1c8f95ebba49b660d590cd1963cb70780dde57dab9d90571b6788c385e01a783`.
Its original RGB image enters the frozen frontend; the same token stream runs
through six optical/electronic stages with in-block optical alpha 0.39. The
physical bank contains 1,600 genuine gallery images and 800 genuine queries,
each captured at all six stages (14,400 CCD frames). The late RGB branch is
inserted **after** the sixth stage, so all upstream weights, SLM images and
raw CCD frames remain identical and are reused read-only.

```text
RGB ── frozen original frontend ── six optical/electronic stages ── last token tensor ─┐
  └── lightweight RGB feature adapter ─────────────── image-token tensor ─────────────┤ 0.5 / 0.5
                                                                                      └─ original electronic readout ─ descriptor ─ retrieval
```

Only the 49 image-token positions are fused at 0.5/0.5; prompt-token positions
retain the sixth-stage output. This is not the earlier two-model output-level
descriptor fusion. The first probe used a new 32,256-parameter CNN on 14×14
RGB pixels and tuned the existing 73,792-parameter readout projection. It
selected by the same 1,200 TRAIN-fit / 400 TRAIN-validation split as the prior
readout adaptation. Validation R@1 was 0.8000 (prior readout-only 0.8100);
retrospective old-query physical R@1 was 0.7500. It is not selected as final.

The second probe uses the **existing frozen** frontend RGB patch embedding,
then a small trainable adapter to 49 tokens. Its selection uses only the
1,200/400 TRAIN split. Its best validation R@1 is 0.8175 at epoch 21, versus
0.8100 for the original readout-only adaptation, a small 3/400-image gain.
The adapter has 201,152 parameters and the existing final projection has
73,792 trainable parameters. The selected adapter checkpoint is saved locally
as `rgb_frozenpatch_late_best_20260929.pt` (SHA-256
`ac971d3b666a08f68f65e3b52170983fd9a5947052242fbf527e22dd4ad56d7f`).
A single fixed old-query physical replay is complete: R@1 0.7475, R@5
0.9175, MRR 0.81993 on the original 1,600 real gallery/800 real query
capture. This improves only 0.00625 absolute over the old readout-only R@1
0.74125 and remains below the requested 0.80. The fixed replay is recorded
in `rgb_frozenpatch_fixedquery_20260929_report.json`; it must not be used to
tune or select another variant. The 800 old queries were already
evaluated during development and cannot be called independent generalization.
No new optical acquisition is required for either late-branch probe; any
future change to the six upstream stages would require fresh stagewise
physical capture.
