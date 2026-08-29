# P12: P11 ImageNet backbone downstream fixed-feedback transfer plan

Updated: 2026-08-29

## 1. Research question

P12 tests one narrow claim across genuinely different output forms:

> After ImageNet pretraining, can the final P11 optical operators be frozen as
> inter-stage backward connectors during downstream adaptation, remain close to
> exact current-operator BP, and consistently outperform matched random optical
> feedback?

P11 is the only source backbone in the main downstream study. P09 and P10 have
already served their architecture-control role and are not additional downstream
method groups.

The exact source is the validation-selected P11 epoch-88 `backbone.pt` from the
completed 90-epoch ImageNet run. Before any downstream run, the implementation
must record the source path, SHA-256, architecture signature, parameter report,
and the eight physical source phases. The frozen Qwen patch/position stem remains
frozen in every method. Unfreezing it would change both the electronic budget and
the scope of the feedback hypothesis, so it is not part of the P12 main table.

## 2. Shared P11 feature interface

The common forward path remains:

```text
224x224 RGB image
-> frozen Qwen Patch/Position Stem (196x1024 tokens; no Transformer)
-> learned 1024->224 adapter
-> eight P11 OEO stages: [token-axis -> channel-axis] x 4
-> task-specific temporary electronic head
```

The optical phase parameters, width-96 electronic residual mixers, constrained
optical gates, token ordering, and three optical banks are loaded unchanged from
P11. The reusable backbone retains 1,204,224 optical phase parameters and a
55.511% optical parameter fraction when the temporary task head is excluded.

Two readout interfaces are allowed:

1. **Global interface.** From the final stage, take the first 196 active token
   rows, learn a three-bank convex fusion, apply token LayerNorm, and concatenate
   token mean and token maximum into a 448-dimensional descriptor.
2. **Spatial interface.** From the final stage, take the first 196 token rows,
   fuse the three banks, restore Qwen block-major order to a true
   `[B,224,14,14]` grid, and attach a lightweight progressive decoder.

The primary FA comparison uses the final stage only. Direct decoder taps from
stages 2/4/6 would give early stages shorter exact-gradient paths and would weaken
the interpretation of an eight-stage fixed feedback experiment. A multistage
lateral decoder may be introduced only if the exact-BP dense-task pilot fails its
performance gate; if used, it must be shared by all four methods and its direct
gradient paths must be reported explicitly.

## 3. Three downstream tasks

Only four feedback methods are compared on each task. The tasks, rather than
extra method variants, provide breadth.

| Task | Output form | P11 feature | Temporary head | Primary metric |
|---|---|---|---|---|
| Caltech-101 (all 101 object classes) | semantic classification and retrieval evaluation | 448-D global descriptor | `LN -> Linear(448,256) -> GELU -> Dropout -> Linear(256,101)`; the 256-D normalized hidden vector is also used for retrieval | Top-1 |
| ISIC 2016 | binary lesion segmentation | final 224x14x14 token grid | under-1M-parameter attention-free depthwise-separable progressive decoder to 224x224 | mean IoU |
| LSP | 14-keypoint heatmap localization | final 224x14x14 token grid | the same decoder family with a 14-channel heatmap output | PCK@0.2 torso |

Secondary metrics are balanced accuracy, retrieval mAP and Recall@1 for
Caltech-101; Dice, sensitivity and specificity for ISIC; and PCKh@0.5 plus
per-joint PCK for LSP.

Caltech-101 is run first because it cheaply validates the full fixed-feedback
engine on a new label space. ISIC then tests dense boundary prediction and LSP
tests structured localization. KADID-10k is not in the first formal batch:
the repository's prior reference-disjoint result was SRCC 0.0824, so its label,
loss and split sanity must be repaired before it can provide an interpretable FA
comparison. It can be added later as a regression stress test without changing
the four feedback methods.

## 4. The only four method groups

For each task and matched seed, first train a task head with the complete P11
backbone frozen and save a `common_start.pt`. The four methods use exactly this
same checkpoint, split, batches, augmentation stream, optimizer definition and
training budget.

| Method | Current downstream forward | Backbone update | Optical connector sent to the preceding stage |
|---|---|---|---|
| NoFT | source P11 | none; evaluate the common head-warmup endpoint | none |
| BP-current | current P11 | adapter, all eight phases, electronic residuals and head | exact derivative of the current operator |
| FA-pretrained | identical current P11 | same trainable tensors as BP-current | fixed ImageNet source phase/operator |
| FA-random | identical current P11 | same trainable tensors as BP-current | fixed seed-matched random phase/operator |

For FA-pretrained and FA-random, every optical layer still uses its current
phase in the forward pass and for its own local phase gradient. Only the error
connector propagated to the preceding optical stage is replaced. Electronic
residual mixers, adapter and task head always use ordinary BP. FA-pretrained is
therefore not a frozen-forward model and FA-random is not a random-forward
baseline.

The token-stage Qwen-to-row-major permutation, token-axis transfer function and
inverse permutation are part of the connector contract. A P09/P10 phase tensor
or an isotropic 2-D propagation operator is invalid feedback for P11.

## 5. Fair optimization protocol

### 5.1 Data and selection

- Caltech-101 excludes `BACKGROUND_Google`; each paired seed creates a
  class-stratified train/validation/test manifest before training. Retrieval
  gallery and query sets are disjoint and stored in the manifest.
- ISIC uses the official 900 training pairs with a fixed image-disjoint
  validation subset; if a patient/group identifier is available, all images
  from that group must remain in one split. The 379 official test pairs are
  evaluated only after validation selection.
- LSP uses a frozen image-level split and never lets augmentations of one image
  cross splits.
- Test data never tune learning rate, epoch count, decoder design or checkpoint.

### 5.2 Head warm-up and adaptation

1. Train only the task head until the validation metric stops improving; save
   the seed-specific common start and evaluate it as NoFT.
2. Tune the shared adaptation recipe using BP-current on one development seed
   only. No FA method receives method-specific hyperparameter tuning.
3. Start BP-current, FA-pretrained and FA-random from byte-identical model and
   optimizer-independent common states.
4. Use raw phase without weight decay, cosine decay with warm-up, AMP, identical
   augmentations and validation-selected checkpoints.

Initial adaptation candidates are phase LR `3e-3`, adapter/residual LR `2e-4`,
and head LR `1e-3`. If the BP pilot has mean phase motion below `0.05 rad` and
does not improve over NoFT, repeat the BP pilot once with phase LR `7e-3`; after
that choice, freeze the recipe for all methods. This preserves a meaningful
phase update while avoiding per-method tuning.

Suggested first budgets are 30 adaptation epochs for Caltech-101 and 50 for
ISIC/LSP, with early stopping used only through the shared validation rule. The
exact batch sizes are chosen by GPU memory smoke tests and then fixed within a
task; global batch and scheduler step count must match across the three updating
methods.

### 5.3 Paired repetition

After a one-seed screen passes, run three paired seeds (`2026`, `2027`, `2028`).
Every seed has one common head-warmup checkpoint and four matched endpoints.
Report mean, sample standard deviation, the paired per-seed method differences,
and a bootstrap confidence interval for the paired differences. If the
FA-pretrained versus FA-random result changes sign across seeds, expand that
task to five seeds before making a strong claim.

## 6. Required engineering checks before a formal run

1. Strictly load the P11 architecture signature and source backbone; loading a
   P09/P10 checkpoint must fail.
2. At the common start, all four methods must have identical normal forward
   outputs within numerical tolerance.
3. At initialization, BP-current and FA-pretrained must have approximately
   identical per-layer phase gradients because current and source phases are
   initially equal. Their gradient cosine should be near one in all eight
   stages; FA-random should be clearly separated.
4. Verify finite, nonzero gradients for every P11 phase, adapter, enabled
   residual module and task head.
5. Test token-axis feedback with the physical row-major permutation and
   channel-axis feedback without it; include an adjoint/finite-difference test.
6. Verify resume determinism, random-feedback reproducibility and DDP
   single-/multi-GPU agreement.
7. Assert that Qwen stem buffers never change and that optical gates remain at
   or above 0.5.
8. Save source/common-start/checkpoint hashes, resolved config, split manifest,
   Git commit, GPU identities and the complete epoch history.

No formal result is interpreted if BP-current cannot learn or if the
BP-current/FA-pretrained initialization gradient check fails.

## 7. Measurements beyond the task score

For every validation-selected endpoint, store:

- primary and secondary task metrics;
- per-stage circular phase RMS/mean drift from the ImageNet source;
- per-stage gradient cosine and norm ratio to the matched BP-current run at
  epoch 0, an early epoch and the selected endpoint;
- optical gate and electronic residual gate values;
- normal, optical-off, phase-random and electronic-skip-off inference;
- trainable optical/electronic/head parameter counts and measured throughput.

Destructive ablations measure dependency of a co-adapted model; they are not
standalone retrained optical/electronic accuracies. Gate values are numerical
fusion coefficients, not optical energy or hardware-compute fractions.

For a higher-is-better metric `M`, also report the BP-gain recovery:

```text
recovery(method) = (M_method - M_NoFT) / (M_BP-current - M_NoFT)
```

This ratio is reported only when BP-current improves meaningfully over NoFT;
otherwise it is unstable and the task is marked non-diagnostic for FA.

## 8. Pre-registered interpretation

Per task, the fixed-feedback hypothesis is supported when:

1. BP-current provides genuine adaptation headroom over NoFT;
2. FA-pretrained recovers at least 80% of the BP gain and remains close to BP
   in the task's primary metric;
3. FA-pretrained exceeds FA-random in paired seeds;
4. pretrained-feedback gradient cosine is consistently higher than random
   feedback and the performance ordering agrees with that geometry.

Evidence on two of the three output forms supports a cross-task trend; the same
ordering on all three supports a strong generality claim. A result on only
Caltech-101 supports semantic transfer, not a universal backbone claim.

The first paper-facing table remains exactly four columns:

```text
Task | NoFT | BP-current | FA-pretrained | FA-random
```

Architecture comparisons (P09/P10/P11), head choices, phase-LR screening and
source ablations belong in separate diagnostic tables or supplementary
material and never become extra feedback-method groups.

## 9. Execution order

1. Implement the shared P11 downstream wrapper, fixed-feedback configuration,
   source/hash manifest and global/spatial heads.
2. Add unit tests and a one-batch real-checkpoint GPU smoke for all four groups.
3. Run Caltech-101 one-seed four-group screen and inspect both accuracy and
   gradient geometry.
4. In parallel, run NoFT/BP-current ceiling screens for ISIC and LSP. Finalize
   the decoder before launching their FA groups.
5. Freeze task recipes, run three paired seeds, evaluate each selected test
   checkpoint once, and generate one machine-readable cross-task summary.

Launch, smoke, watch, resume and summarize commands will be added under the new
experiment's `commands/` directory when implementation begins. No formal
training command should exist before its config and smoke test are executable.
