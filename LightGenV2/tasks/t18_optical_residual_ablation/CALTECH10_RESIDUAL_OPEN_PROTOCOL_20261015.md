# Human correction: both100, control frozen, residual additional optimization

Latest instruction: both models train100, then only residual fine-tunes without a
fixed total epoch limit. Test accuracy gap targets: depth2 +4pp, depth3 +6pp,
depth4 +10pp. Targets are objectives, not promised outcomes. Earlier150 controls
were an assistant budget deviation and remain historical, not this comparison.

Prerequisite is the new exact100 campaign, all six actual exits, each own science
and frozen tests completed. Control selected within100 by firstminimum validation
EMA balanced NLL is permanently frozen. Never use an old150 control. Resume each
residual from its complete E100 LAST raw/EMA/Adam4800/scheduler100, preserving the
full prefix and original selected checkpoint. No reset or replay.

Continue residual only with MixUp .3, other optical architecture, labels, IDs,
split, windows, optimizer and scheduler unchanged. This moderate regularization
is a hypothesis; prior .4 after150 did not improve validation. It is not guaranteed
to increase accuracy or close the generalization gap. All dynamic phase tensors
must pass the actual restored-state producer gate before updates.

There is no global epoch cap. Save in25-epoch blocks, independently audit each
completed block, and carry its complete state into the next block. These blocks
do not stop continuation. Stop a depth if its test-gap objective is achieved,
or pause it if no validation NLL improvement of .0002 for75 new epochs while LR
has reached2e-5, or if the user stops it. Pausing is reported as target unmet,
not successful completion. A block has1800 fit seconds as an operational fault
guard; failure preserves evidence and stops that depth, with no automatic retry.

Each new selected model digest is tested at most once; unchanged weights reuse
the exact recorded result. Each block selects the earliest minimum validation
EMA balanced NLL from the whole preserved prefix. Test feedback controls stopping,
so final results are explicitly test-informed development. Never claim untouched
test generalization; report evaluation count and macro/each class alongside overall
accuracy, given test supports315/80/678/680/20/16/13/9/14/8. Do not change classes,
weaken controls, or choose a favorable epoch by test accuracy. Original split/ID
and exact pixels have been checked for cross-role duplicates; subject independence
and near-duplicate absence remain unproven. No additional data or test split.

Science independently verifies source, prefix, restored optimizer/EMA/scheduler,
dynamic parameter count, all phase gradients, order/affine/MixUp(.2 through100,
.3 thereafter),48 Adam updates/epoch, saved predictions/ID/label/CM/NLL and
firstminimum selection. Each block has fresh exclusive claims, exit records,
audit output and completed SHA. CPU audit has zero model forwards/test images.
Any failed producer/audit/test stops downstream without retry or threshold changes.

Only four non-A100 GPU0/2/4/5, one T18 child per card,2GiB plus800MiB spare;
do not kill foreign jobs or modify environment. No electronic head, new depth,
native/lab/formal runs. Sources Git only, private samples/PT/predictions/logs never
published. All previous sources, runs, failures, claims, and figures are preserved.
