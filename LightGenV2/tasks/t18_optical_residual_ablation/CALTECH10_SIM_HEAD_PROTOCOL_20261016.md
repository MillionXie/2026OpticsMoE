# Frozen optical checkpoints plus one shared 8554-parameter electronic head

Human explicitly corrects the prior fixed-window interpretation: add the SAME
8554-parameter electronic head during simulation, both residual/control same
architecture; thereafter physical adaptation only fine-tunes that existing head.
Latest instruction: few epochs first, retain current residual accuracy AND gap;
residual head may receive extra fine-tuning. All optics remain frozen. Consequently
same-checkpoint and same-contract CCD acquisitions remain valid; never discard them.
Other chat owns lab_* and the single SDK/physical acquisition queue. This source
does not edit that queue or run devices. Human explicitly authorized coordination
and additional L2/L4 acquisition alongside L3, same-arm/layer TRAIN then TEST.

## Architecture and preserved evidence

Original accepted optical parents: L2 residual E1143, L3 E123, L4 earliest E488;
control actual100 selected E98/100/100. Exact paths/SHAs/digests in frozen profile.
All old sources/PT/data/history/failures/claims and CCD data remain untouched.
Load each selected optical model in eval, requires_grad=False. Optical features
are calculated once per actual TRAIN/VAL/TEST and cached privately. Neither frozen
phase nor router parameters enter the optimizer; verify parent SHA and parameter
digest unchanged after extraction. Physical SLM input encodings/phase tensors do
not change when only the FINAL electronic head is trained. Do not reuse captures
if optical weights, LUT/ROI/exposure/encoding contracts are different.

New common inference module caltech10_sim_head_model.CCDHead: raw final CCD478
float32 -> divide by spatial mean clamped1e-6 ->log1p ->adaptive avg16x16 ->flatten256
->standardize using saved SIMULATION TRAIN mean/std(clamp.001) ->Linear256x32
->ReLU ->Linear32x10 ->add log(optical window probability clamped1e-8) ->softmax.
The readout has exactly8554 trainable parameters; mean/std are512 fixed buffers,
not extra trainable parameters. Hardware MUST import this SAME module/contract,
load the trained head PT and its SIM TRAIN buffers, and must not refit mean/std or
create a random extra head. Final layer starts zero to preserve optical baseline;
first layer uses seed17 identical initialization across all six arms. E0 is only
an initialization sanity check and is explicitly INELIGIBLE for model selection.

## Short paired fitting and extra residual budget

Both heads at each depth first receive exactly3 epochs, same seed/order/batch64,
12 updates/epoch(tail57), AdamW lr.0001/weight_decay.001, no augmentation/MixUp or
electronic dropout. Plain supervised CE; gradients only from TRAIN761. Initial
head-net digest identical. VAL157 first minimum balancedNLL selects within E1+
(never E0, max TEST accuracy or favorable epoch). Keep best/last head PT only in
each immutable saved segment. A segment is3 epochs, not a total cap. No-residual
head then freezes at its selected-within3 checkpoint. Only residual head can
continue3-epoch segments with full head Adam state; optics always remain fixed.
Extra residual budget is reported, not presented as equal-budget causality.

Targets are the latest human requirement, not rounded success: residual TEST
accuracy >= its accepted baseline AND residual-minus-trained-control TEST gap >=
the accepted prior gap (L2 about81.724/4.037pp, L3 87.452/7.201pp, L4 93.399/9.493pp;
exact values in profile). Stop an achieved depth; otherwise review after>=15 head
epochs and9 epochs without new minimum validation balancedNLL, preserving unmet
targets. Do not weaken control, alter classes, optimize optical weights or blindly
scan architectures. No guarantees of maintaining gap after learning a readout.

TEST images/features are first materialized only after initial TRAIN/VAL fitting
has fixed a head. Future selected-head evaluations reuse frozen TEST optical
features, once per new head digest; identical heads reuse saved probabilities.
TEST feedback controls continuation: explicitly development metrics, not unseen
independent generalization. Original validation-informed fixed10 classes, IDs,
train761/val157/test1833 and class imbalance remain; report macro and all recalls.

## Verification and resource contract

New CPU proof: exact8554, zero-output optical probability equivalence, independent
NumPy non-divisible adaptive pooling with fixed2e-5 tolerance, expected zero first
layer gradient at zero-output initialization, all six parent identities; no real
optical models/images/test. Each actual GPU arm has its own restored-feature head
micro1vs8 gradient proof with nonzero probe output weights then exact init restore,
zero optimizer updates during proof, optical grads absent. Frozen optical forward
uses existing exact simulation; train/test camera inputs are never substituted by
RGB features in the head. Runtime uses four allowed non-A100 GPUs0/2/4/5, one child
per card, Torch2GiB+800MiB spare, no foreign process changes. Other two arms queue.

Every producer must actually exit rc0 before its own exclusive independent CPU
NumPy audit: verify source/config/opticalPT/frozen identity, cache IDs/support/SHA,
independent CCD probe pooling, TRAIN-only buffers, same initial architecture,
orders/12E Adam/firstminimumE>=1, selected+last train/val head predictions, and
saved TEST IDs/argmax/CM/macro/NLL using independent matrix products. Audit has
0 optical constructions/forwards/test image reads. Own failure is preserved and
stops downstream, no automatic retry or relaxed tolerance. All raw features,
PT/IDs/predictions/logs remain private. Source-only Git; aggregate MD/CSV allowed.
Keep scheduled follow-ups PAUSED; direct active-chat observation only.
