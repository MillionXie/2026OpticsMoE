# Approved RESISC full-size fitting continuation, 2026-10-14

The human explicitly approved continuing to at most 120 total epochs in reply to
the concrete proposal specifying at most two additional fit GPU hours per arm.
Exactly the existing L4 rho0/.3 pair continues from completed E30 raw/EMA/Adam/
scheduler; preserve E29 best, all prefix history and first-minimum validation EMA
balanced NLL selection. No restart from best, replay, third candidate or retry.

Keep all 11198 training and 1399 validation images, fixed classes/split/cache,
original 908552 phase parameters and full physical graph, macro16/micro8, 700
Adam updates per complete epoch and tail14, MixUp.6/LS.08/capture.05/TV.005/EMA.95,
augmentation/seed/order/LR and existing early-stop logic. No test arrays/inference.
The only profile changes from completed micro8 are maximum_epochs=120 and
maximum_seconds=7200, explicitly an additional fitting interval. History seconds
retain the old cumulative offset; the deadline compares only new elapsed fitting
time. The original engine checks the deadline before beginning each complete epoch
and finishes that epoch plus required selected-endpoint evaluation; this boundary
granularity is disclosed. Maximum120 or additional time/early-stop, first reached.

Independent worker compiles the sealed original fit with exactly three reviewed
substitutions: allow only the two budget fields in resume config comparison,
compare additional rather than cumulative elapsed seconds, and record real memory/
microbatch/prior-time/new-budget in resume receipt. It reuses the already validated
scalar-equivalent fixed-mask batch path and repeats actual restored-state 20-class
and tail14 numerical gates in each production child. No model/loss/optimizer edits.
The original 141-file closure remains untouched. New 144-file closure adds this
protocol, worker and CPU auditor; current/original Git blobs verified then frozen.

Admission requires actual completed E30 pair/all processes exited, prior full CPU
audit and execution-migration audit plus correct completed-receipt SHA. Save parent
last/best/history/gradient/raw/EMA identities and paired initial SHA. Output/start
claims are exclusive and failure artifacts remain. Source only main/Git, never SCP.
Only physical GPUs2/5/6 (A1006 allowed), global max6/percard2, at most two children
in this pair, Torch2GiB plus800MiB margin, CPU parent CUDA invisible; no foreign
termination, shared environment or laboratory access.

After both fit_complete and actual parent/children exit, an exclusive saved CPU
audit verifies the new closure and immutable E1-30 parent SHA/history/gradients/
orders/transforms, first new E31, exact raw/EMA restoration, unchanged optimizer/
scheduler, all independent actual order/affine/MixUp/tail14, Adam700*actualE/
scheduler and all11 finite nonzero phase gradients, data/ID/label/argmax/CM/support/
macro/NLL. It changes only the prior science auditor's fixed30 ceiling to actual
configuration maximum120, never uses old fixed counts or synthetic science flags.
No forward or test. Existing out/claim/failure must reject repeat audit.

Validation gates stay val65/macro60/eachrecall20/nozero/last20zero<=half/gap15/
all4nonzero/inputvariation/perclass routing and actual selected-trained BMP.
Small gap alone fails. No eligible -> bounded failure, no automatic added budget.
Selected trained native and independent2/3 geometry/class viability still precede
the formal six, science finalizer/durable all-six locks and first once-test.
Accuracy75 and continuation benefit are not guaranteed. Input information checks
may read only train/cache or saved train/val predictions without model inference,
test or automatic encoding/profile changes. Preserve all old results and plots.
