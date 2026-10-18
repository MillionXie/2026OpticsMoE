# Retained optical phases with gradual existing-electronics adaptation

Human authorization: continue rapid fine-tuning of residual L2/L4/L6, prioritize L2,
use up to five shared non-A100 GPUs, freeze accepted controls and protect the
separately delegated hardware acquisition. No timer, fresh optical/head training,
class/split changes, new inference module or interference with lab collection.

Evidence: canonical learned physical alpha remains approximately .94-.95 despite
the allowed .4001 lower bound. A direct .45 reset collapsed predictions. Previous
last-stage-only adaptation, joint phase-code and weaker prototype trials did not
improve canonical TEST correct counts. Hypothesis: adapting **all existing**
electronic stages and the same 5140-parameter head while gradually capping alpha
can preserve useful features better than a hard reset. This is an unproven
training hypothesis; no promised gain and no attribution to one factor alone.

Four fixed warm candidates: L2 cap .95→.90, L2 .95→.80, L4/L6 .95→.90 over eight
local epochs. At epoch four, stop if TRAIN<90% AND VALID falls more than three
points below that candidate's parent. Eight epochs is a bounded strategy review,
not a claim of convergence. Original canonical weights remain on regressions or
TEST correct-count ties. L2 loads canonical parameter-average e7c140ecd11d52a91bef882a01bc947c3da7f9f72df63d7fcf23055ab503be69;
L4 loads a224cf9742e97351205bda983945d689fa4f6040c56f84f0c51a2c590d03e0e4;
L6 loads 881aeb07280875def1e558897a30480e7b5bc7cc25ccb4d03bf743fc006a956a.
All source441 parent identities and saved independent scientific receipts must
pass; L6 sigmoid legacy state is exactly remapped to physical F64 alpha with
unchanged initial forward. L2 has no single historical epoch; every new epoch is
explicitly local. All learned tensors warm-start; new Adam is intentional.

Only optical raw_router and phase_sets are frozen, byte exact, gradient None.
Train existing d convolution tensors, d alpha scalars and linear W/b: 5160/5180/
5200 parameters, 6/10/14 tensors. No optical phase updates. All RMS derivatives
are enabled in training; clean evaluation values and inference graph are
unchanged. Alpha remains learnable, projected after each Adam update and after
each EMA update to [.4001,current linear cap]. Conv/head/alpha LR .001/.002/.0003
cosine toward .00002; AdamW decay .01/.02/0, norm clip1, EMA .95.

Fixed rich20 TRAIN3141/VALID383/TEST411, original IDs/cache/source roles unchanged.
Two TRAIN views, crop area≥.95, existing affine/color/flip, MixUp .1, LS .02,
feature dropout .05; .5 weighted clean CE+.5 augmented CE+.25 self KD(T2)+.2 JS.
Frozen self targets are canonical saved TRAIN probabilities, never TEST labels.
No CCD or parameter noise in this strategy. Macro16/tail5, micro2,197 updates/E.
Save source/config/command/environment/policy/parent identity, scalar method
trace and training predictions, exact last Adam/EMA boundary; keep best/last PT
only. Best is first minimum VALID balanced NLL (parent included). After fit and
its own independent saved science, evaluate at most best and last EMA once per
new digest; parent digest reuses saved TEST. Internal TEST choice breaks ties by
VALID NLL then earliest local epoch, but global canonical only changes on an
actual additional TEST correct prediction. TEST-informed development, not unseen
generalization or equal-budget optical-residual causality. Controls remain fixed
at 50 and are never forwarded again. Aggregate summaries must include TRAIN/VAL/
TEST/macro, local/parent epoch semantics, actual counts and extra training.

Independent CPU proof: exact parent bytes/trainability, projection/caps/analytic
CE/KL/JS, zero real model forward or images. Actual GPU gate: initial canonical
probability/capture≤3e-6; all20/macro16/tail5 micro1vs2 probabilities/captures≤3e-6,
all trainable gradients finite/nonzero, rel≤.02/cos≥.999, no optimizer or TEST.
Saved science independently checks all optical bytes, ID-label-feature-head-CM-
NLL, first minimum, local orders/affine/crop/MixUp/self targets/losses/cap bounds,
last Adam/EMA update and producer actual rc0/exit. Each new fit/test audit uses
its own exclusive claim, log, result and completed SHA. Any failure is retained;
no automatic retry, source mutation or threshold relaxation.

Four actual jobs at most, one/card 3/4/0/1 with five-GPU global guard and shared
2GiB+800MiB memory policy. No A100, foreign kill or environment change. Frozen
optical phases DO NOT make updated upstream electronics compatible with earlier
CCD: changed electronic/alpha output changes later optical inputs. New candidates
must never silently replace the laboratory's frozen e7c/5c8 release. Lab owner
handles devices/questions with human; this task handles simulation only. All
old states, receipts, failures and hardware data remain intact. Source only Git;
raw JSON/PT/IDs/logs remain private.
