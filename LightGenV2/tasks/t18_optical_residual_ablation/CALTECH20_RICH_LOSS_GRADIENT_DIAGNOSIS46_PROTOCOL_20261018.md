# Actual TRAIN parameter-gradient decomposition, L4/L6 only

Latest continuing expert-analysis authorization, L2 FINAL, no timers or lab work.
Source474=protected472+two leaves. Zero optimizer/model changes, not a new model
performance result. Current prior logit-only KD calculation improved alignment
but ordinary/stronger KD five-epoch results did not improve TEST. Do not call its
temperature discrepancy an established cause of overfitting. Several five-epoch
variants land on the same overall correct counts; inspect actual optical,
electronic and alpha parameter gradient contributions before more loss changes.

Reuse exact frozen459 current method/parents/data, two fixed first TRAIN macros16
from local epoch1, at each depth4/6. Repeated identical clean/augmented inputs,
noise context/MixUp teacher/gates, same model state, only differentiate total,
prototype coefficient0, KD coefficient0, and CE-only without other regularizers.
Use separate function namespaces; never monkeypatch old functions/modules or
alter running workers. The four forwards must agree≤3e-6 and model state SHA
must remain identical each time. Reconstruct prototype/KD by gradient differences
and other regularizers by subtraction; store raw gradient vectors privately.

Report norm relative to CE and cosine per phase/electronics/alpha group. A large
scalar loss is NOT evidence of dominating gradients. Two macros are a local
diagnostic, not a whole-dataset/unique-cause proof. Independent Float64 CPU toy
gradient/additivity proof first,0 real models/images/forward/optimizer. Actual
diagnosis uses32 unique TRAIN images (same32 across depths),256 micro-forward
calls/512 image-forward units total,2 models,0optimizer/VAL/TEST images. All PTs,
data, claims/failures and weights remain unchanged. Save identity/policy/parents/
before-after SHA/gradient arrays/method/loss coefficients and one diagnosis JSON.

Single free non-A100 GPU3 preferred, own PID+UUID exact, one/card and global≤5;
allow other authorized T18 jobs on other cards, do not repeat the old self-PID
resource-guard error. Torch2GiB+800MiB, no foreignkill/environment/DNS. Own durable
CPU/diagnosis claim/out/failure, no automatic retry/threshold relaxation. Source
only Git exact2paths, all472bytes immutable before/afterff; arrays/JSON/PT private.
