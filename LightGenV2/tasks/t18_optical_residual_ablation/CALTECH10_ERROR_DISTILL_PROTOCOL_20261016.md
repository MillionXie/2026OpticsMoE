# Directly followed teacher-correct/student-error distillation (2026-10-16)

Human authorizes active evidence-led expert adjustment without scheduled tasks;
t18 and09 both PAUSED and must stay so. New221=217+these4leaves, own paths only.
Prior hard-target run actually1140/E1117 train80.1577/val75.7962/test81.5603/
macro77.3899/gap3.8734,3new selectedweights/40known tests; no target success.
Old teacher gradient is aligned with CE(cos.6602/.9143) but only17-23%CE norm,
teacher correct/confident on90+%TRAIN and student still~80%TRAIN. This supports
a specific error-correction hypothesis, not claiming KD conflict or guaranteed
accuracy. NO targeting test IDs/classes, no class changes or weaker control.

Restore LAST1140 fullraw/EMA/Adam54720/scheduler1140, no replay/reset moments/EMA.
Retain hardlabels/MixUp0/LS0/focal1.5/TV.01/capture.2/EMA.95/teacherT2/conf.5.
On each actualTRAIN microbatch: original teacher-correct&confident gate remains;
multiply its KD weight4 ONLY if current student argmax != real TRAIN label,
otherwise retain multiplier1 (basealpha.5, effectivealpha2 for errors). Gate
and wrong-mask detached, no optimization of teacher. This raises small aligned
KD gradients for mistakes rather than global regularization. Once1141LR.0003/
scheduler stats restart/globalepoch retained. No inference modification/new
trainable tensor: original2layer479364phases/6tensors, no electronics. Accepted
L3/L4 and frozen100 controls untouched. TeacherE488 training-only/frozen.

CPU analytical weightedKL tests both correct and wrong masks, hardtarget/focal
finite difference and parent identities. NewGPU2 ordinary+method all10/macro16/
tail9 plus first10bonsaiTRAIN group, micro1vs8, all6finite/nonzero, state/teacher
unchanged/optimizer0/test0; error-focus branch must be exercised. Each actual
producer owngate separately. PerTRAINmicro teacher+student probs and labels
private evidence: independentCPU validates sum1/trueorderlabels, reconstructs
gate/wrong/counts/weightedT2KL, no modelforward. New independent auditor also
checks source/config/prefix/rawEMA/shapeinit/Adam48E/order/affine, hardMix identity
from1091, all LR326/726/951/1041/1091/1141 restarts, savedIDlabelCMmacroNLL/
wholeprefix firstminimum. Own produceractualrc0/exit then own science SHA/
completed/auditrc0 before each newselected once-test; inherit40known digests,
sameweights reuse. No max-test epoch selection or threshold relaxation.

10epoch saved stages,1800fitsec faultguard/no totalepochcap. Existing20epoch
NLL(.001)/valaccuracy plus10epoch train(.5pp) stagnation review; target actual
test>=82, no rounding. Directly follow in current conversation, no scheduled
wakeup/task. Failure preserved/downstreamstop/no retry; oldsource/PT/data/claims
failed205 preserved. Shared main includes5otherwindow unpublished benchsource
commits; this source and any new training MUST wait for their own publication
or explicit human permission to include them. Do not bypass AGENTS with branches,
worktrees,private source transfer or rewriting their commits. Newtrial scope
does not authorize lab execution/foreign edits/A100/newdepth/changedphysics.
SourceonlyGit, actual authorizedGitHub/server publication first, exact paths
serial, no force or rawJSON/PT/IDs/log upload. All reported results include
actual/new/selectedE/train/val/test/macro/gap/testcount and test-informed/extra
budget/teacher limits, not unseen independent gen or equalbudget residual cause.
