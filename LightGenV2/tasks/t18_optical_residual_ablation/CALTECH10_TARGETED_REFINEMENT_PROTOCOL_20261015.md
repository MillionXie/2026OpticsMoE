# Evidence-led L2 distillation and L4 calibrated SAM (2026-10-15)

Latest human explicitly requests live expert diagnosis and strategy changes,
not repeating long plateau runs. New203 = diagnosed199 + four new leaf sources.
Training-only diagnostic SHA is frozen in profile; no test arrays used for that
diagnosis. Two train macro groups measured capture-gradient norms only5-12% of
classification, with mixed/negative cosines: do not claim auxiliary dominance.
Original SAM.05 changes0.17-0.32%8bit phase codes; radius.5 changes1.6-2.9%.
Choose.5 by this train-physics diagnostic, not test score. It is a hypothesis,
not proof of improved accuracy. Old SAM/dropout negative results preserved.

Only two immediate concrete strategies: L2_distill on idle GPU2, L4_qsam on
idle GPU4. Avoid busyGPU0, shared global<=4non-A100 and one child/card,
2GiB+800MiB spare, no foreign process changes. Other idle cards need not be
filled by blind duplicate trials. Existing fixed10/cache/IDs/split/RGB100/8bit/
4experts/optical router/readout/inference graph unchanged, no electronic head.
L2 fullLAST950 Adam45600 and L4 fullLAST725 Adam34800 restored exactly; raw/EMA/
moments preserved, no epoch replay. Once LR/statistics restart at951 .0003 or
726 .0001, keep global epoch; prior historical restarts independently replayed.

L2 underfit: train78 and bonsai35.96%trainrecall. Train with frozen accepted
L4E488 optical teacher only, same augmented/MixUp view, teacher eval/no-grad.
Add .5*T^2 KL(teacher_T||student_T), T2, only when teacher agrees with dominant
training label and maxprob>=.5. CE/capture/TV retained; no SAM/dropout in L2
new path. No teacher at student validation/test/deployment. Teacher checkpoint
SHA/modeldigest frozen, teacher parameters never optimized. This is distillation
development, cannot be claimed to isolate residual alone; inference remains L2.
Reference https://arxiv.org/abs/1503.02531 . No teacher test forward or new labels.

L4 overfit: calibrated SAM radius.5, original two complete-macro passes, exact
restore then ONE Adam/EMA, retain MixUp.4/LS.08/TV.05/capture.2/EMA.98. Existing
inherited bestE488 kept. No dropout, no quantization removal/change in inference.
Reference https://arxiv.org/abs/2010.01412 . No test used to set radius.

New CPU identities/compile/analytic SAM proof, each new GPU restored ordinary+
method micro1vs8 all10/macro16/tail9, teacher frozen/numerical parity, then each
actualproducer own gate. Unique claims once, no retry/threshold relaxation.
Every10 new epochs complete saved stage, own independent source/config/PT/
prefix/rawEMA/Adam48E/6or11grad/orderaffineMix/three historical LR restart replay/
per-macro SAM or teacher evidence/savedID-label-CM-macroNLL/firstminimum science
must pass actualauditrc0 and SHA/completed after producer actualrc0/exit.
Only then selected digest test once; inherit all known old197 and earlier digests,
unchanged model exact reuse, shared digest claims. Frozen100 controls/L3 never
trained or evaluated again. Do not choose highesttest epoch or weaken control.

Review every10 new epochs, not wait75. If validationNLL fails .001 significant
improvement AND validation accuracy makes no new high for20 new epochs, stop
L4 for expert review regardless of LR; L2 additionally requires no .5pp training
accuracy gain for10 epochs, distinguishing continued fit progress from stagnation.
Record last-endpoint train/val separately from selected-model metrics. No preset total epoch cap; improve and continue, or
actual L2test>=82 / L4overallgap>=10pp stops.1800fitsec stage faultguard. Actual
review pause is not target success. New human permits evidence-led subsequent
training-strategy changes within these fixed optical/data/control constraints;
before each change record diagnosis/new config/source/gates, preserve all old
claims, no blind sweeping or repeated same recipe. Never silently edit running
source, alter tests/classes or deploy inference modifications.

Summaries always actual/newepochs/selectedE/train/val/test/macro/gap and all
classes+new/inherited test counts. Test feedback is development, extra residual
budget+distillation, not independent generalization or equal-budget residual
causal evidence. Heavy test imbalance and unresolved near-dup limitations remain.
Data/PT/IDs/rawJSON/log private; sources only Git, aggregates MDCSV allowed.
Existing GitHub destination authorized, exactT18 paths serial transactions,
foreignT06 changes preserved. t18/09 ACTIVE, quiet normal, meaningful final/fail once.
