# L4 residual refinement toward10pp (2026-10-15)

Latest human explicitly asks further L4 optimization in parallel with L2. Resume
the independently audited old185 LAST E575, preserving raw/EMA, Adam27600 moments,
scheduler global epoch575 and full575 history. Old selected E488 remains fallback.
Frozen no-residual actual100 is not trained or evaluated again; L3 unchanged.

From E576 only: MixUp.3->.4, EMA decay.95->.98; LS.08/TV.05/capture.2 unchanged.
One LR restart.0001 and scheduler best/bad/cooldown statistics reset at576, retain
global epoch575 and all Adam moments. Later25-epoch segments do not repeat restart.
This is a generalization hypothesis, not a promise. Earlier .4 after150 did not
improve validation; this later restored-state+EMA recipe is separately recorded.

Same fixed10/classes/cache/roles/IDs/geometry/RGB100/4experts/E-G-E-G/908552phase,
11 tensors,8bit optical router, no electronic classifier. GPU4 one child, shared
global allowed0/2/4/5 max4,2GiB+800MiB spare, no A100/foreign process changes.
L2 separate189 continues concurrently without edits.

CPU0model/forward/test restores identities, new GPU4 restored numeric gate and
each actual producer own gate. Independent per-segment science checks fullprefix,
Adam48E, dynamic11tensor shapes and gradients, seed/order/affineMix .2<=100,
.3<=575,.4 afterward, independent LR restart replay, savedID-label-CM-NLL,
actual producer exit rc0 and science audit/completed SHA before any test.

Always wholeprefix firstminimum validationEMA balancedNLL selection, not highest
accuracy/test epoch. Each new selected parameter digest at most one test; inherit
all12 prior L4 digests and reuse unchanged exact results. Test feedback is used
for stopping: explicitly test-informed development, not unseen generalization or
same-budget causal advantage. Show train/val/test/macro, actual and selected epochs,
additional budget and test counts, all classes and negative results.

Target actual overall test residual-control>=.10 without rounding. Control1538/1833;
old residual1712/1833. Need residual>=1722/1833, at least10 net more correct. No
test class filtering, changed split, weakened control, new depths or electronics.
No total epoch cap.25 is operational saved/audit segment,1800fit seconds per segment
faultguard. Pause if validationNLL fails .0002 significant improvement75 new epochs
and LR reaches2e-5. Failures preserve evidence/stop downstream/no automatic retry.
No reopening completed old workers or editing their source/claims/PT/data.
