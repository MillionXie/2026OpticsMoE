# L6 alpha-coordinate precision repair, not a relaxed numerical gate

Original276-source run78fa6d564: L2robust/L4robust/L4SAM gates passed and continue
unmodified. L6 clean parity failed before optimizer/TEST; original failure/claim/
output/source stay unchanged. Four TRAIN-image diagnosis found probability max
error.0017989278 and capture.0000495911, identical physical phase planes, but
first effective alpha differed by5.9604645e-8 after float32 inverse-logit remap.
Differences start after that fusion and amplify across existing8bit SLM encoding.
Diagnosis is not an original gate pass, nor a TEST accuracy measurement.

Only six learned alpha coordinate scalars become float64. Reconstruct old
float32 effective alpha exactly, invert the expanded [.6001,.95] sigmoid in
float64, retain double coordinates, cast effective alpha tofloat32 at the same
fusion operation. Clean physical fields/fusion retain original float32 inputs;
same state keys/shapes, same trainable counts, no extra learned parameters.
New CPU proof requires bitwise effective-alpha equality for all stages/depths;
new actual L6 producer must pass the unchanged3e-6 clean parity and ordinary
micro1/4 dropout/noise gradients. No postfailure tolerance relaxation.

New six-leaf source/run/output/claims/proof/audit namespaces. Explicitly rebind
reviewed276 functions in the new module namespace, no globals/files patched.
Same10-epoch robustness recipe, fresh Adam, EMA, TESTevery2 and TEST-informed PT
selection. Preserve failure SHA6b7bccd08d2350e07b2116ce6b3b21a8707a9d105475f4342095c39c99cbd0bb
at original L6 path; old L6 did zero optimizer updates and has no last.pt/gate.
GPU0 must be free; at most the three already-authorized exact original workers
may overlap on1/2/5. Resource overlap check identifies their commands/PIDs;
globalmax4, one/card,2GiB+800MiB, A100 forbidden. No old worker restart, foreign
kill, TEST gradient, category/split/inference-topology change or old claim edit.
Own new producer rc0/actualexit and own unchanged saved audit with completedSHA
remain required. Report precision repair/failure honestly, not original success.
