# Stable equivalent softmax gauge, source368; preserve source364 first failure

Source364 CPU fit stopped at float32-versus-independent-float64 probability
error>3e-6 before ANY candidate PT or TEST. Keep original failure/rawrc1 and
all outputs/source. It is a real numeric-admission failure, not accepted
scientific result. Do not loosen 3e-6, drop offending beta, or re-enter364.

Four new leaves/source368 remove common class offsets BEFORE float32 cast:
Wc=W-mean_class(W),bc=b-mean_class(b). For every x, the same scalar is removed
from every logit, so exact softmax/classification is unchanged. It does not
change classifier capacity/5140 count or require float64 inference. Candidate
weights remain float32 and same deployed linear module. Original beta grid,
TRAIN fit/covariance/scale, VAL selection and independent all-grid checks kept.

New CPU proof independently retains original failure SHA/no candidate/TEST,
original Gaussian math, actual parentL2 TRAIN and VAL for all6 beta: report raw
and centered float32 errors, prove exact-real gauge invariance<1e-10 and keep
ORIGINAL<=3e-6 for every centered probability calculation. Partial CPU checks
cannot admit production without real_role_centering_proof_passed and no new
CPU failure. If this fails, preserve and do not train/test/retry this method.
24 cached TRAIN/VAL functional-head forwards in this additional CPU proof,
no optical forward, no images or TEST. Additional saved science independently
reconstructs centered W/b from exact same covariance/mean and verifies every
grid score and selection, state frozen except same5140head, old-control SHA.

Only after a fresh unique source368 fit/rawexit0 and own fitSCI/auditrc0/SHA
does each head get one cached TEST result+independent saved test SCI. All
private PT/features/IDs/JSON stay private, source Git only. This diagnosed
numerical equivalence repair is not relabeling old364 as passed or changing
science thresholds. Continuous foreground optimization and max5 GPU0–4 scope
remain; GPU semantic workers unaffected, no scheduler or new inference graph.
