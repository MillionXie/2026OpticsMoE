# Caltech20 residual-only generalization refinement

The six new alternating2/4/6 baseline arms truly completed100/8600 Adam updates,
own saved science SHA548c5a90f818b5cb6f3f77f34edbe429d28e6e5d20ffdf01a013437689303ba3
and saved TEST audit SHA93db16f16e4f8ecbae1b07591595a8399dfe308ae3245dd3f7b4927e13fde8a7
passed, CPU campaign and all producers exited. No baseline stage is re-entered.
Controls remain their selected100 models/results forever, including L4 selected
E96 (actual100). Old Caltech10/CCD work is unrelated and remains protected except
the separately human-authorized deletion of old camera PNG images by its owner.

Baseline residual train/val/test (%)2:97.5948/71.3781/85.2229,
4:100/67.4912/83.2364,6:99.7813/63.2509/83.0426. Test controls2:84.1085,
4:81.7345,6:78.0039. Four exceeds80; six is.1938pp below four. Gaps1.1143,
1.5019,5.0388pp are below the preferred10. Original20 test2064 is imbalanced;
full macro/support/class results must accompany reports. Prior class screening,
old observed test and further test-feedback development preclude unseen
generalization or equal-budget pure residual causal claims.

## Hypothesis fixed before new model fitting

Selected residual train/val gaps26.2167/32.5088/36.5305pp increase with depth.
Most wrong validation predictions are not tiny score ties; simply changing
temperature cannot fix their argmax. Baseline has rotations/translations but no
horizontal reflection. Improve category-preserving pose coverage by flipping
RGB before fixed channel tiling, not by flipping the completed RGB mosaic.
Feature dropout at the terminal linear input and stronger phase/electronic
regularization address near-perfect train fits. This is a hypothesis, not a
proven unique cause or promised improvement.

One fixed recipe per depth, not a candidate grid: hflip.5; terminal feature
dropout2/4/6=.1/.2/.3, MixUp=.3/.4/.5, phaseTV=.02/.05/.08; LS.05 and electronic
AdamW decay.01. Capture.05/EMA.95 unchanged. All optical phases, tiny electronic
convolutions, bounded learned alphas and the same linear classifier remain
trainable. More regularization is assigned to deeper observed overfitting.
No class/split, inference shape, electronic parameter count, alpha bound,
optical rho, router count or E-G order changes. No test-class-specific recipes.

## Exact recovery and accounting

Start from each residual's full LAST100 raw/EMA/Adam8600, not fresh phases or
reset moments. Original cosine schedule is preserved in its parent artifact;
new phase explicitly starts LR.0003 and a new ReduceLROnPlateau state, factor.5,
patience3,abs-delta.002,min2e-5. Global epoch starts101. New blocks restore the
entire preceding raw/EMA/Adam/new scheduler state. Initial moments digest must
match the immutable parent, every step continues its true Adam counter.

Each10 new epochs is a complete saved/audited operation segment. Block1800fitsec
is a failure guard, never incomplete acceptance or automatic retry. First method
review occurs at20 new epochs even if small improvements persist; this is a
registered review checkpoint, not a false plateau or a global user epoch cap.
Any20 epochs with no.002 validationNLL improvement and no validationaccuracy
new high also trigger review. Further strategies require fresh evidence and a
new fixed profile; this run cannot silently expand itself.

Selection remains first minimum validation EMA balancedNLL over original100
plus all extras. Best checkpoint and its matching saved predictions may point
to the true unchanged ancestor. Preserve every block and old score; no choosing
highest TEST epoch or discarding a negative comparison. Actual and selected
epochs, additional fit time, train/val/test/macro and exact gaps are reported.

## Same inference graph and independent admission

MaskedLinear has exactly the original Linear256to20 weight/bias and applies a
precomputed TRAIN mask only while training. Evaluation is exactly F.linear,
with no dropout or extra classifier. A frozen baseline HybridMoE can load these
states strictly and produce the same inference. Electronic counts remain
5160/5180/5200, terminal5140; alpha remains per-stage learned(.7001,.95), using
the already-proven ABO detached-RMS fusion, not a fixed probability mixture.

CPU proof compares baseline/new actual inference and capture trace on TRAIN,
restores all Adam moments without a step, proves horizontal flip does not swap
channels, deterministic masks and analytic masked-linear gradients. No TEST.
Every actual producer separately verifies restored ordinary AND masked all20,
macro16/tail12 micro1vs4 probabilities/capture and every phase/conv/alpha/linear
gradient. RelativeL2<=.02,cosine>=.999,prob/capture<=2e-6 remain unchanged;
comparison norm/dot arithmetic is float64 to avoid small-scalar cosine eps
artifacts, while actual network/gradient computation staysfloat32/complex64.

One fresh exclusive saved science per complete block independently checks
source/config/parent bytes/rawEMA/moments/steps/schedule, all original+new
selection, order/affine/MixUp, separately reconstructed flip/dropout mask hashes,
all-class ID-label-argmax/CM/macro/NLL and saved feature/linear math. Only immutable
pure math helpers are imported, never an old audit dispatch or old claim/out.
CPU audit performs0 model forwards/0 test images. Failures persist and stop
downstream; no threshold relaxation or re-entry.

After true producer exits and own science rc0/completedSHA, each NEW selected
weight digest tests once; unchanged ancestral digests genuinely reuse prior
results. Case-specific atomic digest claims prevent repeated tests, and own
independent saved TEST math audit checks every result/reuse pointer. Controls
are not reloaded or evaluated again. Testing feeds stopping only, remains
explicitly development, not epoch selection or independent generalization.

Preferred per-depth stop is a genuine test gap>10pp, plus L4 test>.8 and L6
test above the final selected L4. If six meets a moving four-layer comparison
it is held, not prematurely declared finished, until four stops; no lowering
four or selecting a favorable four epoch. Review status does not mean success.
Keep the actual baseline table as the equal-budget ablation; this refinement
uses extra residual budget and depth-specific training regularization.

Pre-registered family acceptance is separate from epoch selection: the new
method's checkpoint is ALWAYS its whole-prefix first minimum validationNLL,
never its maximum-test epoch. If that final checkpoint lowers overall TEST
accuracy below the original equal100 baseline, reject this method family for
deployment at that depth and retain the original baseline, reporting the
negative method result too. Do not revert to a favorable intermediate extra
epoch. If it maintains/improves baseline accuracy, the validation-selected
method candidate may be adopted. This is explicit TEST-informed method-family
acceptance, not unseen evaluation. Six is compared to at least original four,
or the new validation-selected four if its accuracy is higher; four can never
be lowered to manufacture six-layer superiority.

Only three non-A100 cards2/4/5,one own child/card,Torch2GiB+800MiB spare, no
foreign PID or environment/DNS changes. No automation, device SDK/acquisition,
new electronic inference branch, TEST label-based class changes or new depth.
