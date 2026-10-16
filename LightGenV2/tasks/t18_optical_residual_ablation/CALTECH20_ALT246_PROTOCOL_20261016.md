# Caltech20: one router and alternating main-depth2/4/6 with linear electronics

Latest direct human authorization changes the previous2/3/4 Caltech10 experiment.
Human confirmed original Caltech20, not Kather training. Old Kather forward was
read from the actual server: prompt.routing called once before expert stages;
between_stage invokes global and the final stage invokes the final global.
Thus2/4/6 mean one/two/three E-G cycles, plus one separate router. Old nine-expert
Kather and current four-expert geometry differ and are not single-factor controls.

Write ownership: only six new caltech20_alt246/model/profile/check/audit/protocol
leaves and new own reports. No old source, run, CCD, PT, failure or lab_* changes.
Original20 classes and cache/split are fixed before training:1372/283/2064 images.
Validation-informed class screening and old TEST observation remain disclosed.
No class rescreening to attain a gap or claim of patient/subject independence.

Graph: current four224 experts,478 global,8bit amplitude AND phase,RGB100,
router once; E-G/E-G-E-G/E-G-E-G-E-G. Router never has electronic fusion or rho.
Each E/G optical CCD passes the existing nonlinearity; a tiny1channel3x3 electronic
residual (9weights+1bias), from the SAME stage input, matches the next amplitude
grid. ABO standalone.model.fuse is the mathematical reference: detach both branch
per-sample RMS, normalize branches, learned weighted sum, renormalize and restore
electronic reference RMS. Each stage alpha=.7001+(.95-.7001)*sigmoid(raw), initial
.8, learned in both arms. This is a normalized FEATURE coefficient, not a fixed
probability mixture, power fraction or a claim of globally>.7 optical contribution
through a product of layers. The fixed.8/.2 probability draft NEVER executed.

Terminal fused amplitude -> fixed pool16x16 -> nonaffine LayerNorm256 -> trained
Linear256to20 (5140parameters), no MLP/CNN/RGB classification bypass. Electronics
are5140+11*depth (5162/5184/5206), including10local weights+bias and1alpha perstage.
Both arms use identical architecture/initialization/electronic training. The
paired ablation changes only coherent optical rho0/.3. All future physical
adaptation must use this same residual/alpha/linear graph and selected trained
state; it is a NEW optical identity, not compatible with old Caltech10 CCD.

Fresh seed17, each arm full100, no early stop; effective16/micro4,tail12,
86Adam updates/E=8600. AdamW phase/head lr.002, decay0/.0001, clip1;
cosine100 to.0002, EMA.95, common MixUp.2/LS.02/capture.05/phaseTV.005.
TRAIN gradients only, validation EMA first minimum balancedNLL selects; full
LAST100 raw/EMA/Adam/scheduler saved, not just best. Per-epoch order/affineMix
hashes, loss/phase-gradient/linear-gradient counts, clean train/val and routing
diagnostics saved. Every phase and linear tensor finite/nonzero at numeric gate.
4 non-A1000/2/4/5 at most one child/card, Torch2GiB+800MiB spare; no foreignkill.
10800 fitsec/arm is a failure guard, not silent incomplete100 acceptance. Failure
preserves evidence and stops downstream; no automatic retry or relaxed gates.

CPU structural proof: parameter counts, real forward router/stage call trace,
black20 normalization, same paired init, rho/alpha bounds, independent NumPy
linear/softmax plus exact ABO fusion function/gradient parity and alpha bounds. Each actual GPU producer independently checks all20 labels,
macro16/tail12 micro1vs4 probabilities/capture and all phase+linear gradients;
zero optimizer updates before fit. Own independent CPU saved-math audit checks
sources/config/rawEMA/firstmin/8600Adam/full100/order+affineMix hashes/cosine/
IDs/support/CM/NLL/linear/softmax/alpha bounds/pooling probes and paired identities. It creates
only its own durable claim/output; old audits are not substituted.

Only after all six baseline arms and their own science receipts truly pass and
producer processes exit does each frozen selected model run TEST once. Controls
then remain100 forever, no retraining or repeated evaluation. TEST arithmetic
independently checked from saved probabilities. Record train/val/test/macro,
actual/selected epochs and real gap even if negative. Accuracy targets: L4rho.3
TEST>.8, L6rho.3 TEST>L4rho.3, preferably residual-control gap>10pp each.
Targets cannot be guaranteed or enforced by weakening a control, picking a
favorable TEST epoch or making a dataset artificially easy/hard after results.

After verified baseline, inspect actual curves and error classes before writing
an evidence-backed residual-only continuation profile. Human authorizes extra
residual tuning; controls freeze. Additional optimization budget and every new
selected weight's test evaluation are disclosed as TEST-informed development,
not independent generalization or equal-budget causal residual advantage.
No automation, SDK, native acquisition, A100, extra depths or new categories.
