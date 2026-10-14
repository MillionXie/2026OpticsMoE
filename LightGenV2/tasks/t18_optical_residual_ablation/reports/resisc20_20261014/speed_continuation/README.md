# Actual RESISC microbatch speed continuation, 2026-10-14

Human explicitly requests current batch/epoch counts and appropriate training
speed optimization. Full11198train/1399val/1403frozen test, every image, original
full908552phase model and20windows remain. Effectiveoptimizerbatch16,700updates
percompleteepoch includingtail14, seed17order/affine/MixUp.6, originalloss/EMA/LR/
scheduler and total30epochs/14400seconds unchanged. Test not used.

The slow runner executed micro1:11198separate forward/backward calls per epoch,
plus1399scalarvalidation calls; every5epochs also full11198train evaluation.
Micro8 executes1400forward/backward chunks (699x2 + tail14in2chunks), still700
Adamupdates, and175validation chunks. No image/epoch subset reduction needed.

A bounded GPU5probe with Torch2GiB first found directmicro2/4/8 failedprobability/
capture/gradient tolerances: probabilitymax1.4663e-5/rho0 or3.8899e-5/rho.3,
gradrelative4.82%/5.46%. Preserve that real failure, no tolerance relaxation.
Batch-dependent fixed-mask reduction arithmetic was isolated by retaining the
same B1energy reduction per image for opticalrouter/fixed20detectors, while
batching propagation/phase/OEO. This changes execution, not masks, physics,
networksize or learnedreadout. TF32false/float32precisionhighest actuallyrecorded;
this evidence is not a claim that TF32 caused the original batch difference.

Separatefresh+savedraw20class/macro16/real tail14 and savedEMA gates passed all
micro1/2/4/8. Chosencommonmicro8 probability/captureerror0, maxgradientrelative
8.2640e-7/mincosine>.99999999999966, all11finite/nonzero. Peakallocated<=856383488
bytes/reserved<=1042284544bytes, within2GiB. Median16sampleforward/backwardGPU5
.73125->.13623seconds rho0, .69702->.13103 rho.3 (~5.37/5.32x). These kernel
measurements omitencoding/optimizer/evaluation/I/O; full-epochresults below.

Oldparent3579436/children3579968/3580043 were identified by actualcommand/run/
CUDAUUID, coherentlast/best/history/gradients/Adam700E/scheduler checked, preserved
and intentionallystopped after eachE21. All exited, oldallocations released; no
foreign process signal or oldfile overwrite. Interruptedpartialepoch may be
discarded and is disclosed; completedepochs were not replayed. Durableexclusive
stopreceipt preserves allcheckpoint/history/sourceSHAs and priorraw/EMA states.
Conservativeprior9868.589157seconds fromoriginalstarttostop consumes the same
14400budget; no extraepoch/fitcandidate/budget extension.

Newrun resisc20_s17_mainL4_rho0rho03_fullres_microbatch_continue30_gpu256_20261014
actuallystarted with CPUparent3740234, GPU6rho.3child3740747/GPU2rho0child3740819.
Running141sourceclosure frozen75be74b96e49e221c6b35a8edd94f98c0bfedac5=
original136 plusfiveindependentprobe/execution/audit/protocolfiles. Original136
current/original513remain, no sharedlibrary/environment or modelsource edits.
Sealedfit is compiled in that process with exactlythree execution/resume
bookkeeping substitutions, reviewed in the speed protocol. Bothproductionactual
restoredstatebatchgates passed, probabilities/captureerror0/gradrel<=7.0576e-7/
all11finite/nonzero. OriginalpairedinitialSHAf7de... preserved; restoredtrained
raw/EMA stateSHAs correctly differ betweenrho. Raw/EMA/Adamsteps/moments/scheduler/
oldselectedbest and entireE1-21history/orders/affine/MixUp/gradients carried;
firstnewglobalepoch22. Resume receipts prove no resets.

ActualfirstcompleteE22rho0/rho.3 ordinaryepochs58.6610/61.7730sec; E2357.1412/
61.0753sec versus oldordinary~421-428sec (~7x). E23val35.9543/39.6712%, macro
35.9327/39.6470%, balancedNLL2.157291/2.044389. This is startup/resume/throughput
evidence, not completefit/eligibility/test or proofaccuracy75/overfitting solved.
No fullCPUaudit/plots/PDFmarker yet at this observation.

Finishat originalglobal30/time/earlystop, actualCPUparent+bothchildrenexit, then
exclusive audit_resisc20_microbatch.py checks originalprefixSHAs/identities/
rawEMA restoration/noreset/commonprofile/fullmanifestN/700E/tail14/eachactual
order-affine-MixUp/Adam/scheduler/11grad/savedIDlabelsargmaxCMsupportmacroNLL;
forward/test0. Save migrationproof before admission to correspondingfigures.
Oldstoppedrunner is not complete: do not restartoldqueue or use its oldauditor/
health to implycurrenttraining. Separatefinalfigures and actualselectednative,
independent2/3 and formal6/science/once-test still require realvalidationgates.
Originalgateval65/macro60/each.2/nozero/persistentzero/gap15/all4/inputvariation
and trainednative are not relaxed. Failureclaims/output retained, noautoretry.

AllJSONreceipts are exactserverbytes with SHA in artifact_transfer.json. Immutable
PTsnapshots stayserver-side and are notGit artifacts. Fullmodel and dataset are
unchanged; this disclosed midrunexecution migration is not claimed bit-identical
trainingtrajectory or a new independent Caltechrecovery experiment.
