# Original Caltech20: accepted alternating2/4/6 refinement snapshot

本轮为原 Caltech20：Router 只运行一次，Expert→Global 重复1／2／3次。
α 按 ABO 的逐样本 RMS 归一化特征融合方式学习，范围(.7001,.95)，
不是固定0.8／0.2概率混合。两组电子结构完全相同，末端线性头5140参数，
二／四／六层电子总参数5160／5180／5200，仿真阶段已训练。

双方均先完整100轮，之后无光残差100轮冻结，只追加有光残差模型。
下表每行训练、验证、测试均来自同一选中权重；实际停止轮次与选中轮次
分别列出，选中较早轮次不等于提前停止。头部单独20步不冒称光学epoch。
四层测试超过80：True；六层测试高于当前四层：False。
二／四／六层与各自冻结对照的测试差距分别1.1143／1.7442／5.2810个百分点，
不能把尚未达到的10个百分点目标写成成功。完整逐类及所有负结果另存CSV。

All six initial arms actually completed100 epochs/8600Adam updates. All three
residual arms subsequently completed20 epochs of generalization regularization.
Six alone completed20 actual L-BFGS updates of its SAME5140-parameter head, then
40 optical epochs of TRAIN-only shallow-teacher distillation, with full Adam/
scheduler restoration and no LR/stat reset between the two20-epoch intervals.
Head updates are
not epochs. No-residual100 models and their original tests remain frozen.
All campaigns/fit/test children have actually exited. GPU project processes0,
A1000. Own science and saved-TEST audits passed; every actual fit/test child
originalrc0. CPU original exit codes were not captured, not invented.

|配置|实际光学训练／选中轮次|末端单独更新／选中步|训练 %|验证 %|测试 %|测试宏平均 %|测试差距（百分点）|
|---|---:|---:|---:|---:|---:|---:|---:|
|2层有光残差|120 / 100|0 / 0|97.5948|71.3781|85.2229|67.1605|+1.1143|
|2层无光残差|100 / 100|0 / 0|94.4606|69.6113|84.1085|66.5121|+1.1143|
|4层有光残差|120 / 112|0 / 0|99.8542|67.8445|83.4787|64.4978|+1.7442|
|4层无光残差|100 / 96|0 / 0|97.6676|66.4311|81.7345|62.6133|+1.7442|
|6层有光残差|160 / 140|20 / 0|99.7085|65.0177|83.2849|62.0726|+5.2810|
|6层无光残差|100 / 100|0 / 0|98.9067|60.0707|78.0039|58.3628|+5.2810|

Four residual test>80: **True** (83.47868217%).
Six residual test>accepted four: **False**
(83.28488372% vs 83.47868217%, -0.19379845points).
Preferred >10point residual/control gaps2/4/6:
1.11434109/1.74418605/5.28100775; achieved flags
False/False/False. Do not round an unmet goal into success,
lower four/control accuracy, change classes, or pick maximum-test epochs.

## Selection and actual method outcomes

First20KD selectedE140 train99.7085/val65.0177/test83.2849/macro62.0726;
the continued interval is independently reported below, not substituted into
the first interval's receipt or historical snapshot.

`method_review.csv` includes every completed method, including negative outcomes.
Genreg20 retained baseline two/six; four selectedE112 gained5 correct tests
(83.2364->83.4787), macro62.8181->64.4978. Same-head six selectedheadstep2 gained
one correct test (83.0426->83.0911), macro60.3422->59.9487: aggregate improvement
did not improve every class metric. Deep KD method selectedE149,
headstep0: test83.23643411,
macro61.52958058; method-family accepted:
False. Every metric in a row comes from the SAME selected
checkpoint. Actual endpoint learning/teacher coverage is separately preserved
in `six_layer_distillation_history.csv`; it is not substituted for selection.

First minimum validation EMA balancedNLL chooses checkpoints; 1e-6 ties keep
the earlier candidate in head/KD stages. Tests never choose an epoch. The
predeclared method-family acceptance keeps its prior accepted model if the
chosen method loses test accuracy. This is explicit TEST-informed development,
not independent unseen generalization or a same-budget causal residual claim.
Residual extra epochs/head updates and KD supervision are disclosed. Same seed17
only; no multi-seed stability claim. Highly imbalanced TEST2064 must be read
with macro and all20 class supports/recalls, not aggregate accuracy alone.

Model-digest TEST counts for residual2/4/6 at this snapshot:
1/2/4
(baseline included); each frozen control1.
No old model was forwarded again when its saved receipt could be genuinely reused.

## ABO verification and exact deployed graph

Actual ABO `tasks/t07_abo_image_retrieval/standalone/model.py` was read. Its
detached per-sample RMS-normalize e/o -> learned weighted sum -> normalize mixed
RMS -> restore electronic reference RMS equations were matched in values and
gradients during admission. Each main-stage alpha is learned in(.7001,.95),
initial.8. This is a FEATURE fusion coefficient, not a fixed.8/.2 probability
blend and not a guarantee of global optical energy fraction. Original ABO bounds
and task state were never modified. The fixed-probability draft never trained.

One router only, E-G repeated1/2/3. Actual historical Kather forward order/router
count was checked, with its different nine-expert geometry disclosed. Current
four224experts/global478/RGB100/joint8bit. Both rho arms have identical electronics.
Each stage has a bias-free3x3 kernel9 plus alpha1; final pooled16/nonaffineLN/
Linear256to20 is5140parameters. Total electronics5160/5180/5200 (float32 parameters
about20.16/20.23/20.31KiB), optical phases479364/908552/1337740. All were trained
from simulation from the beginning; future physical work loads the SAME trained
graph/state. KD teacher exists only in training, never student deployment.

## Saved identities and limitations

- Baseline source649ef57e580f8ee490210f417b7b55468d316b8c /168: science`548c5a90f818b5cb6f3f77f34edbe429d28e6e5d20ffdf01a013437689303ba3`, TEST`93db16f16e4f8ecbae1b07591595a8399dfe308ae3245dd3f7b4927e13fde8a7`.
- Generalization block0001: science`d7982e78b436a8da86b51c0501a4c87e3423795cdc4a7643d253e1ab0037c464`, TEST`befb2998c948553e77ff8bece285f5d481d35948cbbafbc9ae52079388bf3ce9`.
- Generalization block0002: science`c8968e7f440c8996829da4969d79b01410d079e4e9a63454415469c2553653dd`, TEST`2187cfe00e1fd625284ede0916a166a6b1b4572c5b09e198828875774e15bbaf`.
- Same-head science:`464fe46821b0192db96df55db0a39d630b6e7a9460bd292815d0f1534839c40b`.
- Same-head saved_test_audit:`b5c5400bc3867bfa4a3a1d38939f43992b5bb08b49f33fa00955af0a60fc3ca4`.
- First20 KD science:`2785817be614d7ca56f13f1dcee1c414fbe77bef8f4fa18c4f917cf738ebd32b`.
- First20 KD saved_test_audit:`8e5d530d60329e35345ab34eabb58b1ef610c9aed9018b124b528385d0f0588f`.
- Continue20 KD science:`7e962c158e2a380eacd0f2d56bfd42f71f8d4d596e42f6993989269c64233b21`.
- Continue20 KD saved_test_audit:`0fce922cf806b357a0df39c3f7dcfbf0490bf74d023ee6e71b6f89b0823997a0`.

Generalization source5287fa023fc06a7e4b2867f1185110b1d07e2569/174;
head sourcedd0bb004a73182f75f2e243262a3df64e070a0ec/178;
firstKD sourcedfffc29a97bedde106ece9d0a2dd601e354452c0/183;
continuedKD source409187050a61ca031af6ff87412b67f7129a511b/188. These are actual frozen
source identities, distinct from later report HEADs. Final read-only collectors
rechecked source blobs, selectedPT/cache/resultSHA, completedauditSHA/auditrc0,
CPU/producer exits and slots; they executed0new forwards/audits. Selected PT
SHA values are in `depth_comparison.csv`; private PT/IDs/rawJSON/logs stay private.

Fixed data1372TRAIN/283VAL/2064TEST, original20 classes and roles unchanged.
Cache SHA a81e78bbff06ea65e7ceba3771b688178276464e33fdcdedf3796fa2930fa6e1.
Class screening was validation-aware; old Caltech tests were observed; no claim
of subject/near-duplicate independence. Old initial biased-conv GPU gate failure
was preserved and corrected before any optimizer/test; accepted baseline is the
bias-free version. No gate threshold was relaxed, failed claim overwritten,
old test/fit/audit re-entered, or schedule created. Baseline plots remain a
separately labeled equal100 snapshot, not this refined comparison.

## Requested old Caltech10 CCD picture retirement

The authorized acquisition chat completed deletion at2026-10-16 14:58:58BJT:
124512 six-digit raw/cropPNG pictures,82060300260filebytes,actual disk free-space
increase82270609408bytes (82.27GB). Target picture remainder0. It retained48
ccd.npy caches (14224506048bytes),weights/features/predictions/IDs/receipts,
canonical inputs/phases and pilot/response/calibration pictures.45 protected
assets were SHA-verified. Pictures were not duplicated into an82GB backup.
Deletion inventorySHA0972b0619173109186d3f2d66c492e593172238ce7ab28911c1da880164c4671,
logSHA6560a8098f7c13ed4202b95cd18320805b6e9aaf6c89fd250457fd9ff85058c2.
This report does not authorize deletion of remaining caches or new acquisition.
