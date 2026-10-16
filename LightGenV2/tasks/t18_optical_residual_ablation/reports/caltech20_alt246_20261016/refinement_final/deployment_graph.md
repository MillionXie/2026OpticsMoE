# Actual simulation/deployment graph, originalCaltech20

当前推理：Router 一次，Expert→Global 重复1／2／3次，末端池化／归一化／
同一个已训练 Linear(256,20)。每个主层均有相同的小电子残差与可学习α，
逐样本 RMS 归一化后融合，α>.7。不存在仿真不训练末端、实拍另加新头的安排。
以下给出逐步图与部署参数合同；具体采用哪一个权重以同目录CSV的SHA为准。

Router is evaluated once on the encoded original input to compute four expert
weights. The original input amplitude starts the first expert block; router
captured intensity is not substituted as that block's amplitude. Reuse the
same routing weights at every expert block; do not repeat the router.

- 2 main layers: Expert1 -> Global1.
- 4 main layers: Expert1 -> Global1 -> Expert2 -> Global2.
- 6 main layers: Expert1 -> Global1 -> Expert2 -> Global2 -> Expert3 -> Global3.

At EACH main block, the incoming amplitude has an optical path (phase modulation,
propagation, captured intensity and fixed OEO transform) and the tiny electronic
path (identity + .1*tanh(bias-free1to1,3x3conv of normalized input), clipped>=0).
Both amplitudes are independently normalized by detached per-sample spatial RMS.
Mix with learned alpha*optical + (1-alpha)*electronic, normalize the mixture RMS,
and restore electronic-reference RMS. This fused amplitude enters the next block.
Every alpha is learned within(.7001,.95), initialized.8; it is not optical
rho, an output probability blend or guaranteed optical photon/power fraction.

After the LAST fused amplitude: adaptive mean pool16x16 -> flatten256 ->
nonaffine LayerNorm256 -> trained Linear256to20 -> softmax20. Classification
uses this same linear head in simulation and future physical adaptation. Fixed
20 detector windows still measure optical capture for the auxiliary objective;
their raw probabilities are not added via a separate .8/.2 classifier mixture.

The main-depth router count is excluded from2/4/6. Optical captures are3/5/7,
optical phase counts479364/908552/1337740, phase tensor counts6/11/16.
Electronics at each main block:9kernel +1alpha. Terminal head256*20+20=5140.
Total electronic counts5160/5180/5200. These are trained weights from simulation,
not random untrained readout weights. Both optical-rho arms have exactly this
same electronic graph/parameter counts, paired initialization/order within depth;
rho.3/0 differs only in the optical modulation residual during the100-stage.

Feature dropout and the L2 teacher exist only during training, not deployment.
The head-only optimizer's TRAIN standardization was folded into existing W/b;
there is no extra runtime standardization module or newly appended network.
The frozen L2 teacher is not a student state_dict submodule or an exported head.
The current KD candidate restores raw/EMA from the optical training lineage;
the separately tried20 LBFGS head updates are search cost, not falsely added to
the current Adam count or claimed to remain in a selectedheadstep0 candidate.

Deploy the exact accepted checkpoint SHA in depth_comparison.csv, including all
trained phase/electronic/alpha/linear tensors. A later physical-head fine-tune
loads THAT SAME5140-parameter linear head and can freeze all other tensors; do
not instantiate a differently shaped adapter or from-scratch replacement.
No new device execution/acquisition was performed or authorized by this report.
OldCaltech10 CCD archives are not pixel-matched features of these NEW20-class
optical checkpoints. Preserve the retained calibration/receipts/caches as an
archive; do not silently train a new20 head on old10 CCD features or delete
remaining caches without a separate explicit request.
