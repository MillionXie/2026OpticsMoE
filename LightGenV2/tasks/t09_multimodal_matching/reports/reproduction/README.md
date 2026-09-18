# 复现入口

## 2026-09-18：上下布局OEO恢复训练

直接原因已由逐层探针确认：光传播到窗口的能量非零，但原OEO按全场做中心化LayerNorm，再ReLU，会将窗口内低于全场均值的响应全部截为0。两个窗口都为0时，带epsilon的能量归一化输出0.5/0.5；对应ReLU的导数也为0。布局改变与路由分配会改变光强分布，但不能据此说上下布局必然失败。

原MoE最佳权重在验证前32问答中，末层OEO前两个窗口平均能量为1.23467e-4、2.70659e-5，OEO后均为0；两个窗口内中心化值最大值仍分别为-0.03881、-0.06823。该批分类损失对首层、全局层、路由相位梯度范数均为0。这是特定权重/批次的证据，不代表训练每一步全体样本的梯度都为0。此前全验证集检查已确认最佳权重100%零读出。

两次新实验均从头训练两个架构，保持数据、说话人划分、问题、上下布局、两层主光路、每层含末层OEO、窗口、seed17、30轮、batch32、Adam及学习率计划不变，无CNN、无可训练文本编码器、无额外电子分类头、无dropout。只改变两模型共同使用的OEO响应：
- centered Softplus：原中心化LayerNorm后用Softplus替换ReLU，再Softsign，存在正背景响应。
- non-centered Softsign：对非负光强除以空间均值，直接计算u/(1+u)，去掉中心化LayerNorm与ReLU，不在零输入处产生正背景。
两者均重新归一化为单位功率振幅并重置相位。这是OEO传递函数对照，不是数学等价的代码修复，也未取消末层OEO。

|OEO|模型|验证选中epoch|训练accuracy|验证accuracy|测试accuracy|
|---|---|---:|---:|---:|---:|
|原中心化ReLU|MoE|2|50.01%|50.00%|49.94%|
|原中心化ReLU|D2NN|21|92.05%|83.45%|82.76%|
|中心化Softplus|MoE|30|78.80%|76.45%|76.70%|
|中心化Softplus|D2NN|27|55.88%|54.63%|54.96%|
|非中心化Softsign|MoE|28|90.67%|86.24%|84.54%|
|非中心化Softsign|D2NN|30|79.92%|77.58%|77.51%|

每个模型独立按最低验证NLL选择checkpoint；训练、验证、测试使用同一个选中权重，测试不选epoch。独立预测重算准确率、选模规则及checkpoint SHA均已核对。新组最终评估训练/验证/测试零读出比例均为0。恢复可学习性已完成，未证明过拟合消失：新MoE训练与测试仍差6.12个百分点；新D2NN训练也较低，选中末轮，仍可能欠拟合。Softplus的D2NN明显欠拟合，不用其低分证明MoE优势。原同布局D2NN82.76%、旧重复布局D2NN85.06%均保留，不以较弱新版本替代强baseline。

新MoE验证平均四路功率约48.13%、47.34%、3.85%、0.67%，主要使用前两路，未形成均衡四专家使用。均衡本身不是目标，需通过固定路由、专家干预等验证分工价值。其测试读出capture约0.000193，非零不代表硬件信噪比足够；当前仍为理想仿真。

新run为 `audio_raw_twoband_{softplus,positive}_{moe,d2nn}_s17_v1`，测试各加`_test`，位于`runs/simulation/`。Softplus训练源码6867624d，非中心化Softsign源码14b11fd3。完整命令、环境、数据SHA见各run metadata.json。复现使用上节命令，将architecture设为单个moe或d2nn、input-layout设two_band，分别增加`--oeo-activation softplus`或`--oeo-activation intensity_softsign`，其它参数保持不变。MoE/D2NN分别运行于两张RTX4090（GPU UUID见上节前两卡）；已完成测试并释放两卡。

探针位于`runs/smoke/audio_twoband_oeo_probe_s17_v1`、`audio_twoband_softplus_probe_s17_v1`、`audio_twoband_positive_probe_s17_v1`；后两者只替换响应检查原权重梯度，不作为重新训练的性能。独立审计为`audio_twoband_softplus_audit_s17_v1`、`audio_twoband_positive_audit_s17_v1`。图、逐项结果、下载SHA在`reports/figures/oeo_recovery_s17_20260918/`，可通过`python -m LightGenV2.tasks.t09_multimodal_matching.plot_oeo_recovery`从本地run证据重画。

当前结论与后续优先级：
1. 图文：CLEVR颜色/形状存在性判断，固定CNN前端和one-hot文本。30轮无phase dropout测试MoE69.73%、D2NN71.80%；两者均加0.05 phase dropout为72.40%、57.20%，后者欠拟合。对比各自更强配置仅差0.60个百分点，不能用弱正则baseline宣称大优势。本轮未对图文更换OEO。
2. 音文：mini Speech Commands关键词条件判断，CNN版94.23%/94.87%；无CNN旧重复版85.58%/85.06%；无CNN交织版83.85%/83.33%。CNN为相同网络结构、不同任务权重，并非图文那份权重。去掉CNN后仍学得到，但尚无跨配置稳定的大优势。
3. 先固定输入与数据，完成OEO、读出能量及路由利用的对照；给两模型相同调参预算、各自依据验证集选配置，不强求同一种正则对两者都有效。D2NN未收敛时不靠增加dropout解决。反复查看测试后的开发应透明记录，最终论文需要新的独立确认，不能把当前测试当未被参考的最终证据。
4. SONYC-UST v2.3仅核验过CC BY 4.0许可和标注，尚未训练。适合下一阶段音文条件判断，但标签衍生问题不等于原生自然语言描述。先稳定当前实现再迁移，避免同时更换任务、布局及响应而无法定位原因。


最新状态：上下布局音文MoE的零读出原因已定位，并已成对重训MoE/D2NN；见上方“上下布局OEO恢复训练”。图文仍为此前CNN前端版本，本轮没有重训图文。所有结果为seed17探索结果，不能据此宣称稳定优越性。

本任务比较两层MoE＋逐层OEO与D2NN＋逐层OEO。技术参数、编码张量和训练流程见[任务README](../../README.md)。所有图文结果为seed17，训练图像1000张、验证250张、测试250张；每图3正3负问答。视觉前端共用并冻结，文本按配置共享；无电子输出残差、无Qwen。

## 原始编码对照

以下为最低验证NLL检查点；测试在检查点锁定后统一执行。

|文本编码|模型|训练准确率|验证准确率|测试准确率|
|---|---|---:|---:|---:|
|one-hot|MoE|81.45%|72.13%|70.67%|
|one-hot|D2NN|66.37%|58.87%|57.60%|
|稠密正交|MoE|81.22%|69.33%|70.47%|
|稠密正交|D2NN|64.47%|58.27%|55.27%|
|GRU|MoE|78.87%|71.00%|71.60%|
|GRU|D2NN|81.00%|71.73%|72.20%|

直接RGB未使用CNN的对照验证准确率约49–53%。固定编码下MoE领先，GRU下接近，不能宣称普遍领先。MoE训练与测试仍有7–11个百分点差距；部分配置探测窗口能量很低，尚未评估硬件噪声。

训练run：runs/simulation/clevr_visual_aux_s17_v1（CNN），clevr_visual_fixed_s17_v1、clevr_visual_dense_s17_v1、clevr_visual_learned_s17_v1。fixed/learned源码b86c35b5，dense源码cfbb0118。原始RGB为clevr_fixed_s17_v1、clevr_learned_s17_v1。完整命令、环境、源文件SHA及完整commit在各run metadata.json。

独立复核：runs/smoke/clevr_visual_audit_s17_v2/verification.json。报告中的[审计副本](../figures/text_encoding_s17_20260917_v2/verification.json)包含每个best checkpoint SHA256、逐样本指标核验、验证选模及共享前端身份核验。测试run：runs/simulation/clevr_frozen_test_s17_v1，评估源码b6c57c3a；locked_selection.json为事先锁定权重，data_manifest.json为测试数据SHA；[测试核验摘要](../figures/test_s17_20260917/test_verified.json)。

数据目录：/DATA/DATA1/guest3/demo_reproduction_data/clevr_attribute_s17_v1。父manifest SHA256：a5826f81d530e41e776eac528886936d64b5ad413e62268510fa62c0f09e5ade。

## 12轮泛化对照

|设置|MoE训练%|MoE验证%|D2NN验证%|
|---|---:|---:|---:|
|原配置|81.45|72.13|58.87|
|lr0.003|82.53|68.07|53.60|
|特征dropout0.1|79.72|70.53|58.33|
|水平翻转|81.27|71.40|59.00|
|翻转＋dropout|80.15|71.13|57.13|

新增组未评估测试。run命名为clevr_visual_fixed_{lr003,drop01,flip,flipdrop01}_s17_v1；前两组源码095abdaf，后两组8f57db74。独立审计、权重SHA、实际命令与环境在[泛化图目录](../figures/generalization_s17_20260917/)。没有明确解决过拟合，原版本保留为准确率参考。

## 运行方式

服务器仓库：/DATA/DATA1/guest3/demo_reproduction_20260915；Python：/home/guest3/miniconda3/envs/xml/bin/python。以下命令从仓库根执行，DATA指数据目录，OUT必须为任务runs中的未存在路径。

```bash
python -m LightGenV2.tasks.t09_multimodal_matching.prepare --out DATA
python -m LightGenV2.tasks.t09_multimodal_matching.vision --data DATA --out VISUAL_RUN
python -m LightGenV2.tasks.t09_multimodal_matching.run --phase train --mode fixed --seed 17 --epochs 12 --batch 32 --lr 0.01 --data DATA --vision-checkpoint VISUAL_RUN/best_checkpoint.pt --out OUT
```

文字GRU使用--mode learned及默认8轮共享前端预热；稠密固定码用--mode fixed_dense。30轮对照使用--epochs 30，phase dropout组另加--phase-dropout 0.05。所有训练默认关闭phase dropout和其他增强。phase dropout smoke在runs/smoke/phase_dropout_s17_v1，验证单位模透射、评估关闭、相位梯度及更新。

[训练/验证/测试图](../figures/test_s17_20260917/train_val_test.png)；[原始示例](../figures/text_encoding_s17_20260917_v2/validation_examples.png)。原始权重、数据、逐样本预测不提交Git；下载的结果副本随附SHA清单。

## 30轮预算与phase dropout复核

源码c8cec5fe，seed17，同一固定one-hot与共享冻结CNN，从头训练30轮，余弦周期30轮。各配置内按最低验证NLL选模，未再次评估test。

|配置|模型|选择轮次|训练准确率|验证准确率|验证NLL|
|---|---|---:|---:|---:|---:|
|无phase dropout|MoE|8|80.82%|69.00%|0.5794|
|无phase dropout|D2NN|18|81.23%|72.13%|0.5557|
|phase dropout 5%|MoE|22|81.63%|73.13%|0.5693|
|phase dropout 5%|D2NN|30|62.35%|57.40%|0.6567|

D2NN原12轮预算训练不足；延长后准确率明显提高。不能继续把原13个百分点差距视为充分训练的架构优势。phase dropout对MoE有帮助，但显著抑制D2NN，必须同时保留不加dropout的强D2NN。不同配置间也不能仅按差距挑选。MoE phase-dropout组仍有8.50个百分点训练/验证差距，未消除过拟合。
run：clevr_visual_fixed_long30_s17_v1和clevr_visual_fixed_long30_phase05_s17_v1。两组独立核验通过，权重SHA及逐轮曲线在[图目录](../figures/phase_budget_s17_20260917/)。[训练充分性曲线](../figures/phase_budget_s17_20260917/phase_budget.png)。

## 音文预实验

派生数据 /DATA/DATA1/guest3/demo_reproduction_data/mini_speech_matching_s17_v3：去重后6263条训练语音、843条验证语音、867条保留测试语音，按说话人SHA1互斥。每条2个匹配问题，训练12526问、验证1686问；原始8关键词分类与二分类匹配的准确率不能混为一谈。该子集来自Speech Commands v0.01；许可依据压缩包README引用原版本，及Google Research原发布页的CC BY 4.0声明，见数据license_evidence.json。测试只保留记录，未解码或评估。

共享音频前端：同样32128参数CNN，临时8类头1032参数；仅用训练语音标签，预训练30轮后按验证NLL选择并去掉分类头，冻结128维特征。不把关键词预测类别或分类logits直接输入光学网络。未使用音频水平翻转或时间反转。
音频前端run：audio_frontend_s17_v1，源码248a381b，选中第28轮；八类分类训练91.79%、验证85.88%。这不是光学音文匹配成绩。
光学首轮仍为固定完整句子one-hot、两层逐层OEO、共享前端、相同训练预算；仅验证选模，不先假定MoE领先。

## 音文配对结果与精确输入审计

30轮预算，固定one-hot，两层逐层OEO，lr0.01余弦到0.001，batch32，无phase/特征dropout；最低验证NLL选模。训练前端只保留128维特征，二分类匹配标签每条语音正负各一。

|模型|选择轮次|训练准确率|验证准确率|验证NLL|
|---|---:|---:|---:|---:|
|MoE|9|96.63%|94.31%|0.1667|
|D2NN|23|97.17%|94.01%|0.1682|

MoE−D2NN为0.30个百分点；按验证说话人聚类重采样的条件95%区间为[-0.52,1.07]个百分点，包含零，且不包含训练种子及选模不确定性；不能声称明确优势。未评估音频测试集。

重要审计：首轮两张逻辑CUDA设备实际为RTX4090与A100，同一CNN特征的相对L2差约0.12%，未通过逐字节相同输入核验。保留该初试run，但不作为正式配对对照。通过check_features.py在原设备重算，先复现原特征哈希，再保存MoE原输入的固定缓存。D2NN使用该缓存重训；新的两组训练/验证特征及前端checkpoint SHA完全一致。GPU选择改为UUID，避免CUDA与nvidia-smi序号不同。

报告所用run：audio_matching_moe_s17_v1（源码6d661c8b）与audio_matching_d2nn_canonical_s17_v1（源码819d0ec8）。旧audio_matching_d2nn_s17_v1仅作为跨设备数值诊断保留。数据manifest SHA256：e20d789937a44512ce4c91f222ba99145e36c63f3ee386295585d61cc018c0c1。共享音频CNN checkpoint SHA256：ce733c832a1bc7629cf9b5f63f22d46668fc5e1a5b0442cb97b092fca0a9b2b3。

独立核验runs/smoke/audio_matching_audit_s17_v2通过：重算预测准确率/NLL、验证最优轮次、权重SHA及精确特征身份。输入依赖诊断runs/smoke/audio_modality_diagnosis_s17_v1：固定文字后两模型均50%；固定音频特征后49.47%/49.88%；随机打乱音频后48.75%/48.93%。均保留原标签，是扰动诊断，不是新任务成绩。MoE推理时将路由强制等功率，准确率从94.31%降到86.83%；这是同一权重的推理干预，不是重新训练的固定路由baseline。正常路由的各支路功率均值约31.28%、21.52%、30.27%、16.93%，仍为四支路密集加权。

[音文曲线与双模态诊断图](../figures/audio_matching_s17_20260917/audio_matching.png)。图目录保留独立审计、数据互斥检查、实际命令/环境、前端哈希与传输SHA清单。所有本轮训练及诊断进程已退出。

音频复现入口：
```bash
python -m LightGenV2.tasks.t09_multimodal_matching.audio_prepare --out AUDIO_DATA
python -m LightGenV2.tasks.t09_multimodal_matching.audio_frontend --data AUDIO_DATA --out AUDIO_FRONTEND --epochs 30
python -m LightGenV2.tasks.t09_multimodal_matching.run --data AUDIO_DATA --vision-checkpoint AUDIO_FRONTEND/best_checkpoint.pt --mode fixed --epochs 30 --architecture both --out AUDIO_RUN
```
单进程both会只提取一次共享前端特征后依次训练两架构，避免跨设备特征差异。拆分设备运行时应传入同一个已校验的--feature-cache。所有OUT必须为新的任务runs路径，旧run不得覆盖。
# 2026-09-17补评：已锁定模型的独立测试

本节为固定权重复评，未重新训练。测试源码commit `fe5f5d8b`，PyTorch及设备信息、CNN SHA256、数据SHA256见各评估run的metadata.json；checkpoint路径/轮次/SHA256见locked_selection.json。先锁定验证NLL最优权重，再使用测试标签。测试特征只提取一次，所有相应模型共用。光学训练与预训练来源仍见下方历史记录。

|任务/配置|模型|选中轮次|训练accuracy|验证accuracy|测试accuracy|
|---|---|---:|---:|---:|---:|
|CLEVR，30轮，无phase dropout|MoE+OEO|8|80.82%|69.00%|69.73%|
|同上|D2NN+OEO|18|81.23%|72.13%|71.80%|
|CLEVR，30轮，phase dropout 0.05|MoE+OEO|22|81.63%|73.13%|72.40%|
|同上|D2NN+OEO|30|62.35%|57.40%|57.20%|
|Mini Speech Commands音文匹配，30轮|MoE+OEO|9|96.63%|94.31%|94.23%|
|同上|D2NN+OEO|23|97.17%|94.01%|94.87%|

表内三个划分均对应同一个验证选中的checkpoint，训练评估关闭dropout；不是最后一轮训练accuracy与最佳轮测试混用。CLEVR测试250幅图、1500问题；音文867段音频、1734问题、175个未参与训练/验证的说话人。逐样本概率独立复算accuracy/NLL通过，绘图脚本还核对训练审计中的checkpoint SHA与测试锁定SHA一致。只有seed17，不代表多种子结论。

图文有phase dropout的MoE与充分训练无dropout的D2NN相比，测试仅高0.60个百分点；不能报告相对训练不足的dropout-D2NN有15.20个百分点“架构优势”。音文MoE测试低0.63个百分点；按说话人重采样的条件95%区间为[-1.40,0.10]个百分点，不能支持明确领先，且此区间不包含随机种子及选模不确定性。

图文无正则MoE后期训练接近99%且验证NLL明显变差，是过拟合；phase dropout缓解后期恶化，但选中模型训练81.63%、测试72.40%，仍有泛化差距，不能宣称消除过拟合。音文选中模型差距约2.4/2.3个百分点，当前高准确率更应结合简单任务与任务监督电子前端解读。

输出：`runs/simulation/clevr_long30_test_s17_v1`、`runs/simulation/audio_test_s17_v1`；下载证据及SHA清单：`reports/figures/selected_test_s17_20260917/transfer_manifest.json`；图：同目录`train_val_test.png/pdf`；独立复算摘要：`verified_summary.json`。所有评估进程结束，使用的RTX4090显存已释放。

服务器项目根目录下的实际命令（Python为`/home/guest3/miniconda3/envs/xml/bin/python`，CUDA_VISIBLE_DEVICES均为`GPU-1b963983-7909-af6e-0528-f0f0661ab549`）：

```bash
python -m LightGenV2.tasks.t09_multimodal_matching.test_selected --kind clevr --data /DATA/DATA1/guest3/demo_reproduction_data/clevr_attribute_s17_v1 --runs LightGenV2/tasks/t09_multimodal_matching/runs/simulation --out LightGenV2/tasks/t09_multimodal_matching/runs/simulation/clevr_long30_test_s17_v1 --cache LightGenV2/tasks/t09_multimodal_matching/runs/simulation/clevr_frozen_test_s17_v1
python -m LightGenV2.tasks.t09_multimodal_matching.test_selected --kind audio --data /DATA/DATA1/guest3/demo_reproduction_data/mini_speech_matching_s17_v3 --runs LightGenV2/tasks/t09_multimodal_matching/runs/simulation --out LightGenV2/tasks/t09_multimodal_matching/runs/simulation/audio_test_s17_v1 --cache /DATA/DATA1/guest3/demo_reproduction_data/mini_speech_matching_s17_v1/mini_speech_commands.zip
python -m LightGenV2.tasks.t09_multimodal_matching.plot_selected_test
```

每次正式配置完成后评估验证选中权重的测试结果并完整记录；测试不用于选择轮次、阈值或正则。后续调参公平合同和完整电子/光学计算图见任务README的“当前计算图、测试与公平调参合同”。下文“未测试”为各历史记录当时状态，以本节补评为准。
## 2026-09-18：去除CNN与重复输入的重新训练

本次是重新训练，不是替换输入后直接复评旧权重。数据仍为原mini Speech Commands，训练6,263段/12,526问答、验证843段/1,686问答、测试867段/1,734问答；说话人互斥。输入固定log-mel、固定逐词one-hot，零可训练电子参数；保持两层主光路、每层含末层OEO、两个64×64读出窗口。不能称为没有电子计算，因为STFT、插值、归一化和OEO仍为电子处理。

|输入布局|模型|选中epoch|训练accuracy|验证accuracy|测试accuracy|
|---|---|---:|---:|---:|---:|
|无CNN，旧三槽重复|MoE+OEO|13|94.20%|86.77%|85.58%|
|同上|D2NN+OEO|20|95.07%|85.65%|85.06%|
|无CNN，上下两区|MoE+OEO|2|50.01%|50.00%|49.94%|
|同上|D2NN+OEO|21|92.05%|83.45%|82.76%|
|无CNN，按行交织|MoE+OEO|12|92.27%|82.50%|83.85%|
|同上|D2NN+OEO|19|92.50%|84.76%|83.33%|

表中同一行的三个成绩来自各自验证NLL最低的同一权重。NLL=-mean(log(p_true))，考虑正确答案的概率，不只看argmax；因此选中的epoch不一定具有最高accuracy。两个模型分别选各自最优epoch；不同配置也分别选，不共用一份pt，不依据测试分数选型。旧三槽MoE最后一轮训练98.68%不是表内第13轮的94.20%，不能混用。

旧有CNN参考测试为MoE94.23%、D2NN94.87%。去掉CNN后旧排布仍约85%，说明不是完全依赖CNN才能匹配；也不能把CNN版的性能全归于光学部分。去重复后的交织布局能学习但没有超过重复布局；不重复不等于更优。仅seed17，两个可学习布局的测试差距均约0.52个百分点，不据此宣称稳定架构优势。

**退化读出证据：** 上下分区MoE的验证选中权重在所有验证样本上两个窗口能量都为0，输出概率完全为0.5/0.5，验证NLL=0.6931473，验证zero_detector_fraction=1。该模型保留为失败对照；NLL规则没有选择错误权重，而是这些epoch均未形成有效分类，均匀输出反而比自信错误的输出损失低。末层OEO的截断及路由偏置是后续排查重点，不能把此次监测新增描述成已修复。其余五个无CNN模型的选中验证权重zero_detector_fraction=0，仍有明显泛化差距。

训练/测试run：`runs/simulation/audio_raw_legacy_s17_v1`、`audio_raw_twoband_s17_v1`、`audio_raw_interleaved_s17_v1`，各自测试run追加`_test`。训练源码前两组`bfb7474b`，交织组`4db53f15`；后续`ebffcd09`诊断支持布局、`fb4cd716`增加逐样本零能量比例，均未更改模型输出数学或选模规则。metadata.json包含完整命令、PyTorch/CUDA环境、GPU、参数和数据manifest SHA；best_checkpoint_sha256见各result.json及报告verified_results.json。三组测试输入张量和问题SHA逐字节相同，复用原划分；不经过CNN所以不存在CNN跨GPU特征缓存差异，但跨布局分别运行于RTX4090/A100的训练浮点差异仍需后续同卡复核。

公共训练命令（在服务器工程根目录，Python为`/home/guest3/miniconda3/envs/xml/bin/python`）：

```bash
python -m LightGenV2.tasks.t09_multimodal_matching.run --data /DATA/DATA1/guest3/demo_reproduction_data/mini_speech_matching_s17_v3 --out LightGenV2/tasks/t09_multimodal_matching/runs/simulation/RUN --mode fixed --architecture both --epochs 30 --seed 17 --input-layout LAYOUT
python -m LightGenV2.tasks.t09_multimodal_matching.test_selected --kind audio --audio-runs RUN --data /DATA/DATA1/guest3/demo_reproduction_data/mini_speech_matching_s17_v3 --runs LightGenV2/tasks/t09_multimodal_matching/runs/simulation --out LightGenV2/tasks/t09_multimodal_matching/runs/simulation/RUN_test --cache /DATA/DATA1/guest3/demo_reproduction_data/mini_speech_matching_s17_v1/mini_speech_commands.zip
```

RUN/LAYOUT分别为上述三组run与`legacy`、`two_band`、`interleaved`。默认batch32、Adam lr0.01余弦至0.001、梯度裁剪1、无dropout。前两组GPU UUID依次为`GPU-4d8bfdb9-8777-05a6-3811-ab18ff4eadfd`、`GPU-1b963983-7909-af6e-0528-f0f0661ab549`；交织为`GPU-3f60d773-4dd8-d706-ef48-a4dfc5af6c55`。同一run内两个架构在同卡顺序训练，共用同一数据张量。

输入、功率、梯度smoke在`runs/smoke/audio_layout_s17_v1`、`audio_layout_interleaved_s17_v1`；三组独立预测/选模/权重SHA审计为`runs/smoke/RUN_audit`。固定模态扰动及读出检查为前两组`RUN_diagnose_v2`、交织`RUN_diagnose`，均通过验证基线复核；不是重新训练的消融baseline。报告图与汇总在`reports/figures/audio_input_s17_20260918/`：`input_and_readout`为示意图，`no_cnn_learning_curves`为学习曲线，`no_cnn_train_val_test`为柱状图，均有PNG/PDF；`verified_results.json`记录成绩与权重SHA，`readout_audit_summary.json`记录配置、输入SHA与零窗口比例，`transfer_manifest.json`记录下载证据SHA。原始逐样本预测保留在runs，不放入论文图。全部进程已结束，GPU资源已释放。
