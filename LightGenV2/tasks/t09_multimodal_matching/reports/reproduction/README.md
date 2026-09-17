# 复现入口

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
