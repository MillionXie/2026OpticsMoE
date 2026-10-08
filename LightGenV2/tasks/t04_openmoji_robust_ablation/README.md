# T04 OpenMoji 鲁棒性消融：当前统一入口

## 2026-10-08 授权：累计组电子适配，目标89.0%–89.5%（进行中）

88.70%的best/last及完整报告现已本地备份，双PT SHA核验一致。
用户继续授权后，有限一次TRAIN特征mixup对照：从.8870 best继续、120轮，
原.08遮挡/.05标签平滑/定位1.5/困难重放500及decoder30162参数保持。
仅新增mixup loss权重.2：批内TRAIN随机配对，lambda均匀取[.25,.75]，
特征凸组合；完整监督分别用两个原source/edit/target计算，再按lambda加权，
不对离散类别ID插值。总loss=.8原配对目标+.2混合监督。
依据是TRAIN .9995对TEST .8870仍有明显差距；这是特征空间泛化假设，
混合特征不声称真实光学可实现或CCD标定，无新增推理层、TEST梯度/VAL。
先GPU单轮短测和strictCPU/上游保护，通过后唯一正式GPU任务；保留.8870，
不保证提高，不扫描混合比例/TEST阈值，原LSP本地GPU训练不受影响。

GPU标签平滑正式120轮已完成，任务返回0/实际Python退出/GPU回落400MiB空闲。
strictCPU best .8870/e105、last .8870；best TRAIN .9995，上游保护通过，
比此前.884提升.30pp、距.890仍.30pp，仍未达标。best SHA
10ed4da3f7b6202a71fe32aecad63bf57ac7398640db4cef6cb30091d68e1d3f；last SHA
94808c6479a2610b2303c7a4f98cde2aed27d33407ec8785233e65f68f25114c。
原best/last/全部报告保留，取回本地备份进行中；当前没有下一轮梯度训练。
一次预声明best/last固定50/50平均`editor16_align_decoder_smooth005_bestlast_average_20261008`
strictCPU .8860、TRAIN .9995，没有提高，不替换.8870，不扫描比例或阈值。
这不能解释为泛化差距消失；后续须有TRAIN-only新依据，不机械重复平滑权重扫描。

用户最新继续授权改进泛化并使用实验室空闲GPU。原.8840 best本地best/last/完整
report/history/protocol/逐样本备份已完成，两PT SHA与实验台报告相符。
`editor16_align_decoder_cpu_gpu_audit_20261008`只审计原TRAIN并计时临时decoder副本，
没有保存训练PT或用TEST梯度：TRAIN changed .999，对应TEST .884，差11.5pp，
支持存在明显拟合/泛化差距，但不能排除TRAIN/TEST分布差异。
CPU/CUDA各预热5步计时30步、batch32含缓存组批/监督loss/backward/AdamW，
CPU15.538ms/步、4060为5.763ms/步（约2.70倍）；此计时不含配对扰动和全TEST评估，
不是整轮或全流程加速承诺。实验台GPU原空闲，正式改用cuda，不抢本地LSP显卡。
有限一次类别CE label_smoothing=.05，保留.08遮挡/其他损失及120轮；composed目标、
edit标签和推理门限不变，从.8840 PT继续，不新增层/更改光学上游。
入口Git24aff1ad5已三端同步；GPU单轮短测loss .727552、last .8780，strictCPU
best仍为初始.8840、上游保护通过，不能称该短测提高精度。
正式唯一任务`OpenMoji_CachedSmooth005_GPU120_1008`，run
`editor16_align_decoder_smooth005_gpu120_20261008`，launcher28060/实际Python7528，
已真实epoch4/loss .7037337，GPU约580MiB。TEST每5最高开发选模/noTEST梯度/noVAL；
终了新增strictCPU TRAIN与TEST best/last报告，保留全部历史最佳，不保证.890目标。

最新TRAIN-only诊断：.8810 best在原TRAIN1000 changed-cell .996，说明继续增加
训练拟合不是主要目标。有限一次从该best继续，唯一配置变动为TRAIN特征遮挡.02→.08，
保持clean/noisy配对、噪声.01、原decoder、定位1.5、困难重放500及120轮预算；
不新增层或改变光学上游，不从TEST挖困难样本。Git b9d1bacc8已三端同步。
`editor16_align_decoder_mask008_smoke_20261008`单轮loss .1987193，strictCPU .8830，
上游保护通过；这是单轮短测，保留独立PT，不将其当正式120轮结果。
首次直接后台派发`editor16_align_decoder_mask008_120_20261008`的PID25508已退出，
仅落盘初始best/protocol，无梯度history；保留该失败启动，不称训练完成。
改为既有任务模板的独立计划任务`OpenMoji_CachedMask008_120_1008`，正式run
`editor16_align_decoder_mask008_scheduled_20261008`，launcher16344/实际Python7696；
已真实epoch2/loss .2111679，epoch1 .1987193；源仍b9d1bacc8，同级完整log。
不是双训练：重派前核无原Python，终了仍须strict best/last审计备份，.8810不覆盖。

上述计划任务现已完成120轮并返回0、Python退出：strictCPU best .8840/e70，
last .8795，上游保护通过；best SHA4c81abd78457aab91526d8a112959a612606d02028dedb961254ea43ae018d57，
last SHA4d1907d8c128fe70f78f8f0002942e046f85268c414b29c4307d0d33fe4e5841。
较.881提高.30pp，仍距.890目标.60pp，不能称任务达标；best/last及完整报告/逐样本/
history/protocol均保留实验台，正在取回本任务`runs/hardware`作本地SHA备份。
本轮结束没有下一训练进程，不机械反复提高遮挡率；下一步以TRAIN错误组成检验泛化正则。

最新继续授权：数据已在实验室既有rank64项目，同上游`lr1e5_wd005_160_20261008`
的TRAIN/TEST特征各1000、各约153MB，不需光路采集或传回服务器。
定位1.5对照strict CPU best .8780/e25、last .8725；best SHA
3b07b50d7aba413cc0d202b47c73460098d8264b39cc55923db5adacedf9de40已本地备份核验。
增加未编辑区域保护.5的对照best .8745/e15、last .8680，不替换.8780。
下一有限对照从.8780初始化，仅在完整TRAIN1000之外每epoch增加500次TRAIN困难样本
重放；固定权重为初始TRAIN changed-cell错误率的`1+2*error`，不从TEST挖困难样本。
原decoder/光学上游/损失定位1.5/正则化保持，120轮CPU，TEST每5最高开发选模。
Git c26cc967d已接入实验台；先`editor16_align_decoder_hardtrain_smoke_20261008`单轮
实测及上游审计，通过后才派发正式run，不将派发或短测称为新性能改善。

`editor16_align_decoder_hardtrain120_20261008`正式120轮已完成，任务返回0；
strict CPU best .8810/e45、last .8685，全部非decoder上游保护通过。
best SHA a98c93a27e89ee6d38a8222b7c970a7772e574c4d53f7faf102a8c18ff345c36，
报告及best已取回本地且SHA一致。较.8780提高.30pp，距.890仍.90pp，未完成目标。
单轮短测.8805仍保留，不替代正式最佳；当前没有下一轮训练，先检查TRAIN错误组成，
不机械重复120轮或继续无限定位权重扫描。TEST选择均为开发口径，不是独立泛化。

编辑定位监督对照`editor16_align_decoder_editfocus120_20261008`已完成120轮，
strict CPU best .8740/e5、last .8665，上游保护不变，任务返回0。
最新用户授权继续，有限单因素`editor16_align_decoder_editfocus1_120_20261008`：
同.8695初始平均PT、同缓存/seed/120轮/正则化，仅`--changed-edit-weight .5→1.0`。
唯一CPU任务`OpenMoji_CachedEditFocus1_120_1008`，不SDK/GPU、不复采、不增层；
完整TEST每5轮最高开发选模，无TEST梯度/VAL选模。不是从.874继续训，保持起点
相同便于配置比较。最高.874仍保存，不把后续较差PT覆盖它；终了严格CPU审计。

权重1.0对照已完成：strict CPU best .8770/e5、last .8710，上游保护通过，
任务返回0且进程退出；best SHA14ef916c0d9392306b6034b6f57292be8007e7d99893b439b5fe2f6297e7273b。
0/.5/1.0定位项对应.870/.874/.877，有限再验证1.5（同初始.8695、同seed与120轮），
唯一`OpenMoji_CachedEditFocus15_120_1008`，run `editor16_align_decoder_editfocus15_120_20261008`。
不保证继续上升，保留全部对照，核验新增定位收益与未编辑格误改；本系列不无限扫描权重。

用户要求继续增强电子泛化。新增`lab_cached_decoder_regularize.py`有限对照：
只读同上游已核验TRAIN/TEST特征，严格复测原CPU基线.7315后，从.8695平均PT
初始化原decoder；不重建特征、不采光路、不增层。TRAIN-only特征gain .98–1.02、
2%遮挡及1% RMS相对高斯扰动，clean/扰动监督各.5、一致性.05，AdamW lr1e-5/
wd.05，120轮预算。此为电子特征正则化假设，不是CCD标定模型；TEST不扰动、
不进梯度，每5轮最高开发选模，无VAL选择。先CPU单轮短测及best/last严格审计，
通过后才正式训练。单轮CPU短测现已返回0：原基线.7315、初始.8695，实际loss
.18736354，best/last严格CPU复载与上游保护通过；短测不是性能改善证据。
已派发唯一`OpenMoji_CachedRegularize120_1008`，实验台既有项目run
`editor16_align_decoder_regularize120_20261008`及同级log，CPU预算120轮；
source Git c9035d398通过bundle只接入该新入口，旧加载/采集backend不改。
该120轮已返回0、CPU进程退出：严格CPU best .8700/e15、last .8620，
上游保护通过，best SHA402f5637e9aae8370dbc53630c129a067c0fafa29c0b1963a2d9dff5c9ac3889，
last SHA490a20dd5eee210b162b71c35cab3110cd41ad99c4040f43227acc69980f5c5a。
较同上游平均PT仅提高.05pp，仍距用户.890目标2pp；不把历史另一上游.8705合并。
报告本地备份，原best/last/逐样本/history/protocol均保留；无SDK/GPU占用。

暂不推进单措施消融。历史87.05%是早期editor16累计G5的真实CCD适配：
原clean CPU .9165、直接 .5730，独立TRAIN1000梯度，原decoder30162参数、
160epoch、TEST1000每5epoch开发选模，第105轮最佳，strict CPU .8705。
原PT SHA094d01157e63d56d195f154bc0671e9e2ce43ac32df34f47d893ea233c0489a4；
适配best SHA7f3268c8d94dd2ec1389a9ebc247b0a1a5301a9ae9e43315177364803cd5e40a。
这不是后来直接.7315的PT，不能将两个版本拼成同一恢复链。
历史report/strict_reload/history在私有`editor16_robust_chain_20261003/g5_adapter`保留。

后来的累计对齐PT d4481ef804dbd8f27f2f24f98c08dc7bf161b3137cefcc2613310291fe56f653
自身CPU仿真.9120、直接.7315；已有末端适配.8670，低学习率160轮及另一320轮配置
最高strict CPU均.8685。320轮结果best SHA
cb1864fe7217bb4d0891d3712c37173877c84746baa4ed756fbdc989e7a77140。
均按无trick仿真.8950作为用户目标参照；89.0%即相差0.5个百分点，不换成较低参照。

新有限run `editor16_align_decoder_lr1e5_wd005_160_20261008`已派发既有实验台任务
`OpenMoji_AlignDecoder_LR1e5_WD005_1008`：从精确d4481ef8原部署PT重新训练，
同独立TRAIN1000/固定TEST1000，各原6000CCD只读复用；AdamW lr1e-5/wd.05，
160epoch/seed927，TEST每5轮最高选PT开发口径，无TEST梯度/无VAL选模。
仍只训练原decoder，不改光学相位、alpha或架构。理由是历史TRAIN接近饱和、
延长320轮未提高TEST，有限检验低学习率与更强正则化，不保证目标或宣称单因素因果。
派发成功不等于开始梯度；须核CPU缓存基线.7315、实际epoch/loss及终了strict CPU。
原上游保护SHA f385956d653201b10c1bcc1bb3e8bd1f6a25a987fc328f660e6666d26e3bf66f
必须不变。最终保存best/last/逐样本/报告和SHA，确认进程及GPU释放。

沿用Git封存源码`bc1951cfbd0bcbd60a6111884286b506790e20fb`，未修改实验台源码；
入口`lab_editor16_robust_chain.py` SHA0adc3fc6...bccb62、原实际适配backend
`lab_tune_g2_test.py` SHAd4c0c4fd...2c9a0f；来源逐项见
`maintenance/storage/T04_WINDOWS_SOURCE_PRESERVATION_20261005.json`。
参数仅在任务命令中配置，不新增工程/分支/worktree，不打开SDK、不重采、不覆盖旧run。
原始CCD/PT和ABO等封存保持，外部上传暂停。该160轮run现已完成，选择e160，
strict CPU .8680，best SHA080db54660b95503ebc89388cd525d4e4ff1694feb50972b0f6daf27bc216c60；
原光学上游SHA不变。TRAIN changed-cell .979，未达用户>=.890目标。

有限一次`editor16_align_decoder_average050_20261008`已完成：预声明50/50平均
同上游两份训练后原decoder权重（父.8685/.8680），不增加推理层、不做梯度或阈值扫描。
TRAIN .984、完整TEST1000 strict CPU .8695，较同上游此前最高.8685提高.1pp，
仍距.890目标2.05pp；不是独立泛化，不能与不同上游历史.8705拼成一条链。
平均PT SHA75bc667d3724ec55f5c6a7607e361ef236979f1a63ed95907e7c42ae9502b972，
严格保存后默认CPU复载通过、全部非decoder张量逐项相同、保护SHA保持f385956d...bf66f。
报告/逐样本已本地备份，PT原件保留，CPU进程退出，不占SDK/GPU。入口
`lab_decoder_average.py` Git9283481ad，仅一次对照，不继续无限TEST扫分。

## 2026-10-07 新授权：单对齐空间扰动训练（结果待实拍）

不改变下方封存rank64结果。editor16的新对照仍使用同一初始权重、完整TRAIN5000、
seed73和120epoch；插值概率1、单前向、paired-clean=0、consistency=0，不含CCD/DC，
不做实拍后电子微调。原光电参数共同训练，alpha/frontend冻结。
`train_editor16_robust_chain.py`新增TRAIN-only `--spatial-shift 1`
`--spatial-rotation .25 --spatial-dropout .02`：复光场实虚部共享仿射、零填充，
16×16粗空间遮挡，无inverted-dropout放大；每次光学调用/批共享几何和遮挡。
平移单位是仿真像素，旋转单位度，范围均为实验假设而非设备标定。
eval关闭全部新扰动，默认参数全零保持旧路径；不增加推理层。
四项合成测试在服务器Torch通过（恒等、零场/梯度、实虚一致、遮挡不放大），
不能代替新权重的严格CPU审计或实际光测。新PT必须独立新CCD，不复用旧上游采集。
仅在既有运行工作树按Git精确任务文件同步，保护原overlay，不新建工程副本。

封存指标日期：2026-10-02；源码治理状态更新至2026-10-06。主要仿真、捕获和适配源码
已归主线，但不是已经通过独立环境/设备回归的完整部署包，也不覆盖其他窗口的新实验。
旧 `t04_semantic_interaction` 是共享模型/数据后端，
不是可随意删除的试错副本。

## 当前 rank64（不要混用 rank16/rank48/完整头）

仅取回主线时可先看[rank64封存版本说明](reports/reproduction/RANK64_SEALED_20261002.md)，
其中有原仿真／直接实拍／微调的完整对应及PT路径；原私有完整报告和逐样本未搬走。
该说明绑定2026-10-02版本，不覆盖其他窗口的新实验。

主线入口已解除旧ABO控制脚本及资产工程source配置的依赖：采集显式提供原机器
配置、相位SDK/LUT，由已审计的主线SHS模块控制。末端微调也从主线读取模型配置。
见[参数与剩余资产要求](reports/reproduction/README.md)。本轮未切换实验台现用目录，
未打开设备；现场回归仍待排期，不能据此声称新测精度或全部迁移完成。

| 版本 | 原正常仿真（实验台 CPU） | 同权重直接实拍 | 原 decoder 微调 | 已有 bias 校准后 |
| --- | ---: | ---: | ---: | ---: |
| G2 无 robust trick | .9390 | .6185 | .9015 | .9045 |
| G5 CCD噪声+DC30%+训练内网格代理 | .9270 | .6910 | .9180 | .9305 |

服务器正常仿真 G2/G5 为 .9385/.9290；和实验台 CPU 计算路径分别列出，不能混成同一测量。
G1 理想与 G2 直接部署共用原 PT。G5 校准后 PT 自身理想仿真 .9295，不用它降低原 G1 .9390 比较基准。
共享头271,384参数；实拍仅训练原 decoder 的30,162参数，不增加层、不改光学上游/alpha。
微调用独立实拍 TRAIN2000 梯度，原 TEST1000 每5epoch选PT及 bias 校准，用户明确授权；
TEST 不进梯度，但这些是开发指标，不是独立测试。保留未校准最佳、last和校准最佳。

| 最终候选 | SHA256 | 原证据入口 |
| --- | --- | --- |
| G5校准最佳 | `1fa31ec7b30a554280d9115b54f580d40b9c754f805db4ec9714ec28d22af41f` | [主线G5版本及原报告身份](reports/reproduction/RANK64_SEALED_20261002.md) |
| G2校准最佳 | `1cbc3d2574827272dafee7102a4a402ace7eab3f373ea177cbc02e66d90fe6db` | [主线G2版本及原报告身份](reports/reproduction/RANK64_SEALED_20261002.md) |

原私有资产目录`handoffs/openmoji_robust_ablation_20260928/midrank48_candidate/`
包含 PT、report、history、逐样本和严格默认推理重载记录；不复制大权重进 Git。
该目录16份此前无恢复身份的历史辅助程序已原字节三端保全，见
[用途与恢复清单](../../../maintenance/storage/OPENMOJI_HISTORICAL_TOOLS_20261006.json)。
它们含rank48诊断、早期rank64准备、已过期队列及被拒绝的额外residual试验，
不是16个当前入口，也不能替代本页rank64最终协议；原PT/CCD/结果与测速均保留。
主线入口不要求新机器先拥有该私有目录才能阅读版本说明；原完整报告按封存页SHA定位。
每组 TEST1000 与独立 TRAIN2000 各六层，分别6000与12000真实CCD，保留源身份及收据，不跨组复用。
2000us/GainX4/wait240、既定ROI和方向、保零 bounded BMP；原捕获合同不变。
当前 rank64 精确版本的单样本延迟尚未测量，不能借用旧完整头或rank48时间。
正式报告的逐文件SHA与四项候选身份已登记在本任务的
[封存证据清单](reports/reproduction/FINAL_IDENTITY_20261002.json)；原完整报告/逐样本数据没有搬走。

## 实际代码与迁移边界

- 仿真实际源码：训练服务器 `.worktrees/t04_openmoji_robust_20260928`。
  已保护 HEAD `3b956503a4ecae9b6c20c789ead28621e42e6eeb`，另有未提交 profiles/README 和 rank64 runner，
  以 SHA overlay 保留；单拿 HEAD 并不代表最新版。
- 实验台：`E:/code/guest/2026OpticsMoE/OpenMoji_Robust_Rank64_SHS_20261002`，
  使用 `source/LightGenV2/tasks/t04_openmoji_robust_ablation`、共享模型及原设备模块。
- [源码来源清单](../../TASK_REGISTRY.json)；[复现入口](reports/reproduction/README.md)。
- 迁移必须连同 lowrank64 共享头支持、bounded profiles、CPU CCD 边界、TRAIN2000 微调与已有 bias 校准核对；
  不能把旧默认 rank16 捕获脚本直接标为 rank64 最新版。

本轮不启动新训练/采集/评估、不恢复外部上传。现有源码的发布/测试不等于
完整机器资产闭包、现场回归或实际运行目录切换；现用工程继续保护。

### 2026-10-05 源码治理补充（不覆盖上面的实验结果）

已发布共享 `lab_runtime` 和历史 `standalone` 依赖，实际服务器导入及六项合成CPU边界
测试通过；后者仍是历史标准头入口，不是rank64部署默认。实验台36份源码已用Git原字节
封存，其原目录和有效实验数据未动。

rank64原G2/G5权重的精确身份现集中在
[`configs/lab/rank64_20261002.json`](configs/lab/rank64_20261002.json)，两份原PT在实验台
重新计算SHA与该配置一致。`lab_checkpoint_identity.py`提供不加载模型／设备的文件SHA和
架构身份检查，九项测试通过；测试无需私有归档Git引用。
统一 `lab_shs_capture` 已接入显式 `--lab-identity`，不传仍是历史rank16，不能只换PT来
复现rank64。rank64要求CPU／2000us，并采用实验台原有保存前暗帧守卫；22项无设备单元
检查和实际服务器依赖导入／六项合成CCD边界检查通过。尚未做现场采集回归，
TRAIN2000微调及bias校准入口已按实际代码归入主线。现用rank64原工程继续保护，
完整迁移、设备依赖和现场回归尚未完成；这些治理检查不代表重新测量准确率。

统一入口的rank64选择参数为：
`--lab-identity LightGenV2/tasks/t04_openmoji_robust_ablation/configs/lab/rank64_20261002.json`。
项目／数据／设备依赖仍需提供原审计配置；本轮不自动执行该入口或替换现用脚本。

TRAIN2000入口为 `lab_tune2000.py`，已有bias校准入口为
`lab_calibrate_rank64.py`，均显式使用 `--group g2` 或 `--group g5`；不再依赖
运行过程中偷偷替换组别映射。保留原TRAIN2000梯度、每五轮TEST开发选模、仅原decoder
微调及现有edit_head.bias校准，不增加网络模块。TEST选模结果不是独立泛化指标。
五项入口合同测试及实际服务器九项CPU依赖／合成边界检查通过，包括原rank64 decoder
30162参数、bias保存后严格重载和默认门限等价、上游修改识别。这些测试未加载正式PT，
未重评数据集或打开设备，不能替代完整资产配置与现场回归。
源码发布 `20b8e46951a334a6414aca23578cef26356111b4` 已同步本机、GitHub及服务器main引用；
实际运行checkout、原数据与权重没有切换或修改。

补充TRAIN准备入口为 `lab_prepare_extra_train1000.py` 和
`lab_prepare_extra2_train1000.py`，按实验台原脚本原逻辑保留（仅文件名归入任务目录）。
依次使用seed1002／1003、每种操作250条，排除TEST和此前TRAIN的ID及原图SHA重复；
输出已存在时拒绝覆盖。这两个入口不会调用设备或选择模型。仍需提供原project及
相邻 `OpenMoji_Lab_SHS_8um/data` 数据位置；本轮仅用临时合成文件验证，未生成正式新划分。
