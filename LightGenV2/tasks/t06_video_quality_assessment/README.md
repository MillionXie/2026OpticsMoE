# T06 视频质量评价

## 原训练集2250条六层实采与读出头适配（20260914）

**20260915用户撤回自动换层授权：当前必须人工通知才换相位。**
单层入口`lab_manual_stage --single-write-phase --capture-only --verify-phase-optically --phase-reference-dir ...`：
只写一次所指定相位并保持，三次参考拍摄和采完复查不再切flat或重复写相位。
本层完成后状态为`stage_complete_wait_for_user`，不生成下一层、不换mask、不启动微调。
此前自动采集的可疑router CCD隔离保留，重采不跳过这些旧PNG。停止仅作用于本任务拥有的进程/租约。

20260915相位复核：用户发现language CCD疑似默认文字，原2250会话的语言层与微调队列已暂停，
不可仅凭SDK返回值、BMP SHA或亮度合格验收。`lab_supervise --verify-phase-optically`在同一相位SDK
持有进程中，先固定全白振幅、150μs诊断曝光，执行flat→target→flat→target，要求同状态重复PCC≥0.95、
异状态PCC≤0.98且两者分离≥0.02；采完当前层再用相同探针检查PCC≥0.95。探针不进入数据集，
诊断曝光不改变正式采集配置。还可用`--phase-reference-dir`核对已登记、SHA锁定的目标参考光场：
跨时间参考同时要求原始PCC≥0.90、固定sigma=1 ROI像素轻平滑后的结构PCC≥0.98，以区别细碎光斑波动和错误图样。
平滑仅用于跨时间诊断，切换/同状态重复仍用未平滑PCC，正式CCD绝不据此平滑或逐图归一化。
该检查证明可重复的切换响应，不证明相位逐像素准确，也不证明整批每一帧都正确。
失败即停止，不自动放宽阈值。原始证据保留于对应run，旧test CCD也需复核后才能重新启动适配。
`lab_exposure_session --retain-prefix 3`可在新会话仅继承SHA及有效设置核验通过的前三层，
不复制可疑语言CCD、下游BMP或推理输出；第四层起必须重生成和重采。原会话不覆盖。

本轮用户授权自动换完六层。`build_lab_package.py --bench shs --target spatial --split train`
导出原Spatial 0.6710权重对应的2250条原训练视频，每条4帧、一幅场，共13500次采集。
新会话独立，不覆盖既有558条test CCD。曝光沿用已验证的400μs，语言router/expert为1600μs，
240ms等待；两层曝光补偿、ROI、方向及相位255-g编码均继承已验证合同。
`lab_supervise.py`调用共享相位持有器：当前层采集、下一层prepare和逐文件SHA审计都成功才换层；
黑场/饱和/身份错误停止，不跳过样本。状态在指定run的`supervision/status.json`，放`STOP`可停采集。

训练包包含原558条test的实测读出特征（不是模拟CCD），以及`adapt.py`。
启动`adapt.py official_queue --project . --session-dir sessions/train2250_20260914
--output sessions/train2250_20260914/readout_adaptation --eval-cache evaluation/test558_readout_cache.pt
--device cuda --epochs 100`，会等待全部六层与审计完成，再提取训练特征并在实验电脑GPU训练。
仅更新原有readout，lr1e-5、batch64、L2-SP0.1、EMA0.98、100epoch；不加载之前用test微调的权重。
训练2250/test558身份必须不重叠，test不反传；每epoch按test SRCC选best，故不是未触碰测试集。
保留best/last，冻结张量哈希必须相同。`readout_adaptation/queue_status.json`为后处理状态，
最终查看`readout_adaptation/adaptation/original_train2250_test558/results.json`。
该命令不操作设备，不会与等待期间的采集争用GPU；真正训练在实采评估完成后才启动。

## 20260914：空间实测读出头适配

`adapt_measured_readout.py` 是离线入口，不调用设备。基于原始Spatial 0.6710权重及
`spatial_lang1600_20260914` 全558视频六层实测CCD，先逐图校验SHA并回放原始forward，
缓存最后 `model.readout` 的三个输入；缓存预测必须复现微调前实测结果。
服务器特征回放和训练统一关闭CUDA matmul/cuDNN TF32。跨Windows torch2.8/cu126与Linux
torch2.6/cu124存在约0.01 MOS的卷积数值差异：先以GPU/CPU FP32对照核验，再联合限制最大
单条误差0.03 MOS、全量SRCC差0.0001和RMSE差0.005；逐条差异及通过/失败保存在
`measured_readout_cache.replay_audit.json`。不得只放宽单条阈值而忽略排名变化，原采集预测保留。
只有原有 `readout.*` 可训练，光学mask/router、alpha、所有前置电子层和MOS尺度不变，
没有新分支或教师损失。全量和80%两版相同初始化/seed/学习率，各100epoch，AdamW、
Smooth-L1 + 0.2排序 + 0.1相关性损失；只保留best和last完整checkpoint。

- `full100`：558条全部反向传播，以同集SRCC选best；这是适配集重代入指标，**不是独立test**。
- `split80`：固定seed 20260914，446条训练、112条不反向传播，以112条SRCC选best；
  该留出集仍参与选模，**不是完全未触碰的最终test**。保存具体视频身份，不能把两版不同分母直接作公平性能比较。
- 原实测SRCC 0.5786364901保留不变；适配权重不覆盖原固定权重、不重新生成采集输入。
- best包括epoch0原始头，若没有改善不会强行交付更差权重；保存冻结参数SHA与实际变动张量清单。

服务器工作目录切到对应Git commit后：

```bash
python -m LightGenV2.tasks.t06_video_quality_assessment.adapt_measured_readout extract \
  --project /absolute/path/original_spatial_lab_package \
  --session-dir /absolute/path/spatial_lang1600_20260914 \
  --output /absolute/path/run/measured_readout_cache.pt --device cuda
python -m LightGenV2.tasks.t06_video_quality_assessment.adapt_measured_readout train \
  --cache /absolute/path/run/measured_readout_cache.pt \
  --checkpoint /absolute/path/original_spatial_lab_package/weights/best_checkpoint.pt \
  --output /absolute/path/run/adaptation --device cuda --epochs 100
```

新产物位于 `runs/hardware/spatial_readout_adapt_20260914`，不覆盖原会话；适配后checkpoint含
`hardware_adaptation`来源信息，不能直接冒充原SHA固定推理包，需要另行经过适配权重部署验证。
两版已各完成100epoch：全量同集SRCC 0.9987881（epoch97，不是test）；80%版留出112条
SRCC 0.6177597→0.6271390（best epoch1），之后过拟合；该权重全558条SRCC 0.6330451。
原始未适配结果不变，详见[实测读出头适配报告](reports/reproduction/SPATIAL_SHS_READOUT_ADAPT_20260914.md)。

后续仅训练手段优化入口为 `tune_measured_readout.py`，固定配置
`configs/spatial_hardware_readout_tuning.json`：三组低学习率/L2-SP原权重约束/EMA，单GPU顺序执行，
每组100epoch。保持原446/112视频身份不变，启动前逐ID核对旧`split.json`；不训练全量版，不引入新网络。
EMA只作为一个候选参数状态，推理不增加分支；以112条留出SRCC选epoch和试验，不能将多次选择后的数值
当独立test。每组只保留best/last，原先best也纳入最终候选比较，不强行替换为更差的新权重。
三组与额外“仅末端402参数”对照现均完成100epoch，最好为lr1e-5/L2-SP0.1/EMA0.98的epoch64：
同一112条留出SRCC **0.6281555452**、RMSE9.0845031880；较此前SRCC0.6271390仅小幅改善。
最佳权重已下载到同一run的`regularized/low_lr_ema/split80/best_checkpoint.pt`并校验，尚未部署。
完整对照表和限制仍在上述同一实测读出头适配报告，不把混合训练/留出的全558条分数当test。
调用时提供 `--cache --checkpoint --reference-split --reference-result --output`，分别对应上一轮特征缓存、
原始固定权重、上一轮80%版split/results、任务下新run目录。只在`runs/hardware/`保存新产物。

## 当前SHS实验台迁移（2026-09-14）

打包入口 `build_lab_package.py --bench shs`、运行入口 `lab_bench.py` **仅绑定** Temporal `multivideo16x4_rank_s163`
（SRCC 0.8044、16视频×4帧）与 Spatial `spatial_readout_1m_srcc067`（SRCC 0.6710、
单视频4帧），均校验正式checkpoint SHA256。它直接复用模型原始forward的六次传播边界，
避免旧hardware_contract遗漏最新电子残差或标量门控。构建时检查原模型与六层仿真CCD回填
一致性；全558条缓存交付还必须通过目标SRCC复评。未通过不得声称已迁移完成。
硬件操作与两台电脑分工见 [COMMAND_SHS.md](hardware/COMMAND_SHS.md)。默认严格逐层，只有用户
通知后才换相位并开始下一层capture。2026-09-14这轮用户另行明确授权AI监督并换完全部层，
因此按每层检查后换层执行；这不是后续所有实验的默认授权。ABO数据与会话不受影响。

本轮两个完整558条test离线包位于任务的 `releases/20260914_shs_temporal08044.zip`
和 `releases/20260914_shs_spatial06710.zip`。固定权重复评分别为 SRCC
**0.8043868643** 和 **0.6710968960**；空间历史参考为 **0.6710079009**，两次差约
0.000089，不能把复评值假写成历史值。权重SHA与原候选一致，六层浮点CCD回填一致性检查
通过；这些是仿真检查，不代表实测精度或8μm重采样后的物理精度。

师弟电脑部署目标为 `E:\code\guest\2026OpticsMoE\LGVQ_Temporal_Lab_SHS_8um`
和同级 `LGVQ_Spatial_Lab_SHS_8um`，复用同级ABO工程已有驱动和GPU环境。部署是否完成
以本地 `runs/hardware/shs_20260914/<temporal|spatial>/deployment.json` 为准；相位BMP
复制在该目录的 `phase/`。每项任务初始 `pilot01` 只准备4幅场（时间64视频、空间4视频）；
完整时间35幅场×6层=210次采集，完整空间558幅场×6层=3348次采集，不含补拍。
当前包是固定权重实测推理/评估包，只带test缓存，不宣称含完整微调训练数据。
400μs曝光、240ms等待继承自当前ABO配置，仍须对LGVQ各层检查暗场/饱和；部署不启动硬件。

2026-09-14 实采使用新ROI文件 `E:/code/guest/20260914.txt`，按此前已测的左右镜像
把相机TR对应到逻辑TL；不将相机画面角标签直接当模型角标签。新会话为
`temporal_full_20260914`，覆盖全部35幅场/558视频。单层本地协调器为
`lab_manual_stage.py`：主线程持有指定相位，带心跳的师弟电脑桌面任务只capture当前层、
prepare下一层输入，然后停止等待人工换层；第六层后才evaluate。它不自动切相位。
本轮入口与质量统计在 `runs/hardware/temporal_full_20260914/00_查看这里.md`；
六层已完成210/210张正式CCD、558个完整test视频，SRCC **0.7977138739**、PLCC **0.8091420696**。
完整数据已下载本地并逐文件校验SHA256，逐视频结果已独立复算，无筛选或剔除视频。
第五层信号弱但两次复拍可重复；第一层光场与仿真仍有明显差异，不能将最终相关性接近仿真
当成光学完全匹配或光学贡献证明。源码、指标和限制见 [本轮实测报告](reports/reproduction/TEMPORAL_SHS_20260914.md)。

空间实采另建 `spatial_full_20260914`，使用上文固定空间权重和20260914新ROI，
400μs曝光、240ms等待、Gain_X4。每层558幅，完整六层3348张；与时间会话独立。
统一查看入口为 `runs/hardware/spatial_full_20260914/00_查看这里.md`。
用户另行授权本轮AI监督并切完各层。空间六层3348张、558视频已完成，SRCC **0.5786364901**、PLCC **0.6154004576**，
低于仿真SRCC 0.6710968960。因第四、第五层信号不足，诊断后两层改为1600μs并按曝光倍率补偿；
其余层仍400μs，最终会话`spatial_lang1600_20260914`，继承层有有效设置及原记录SHA验证，未筛选视频。
完整限制、身份和指标见[空间实测报告](reports/reproduction/SPATIAL_SHS_20260914.md)。

> 2026-09-10 架构审计：`spatial_single_video4_srcc06665` 含冻结预训练
> ResNet18 前端，现已降级为“非合规性能上界”，不能作为正式方案引用或部署。
> 当前正式自研卷积候选为 `spatial_single_video4_custom_conv`：不含命名预训练
> backbone，额外 E1 卷积为 316,568 参数，SRCC 0.6553。

## 当前结论

Spatial 仍是一条视频均匀取 4 帧并排成 2×2，**没有多视频复用**。旧归档
`spatial_single_video4_srcc06665` 在 558 条 test 视频上得到 SRCC 0.6665，但它
使用了冻结预训练 ResNet18，已被架构审计否决，只能作为容量上界。其证据保留在
[`reports/paper_results/spatial_single_video4_srcc06665`](reports/paper_results/spatial_single_video4_srcc06665/README.md)，
不得当作正式结果。当前正式自研卷积候选在同一 558 条 test 上得到 SRCC 0.6553、
PLCC 0.6849；同 checkpoint 关闭光学为 SRCC 0.5624，光学贡献为 +0.0929。它保留
物理光 Router Top-2、四个光电融合阶段和固定 20% 未调制分量，运行时不含
MobileNet/ResNet/VGG 特征依赖。结构说明见
[`SPATIAL_CUSTOM_OEO_ARCHITECTURE.md`](SPATIAL_CUSTOM_OEO_ARCHITECTURE.md)。

面向部署的首选压缩版本为 `spatial_single_video4_compact_readout`：它只替换四层
光学之后的读出头，把读出头从 1003.1 万压到 212.3 万参数（减少 78.84%），完整
学生从 1285.0 万降到 494.2 万参数；完整复评 SRCC 为 0.6547，较原模型仅低
0.00059。结构与证据见 [`SPATIAL_COMPACT_READOUT.md`](SPATIAL_COMPACT_READOUT.md)
和 [`reports/paper_results/spatial_compact_readout_20260910`](reports/paper_results/spatial_compact_readout_20260910/README.md)。

当前主版本是 `temporal36_balanced`：一个视频均匀取 36 帧，以 6×6 lane 放进同一个
478×478 有效光场。四专家光学 Top-2 router、六次光传播、20% 名义未调制直流分量、
鲁棒位移/相位/CCD 扰动和目标专属电子读出头保持不变。

正式测试集 558 个视频：SRCC 0.8454、KRCC 0.6394、PLCC 0.8650、RMSE 7.183、
MAE 5.451。平衡候选四专家占比正常，结果详见
[`reports/paper_results/temporal36_balanced`](reports/paper_results/temporal36_balanced/README.md)。

新的正式仿真候选 `temporal_multivideo9x4` 把 9 条互不相关的视频各取 4 帧，以 3×3 视频
tile、每 tile 内 2×2 帧的方式放入同一 478×478 光场。它不是把 Temporal-36 checkpoint
改名，而是新的六次全场相干传播模型，输出形状为 `[B,9]`；其中每个数仍只评价一条视频。
正式单 seed 结果为 SRCC 0.8082、KRCC 0.5999、PLCC 0.8131、RMSE 8.226、MAE 6.109；
视频路由最大专家占比 28.0%，跨样本选择变化率 55.7%，九槽位循环审计平均 SRCC 0.8019。
完整证据见
[`reports/paper_results/temporal_multivideo9x4_contentroute`](reports/paper_results/temporal_multivideo9x4_contentroute/README.md)。
Temporal-36 保留为“单视频 36 帧”的独立基线，二者不能混报。

最新吞吐优先候选 `temporal_multivideo16x4` 在同一光场同时处理 16 条视频、每条 4 帧，
输出 `[B,16]`。正式单 seed 结果为 SRCC 0.8044、KRCC 0.5968、PLCC 0.8180、
RMSE 7.991、MAE 5.992；没有达到预设 SRCC≥0.81，但一次光场输出数相对 9×4 增加
77.8%，且没有发生全局专家坍缩。正式配置和诚实的限制说明见
[`reports/paper_results/temporal_multivideo16x4`](reports/paper_results/temporal_multivideo16x4/README.md)。

论文“大模型 baseline”的 Spatial 行不是 Temporal 结果复用。它固定每视频 4 帧、
448×448 输入、冻结 `Qwen3-VL-2B-Instruct`，只训练 5×2048 个质量词输出行。正式配置为
`configs/baselines/qwen3vl_spatial_quality_tokens_4f_r448.yaml`，执行命令见仓库根目录
`RTX5090D_QWEN_BASELINE_PROTOCOL.md`；性能、单视频速度和板卡功率必须在同一 RTX 5090 D
dataset-once run 中产生。

## 不可静默改变的任务合同

- Spatial 与 Temporal 是两个独立单指标模型；当前 profile 只输出一个 Temporal MOS。
- Temporal prompt 必须存在；不能把文本输入从模型合同中删除。
- 冻结 Qwen 图像/文本前端负责生成缓存；学生推理图不得加入 Attention 或 Transformer。
- 当前 router 是四专家光学 Top-2，不是电子 router。
- 当前物理合同是 532 nm、17 µm 仿真像素、10 cm、518 canvas、中心 478 有效孔径。
- 当前正式鲁棒仿真包含至少 20% 名义未调制光功率；四个融合 alpha 受同尺度归一化约束。
- 当前 Temporal-36 表示一个视频的 36 帧。改成多视频必须新建 profile、checkpoint 和报告。
- MultiVideo-9×4 的 9 个标签和 9 个输出必须一一对应；不允许九视频聚合成一个分数。
- MultiVideo-9×4 必须整幅联合传播；逐视频传播后软件拼接只能作为容量上界，不能报告为硬件结果。
- MultiVideo-16×4 同理输出 16 个独立 MOS；64 帧只共享光传播，不共享标签或读出结果。
- 以上任一项变化都不能覆盖 `temporal36_balanced` 的名称或结果。

## 当前源码关系

为了不复制并分叉已经验证的核心模型，当前唯一后端仍是：

```text
experiments/qwen3_vl_2b_lgvq_single_metric_o2_16frame_54
```

本地和源服务器核心源码 SHA256 已核对一致。`LightGenV2` 现在接管唯一入口、新 runs、
正式报告和 releases。这个兼容关系集中写在一个 profile 中：

```text
configs/lightgen/temporal36_balanced.yaml
```

## 仿真操作顺序

以下命令都从 `2026OpticsMoE` 根目录执行。

```powershell
# 1. 环境和文件检查
python -m LightGenV2.scripts.check_environment --task t06

# 2. 不加载数据的 CPU 冒烟测试
python -m LightGenV2.tasks.t06_video_quality_assessment --phase smoke

# 3. 正式训练前检查缓存、manifest 和初始化 checkpoint
python -m LightGenV2.tasks.t06_video_quality_assessment --phase preflight

# 4. 正式训练；新产物自动进入本任务 runs/simulation
python -m LightGenV2.tasks.t06_video_quality_assessment --phase train

# 5. 用正式平衡 checkpoint 评估
python -m LightGenV2.tasks.t06_video_quality_assessment --phase evaluate
```

上面未显式指定 `--profile` 时默认运行 Temporal-36。运行单视频 Spatial-4 时必须显式写：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --profile spatial_single_video4_custom_conv `
  --phase evaluate
```

指定 checkpoint 或输出位置时使用：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment `
  --phase evaluate `
  --checkpoint D:\path\model.pt `
  --run-dir D:\path\evaluation_run
```

每次入口都会保存 `resolved_launch.yaml`、`launch_identity.json`、`command.txt` 和
`run_manifest.json`。继承配置中的数据路径会先冻结为绝对路径，因此把输出迁移到
LightGenV2 后不会错误地相对到新 config 目录。

如果本机缓存位置与源服务器不同，复制根目录 `paths.example.yaml` 为
`paths.local.yaml`，仅填写 `t06` 下有差异的项；入口会在生成最终配置时应用这些
覆盖，不需要修改正式 profile。

## 硬件与交付

- 六阶段顺序：[`hardware/README.md`](hardware/README.md)
- 构建实验室完整包：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.build_lab_package
```

该命令默认使用 SHA256 为
`159b1d8cd31aa5f817d274f2930129601d4f0a365f01c430a8fefcc5989c8730`
的 Temporal-36 平衡 checkpoint，并把 ZIP 写进 `releases/`。如果当前机器没有权重，
命令会明确报出缺失路径；可传 `--checkpoint`，或直接在源训练服务器构建。

## MultiVideo-9×4 训练

新图的任务内入口为：

```powershell
# 结构、梯度与禁止 Attention/Transformer 检查
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo9x4_formal.yaml `
  --phase smoke

# 缓存/manifest 检查
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo9x4_formal.yaml `
  --phase preflight

# 评估已经选定的正式权重
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo9x4_formal.yaml `
  --phase evaluate `
  --checkpoint LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/multivideo9x4_contentroute_d30_s114/best_checkpoint.pt
```

训练集每个 epoch 都会重新把 2,250 条视频随机分成 250 组，并随机交换组内九个物理
slot；测试集固定为 62 个物理场、558 条视频，指标仍对 558 个单视频预测计算。训练损失
同时包含 MOS 回归、排序/相关性、教师软标签、光电对齐、Top-2 专家均衡、保护带能量和
周期性 slot 置换一致性。相位调制保留 20%–35% 相干直流分量、k 空间限制和输入/相位/
CCD 位移扰动。配置目录只保留 `base` 与 `formal`；历史 sweep 参数已经写进正式报告，
不会再以一批失效 YAML 干扰后续操作。

六次传播的含义依次为：36 帧光 router、144 帧专家、9 个视频内帧融合、9 个视频
router、36 个视频专家、9 个视频 global。视频 router 只读取每视频 4 个已受 prompt
条件化的帧摘要；完整的 4 个图像 token 与 38 个文本 token 仍进入后两级专家/global，
避免公共 prompt 能量把不同视频的路由差异淹没。后端共享同一个无 Attention/Transformer 的
电子读出头，对九个视频分别调用，最终输出九个连续 Temporal MOS。

当前六张 mask 的实际覆盖率和 9×4 多视频设计见
[`reports/handoff/MASK_LAYOUT_AND_MULTIVIDEO9X4_PLAN.md`](reports/handoff/MASK_LAYOUT_AND_MULTIVIDEO9X4_PLAN.md)。

正式候选训练完成后，必须做 9 次循环换位审计。该审计把同一批视频依次放入九个物理槽位，
同时检查路由是否随样本变化、预测对槽位是否稳定，以及在**不重新训练**的条件下关闭光支路会下降多少：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo_audit `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo9x4_formal.yaml `
  --checkpoint LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/multivideo9x4_contentroute_d30_s114/best_checkpoint.pt
```

`slot_cycle_audit.json` 是正式比较依据；只报告九个槽位合并后的全局专家占比不够，因为不同槽位固定选择
不同专家也可能伪装成“均衡”。视频级 router 的 `selection_variation_fraction` 必须大于零，才能说明
Top-2 选择确实随视频内容改变。

正式损失中的路由 diversity 项最小化 `H(expert|sample)-H(expert)`：一方面让单条视频的
Top-2 选择明确，另一方面让同一物理槽位上的不同视频使用不同专家。固定选同一对专家、
或者对四个专家始终犹豫不决，都不会被误判为有效均衡。

最佳 checkpoint 的六次相位排布可按论文常用的 Arial 7 pt 生成两张 18 cm × 5 cm 图：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.visualize_multivideo_masks `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/<formal_profile>.yaml `
  --checkpoint LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/<run_id>/best_checkpoint.pt
```

黑色只表示该次传播中没有可训练相位的保护区，不代表实际 SLM 必须在该处吸收光。输出同时包含 PNG、
嵌入字体的 PDF 和每次传播的占用率/相位统计 JSON。

## MultiVideo-16×4：16 个视频并行

该 profile 在相同的 518 仿真 canvas、478×478 有效孔径、10 cm 距离和六次全场传播下，
同时评价 16 个互不相关的视频，每个视频固定取 4 帧。输出为 `[B,16]`，每个槽位仍对应
一个视频的连续 Temporal MOS，不会把 16 个标签聚合成一个结果。

精确排布为：4×4 个 115×115 视频 macro tile，起点为 `[3,122,241,360]`，相邻视频间
4 pixel；每个视频内部为 2×2 个 56×56 frame lane，相邻帧间 3 pixel。因此第一部分是
8×8、共 64 帧并行。帧专家为 27×27、内部间隔 2 pixel；视频级专家为 56×56、内部
间隔 3 pixel；global 相位为每视频 111×111。有效孔径尺寸和光路均未改变。

探索候选已经完成并清理。今后复现只使用唯一正式配置：

```powershell
python -m LightGenV2.tasks.t06_video_quality_assessment.multivideo `
  --config LightGenV2/tasks/t06_video_quality_assessment/configs/lightgen/temporal_multivideo16x4_formal.yaml `
  --phase train
```

新正式训练默认只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。每 5 epoch 仍测试并
更新同一个 best 文件，但不再生成周期相位 PT。只有独立命名的 mask-evolution 研究才可
把 `phase_snapshot_interval_epochs` 改为正数。

## 冻结 Qwen 方案二电子 baseline

`qwen3vl_quality_tokens_r448` 是纯电子 baseline 的正式 profile，不改变上述光电模型。
Qwen3-VL-2B-Instruct 全部冻结，只训练五个质量词 `Bad / Poor / Fair / Good / Excellent`
对应的 5×2048 输出行，共 10,240 个参数；五路 softmax 概率按训练集 MOS 范围内的五个
等距质量分数加权，得到一个 Temporal MOS。

每个采样帧固定为 **448×448**。当前 Qwen3-VL 视觉前端是 16×16 patch、2×2 空间 merger
和 temporal patch size 2，因此有效空间步长为 32；448 可被 32 整除。4/9/16 帧在视觉
block 0 前分别形成 1,568 / 3,920 / 6,272 个 1024 维 patch token；空间 merger 后分别
成为 392 / 980 / 1,568 个 2048 维视觉 token，再与 prompt 的文本 token 进入语言模型。
正式表中的输入始终是 448×448，不是 700 多像素。

4、9、16 帧的抽帧位置、65% 中心裁剪、Temporal prompt、训练/test 划分和随机定位解码
保持固定。每个帧数必须用对应特征训练自己的五个输出行。

正式计时是一组方案/帧数对应一个独立进程：模型只加载一次，不做显式 warmup，不做
test 前推理，第一条也进入 558 视频总体统计。主时间从 Vision Transformer block 0
输入到标量分数在 GPU 上就绪；模型加载和 MP4 解码/裁剪/resize/tokenizer/H2D 分栏保存。

```bash
CONFIG=LightGenV2/tasks/t06_video_quality_assessment/configs/baselines/qwen3vl_quality_tokens_r448.yaml
MODEL=/absolute/path/Qwen3-VL-2B-Instruct
MANIFEST=/absolute/path/lgvq_split.csv

# 检查分辨率、帧语义、模型、manifest、GPU 和 Git 身份
python -m LightGenV2.tasks.t06_video_quality_assessment.quality_token_resolution \
  --config "$CONFIG" --phase preflight --model "$MODEL" --manifest "$MANIFEST"

# 可断点续跑：依次提取 4/9/16 帧特征、训练 50 epoch、各自完整评估 558 个 test
python -m LightGenV2.tasks.t06_video_quality_assessment.quality_token_resolution \
  --config "$CONFIG" --phase all --model "$MODEL" --manifest "$MANIFEST"
```

如果中途终止，可把 `--phase all` 换为 `extract`、`train` 或 `benchmark`，并用
`--frames 4|9|16` 只续跑一组。正式结果进入 T06 自己的 `runs/simulation/<run_id>`；
论文图和紧凑结论进入 `reports/paper_results/`，大特征、checkpoint 和逐视频 CSV 不提交 Git。

当前 5090D 的 448×448 正式结果、完整指标表、token 几何、计时边界和论文图见
[`reports/paper_results/qwen3vl_quality_token_baseline_r448`](reports/paper_results/qwen3vl_quality_token_baseline_r448/README.md)。

## RTX 5090 D 光学 MoE 分段计时

T01–T04 与 T06 的 CCD 后串行电子处理、并行残差、跨层 SLM 场重建、bridge 和任务头已经
在同一块 RTX 5090 D 上按计算图边界正式测量。每个分量 50 次预热、1000 次正式同步调用；
T06 每次调用是一幅同时承载 16 个视频×4帧的物理场。任务级临界路径、能量口径、冻结
Qwen 对照和论文图统一见
[`reports/5090d_moe_and_qwen_baselines_20260907`](reports/5090d_moe_and_qwen_baselines_20260907/README.md)。

该报告明确区分物理光场时间、可被光路覆盖的并行残差、必须串行的 CCD 后处理和任务尾部。
其中 80.388 W 只能计算光学设备能量代理；在没有同步采集电子处理 GPU 功率前，不把它写成
完整光电系统能耗。
