# T04 语义交互（OpenMoji）

## 2026-09-17：用户指定展示第65轮

恢复入口：`offline_tune.py --project 工程 --session test1000_02 --output 新run --replay-from 原100轮run --epochs 100 --stop-after-epoch 65 --lr 0.0001 --batch-size 32 --seed 20260916 --device cuda`。复用校验后的实测缓存，从原始模型及新AdamW重放，不加载原best/last；100轮学习率曲线不缩短。每轮记录与原历史的指标/loss误差，最后重载last核验全部三组、四操作指标；失败标为replay_mismatch，不冒充恢复成功。选定权重是新run的last（epoch65），不是自动best。

实测末端适配指定结果为epoch65：200条选模留出集修改格准确率0.8575、整场景0.725。完整口径及指标见[第65轮结果](reports/reproduction/HARDWARE_EPOCH65.md)。这不是原自动best，也不是未微调全1000条结果。2026-09-17已从原始模型按100轮日程重放到65轮，源码`7fdd931e`；重载PT及独立进程复评全部1000条分区/分组指标与历史第65轮误差0。选定权重为`runs/hardware/head_replay_epoch65_20260917/last_checkpoint.pt`，SHA256 `8d7c2f6788a9ac67f79f28a3b9065ca633a7d31f7beddde313fdeb87e5c96557`，已下载本地。原第65轮PT未保存，故不能声称逐位恢复；中间训练存在微小数值差异，核验记录如实保留。原始记录和部署配置未覆盖。

## 2026-09-16：续训至200轮已完成

从100轮last恢复末端读出头和AdamW状态，固定原最终学习率1e-6续跑101～200轮；800/200划分、冻结范围和选模规则不变。源码`440b1a8f`，实际GPU环境5项测试通过，任务退出码0。没有重新采集。
200轮仍未超过epoch50：200条选模留出集的修改格/整场景准确率，epoch50为0.8775/0.745，epoch100为0.8625/0.730，epoch200为0.8600/0.730；epoch200编辑IoU为0.8266666667，物体F1为0.9480288600。继续保留epoch50作为best，同时保存epoch200的last。best重载后全1000条全部指标误差0。
本地完整201轮指标、曲线、best/last和SHA256下载清单位于[runs/hardware/head_adaptation_800_200_e200_20260916/00_逐轮结果.md](runs/hardware/head_adaptation_800_200_e200_20260916/00_逐轮结果.md)。师弟电脑对应`OpenMoji_Lab_SHS_8um/runs/head_adaptation_800_200_e200_20260916`；缓存复用旧100轮目录，旧结果不覆盖。200条参与选模，不是独立最终test；全1000条包含800条适配训练数据。

## 2026-09-16：实测CCD去光与末端适配

新增离线入口`lab_adaptation.py`，不打开任何相机或SLM。对`test1000_02`完整1000条先复现六层实测结果，
再以同一原权重关闭两个模态全部光支路、电子系数恢复为1（与仿真去光协议相同，不单独训练去光模型）。
程序检查正常模式每样本经过6个实测替换边界，去光模式执行0次光传播。

用户确认四操作各200条适配、50条留出，总计800/200，seed20260916；固定随机划分不看正确率。
100epoch、AdamW lr1e-4、cosine、batch32；仅微调`shared_readout`的post_film、coordinate_projection、editor、decoder。
冻结相位、router、alpha、全部光电前端、语言位置汇总/language_pool及task_head，避免修改光学输入却继续使用旧CCD。
缓存最后读出头的输入，epoch0必须复现原1000条指标；每轮打印/保存两子集及全1000条、四操作全部原指标。
best按200条留出集scene-exact优先，其次changed-cell、IoU、F1，另保留last。该留出集参与选模，不能称独立最终test；
全1000条指标包含800条训练适配样本，也不能称独立test。原权重、CCD、`results.json`均不覆盖。

小型增量ZIP入口：`python -m LightGenV2.tasks.t04_semantic_interaction.build_lab_package --shs-offline-output 输出.zip`。
ZIP须校验SHA并解压到新的独立目录，不覆盖原硬件runtime。入口：
`python offline_tune.py --project 原硬件工程 --session test1000_02 --output 新run目录 --epochs 100 --device cuda`。
输出`same_checkpoint_remove_optical.json`、`split.json`、`epochs.jsonl`、长表`epochs.csv`、`all_metrics.png`、
`best_checkpoint.pt`/`last_checkpoint.pt`与`summary.json`。权重是末端shared_readout状态，先载入固定原模型，再载入该状态；不是完整新模型。
本轮100epoch已完成（源码`de8084b1`，师弟RTX4060；4项实际环境测试通过）。正常六层实测复评精确复现，
同权重去光1000条修改格0.4065、整场景0.2730（原实测0.8830/0.6050），不解释为光贡献百分比。
微调343316个末端参数，best为epoch50：200条选模留出集修改格0.8775、整场景0.7450、IoU0.8358333333、F1 0.9522550505；
同一留出集epoch0为0.8750/0.6050/0.7744166667/0.9129123654。epoch100修改格0.8625、整场景0.7300，存在后期过拟合。
best重新载入后全部1000条指标与记录误差0，原模型及被冻结的语言汇总状态哈希未变；任务退出码0、GPU已释放。
师弟结果目录`OpenMoji_Lab_SHS_8um/runs/head_adaptation_800_200_20260916`；本地同名run位于本任务`runs/hardware/`，
`00_逐轮结果.md`列出epoch0～100三组整体8项指标，CSV/JSONL还包含四操作分项；小文件与best/last均已SHA256核验下载，
`readout_inputs.pt`缓存保留师弟电脑。日志在`offline_adaptation_20260916/logs/job_20260916_215236.log`。

用户随后要求延长至200epoch：`--resume-from 原100轮run --epochs 200 --output 新run`，恢复epoch100的last权重及AdamW状态，
保持保存的最终学习率1e-6恒定，不重启cosine/不加热。复用原CCD缓存、800/200划分与冻结范围；续跑前复现epoch100指标。
新run继承0～100轮历史及原best，只追加101～200轮；旧run不覆盖。最终best在全部0～200轮中按相同规则选取。

## 2026-09-15：SHS单电脑六层部署

新增 [COMMAND_SHS.md](COMMAND_SHS.md)：师弟电脑同时连接Holoeye振幅、Meadowlark HDMI相位和SHS相机。
固定主版本 `routerfill_shared` epoch40（修改格87.15%，整场景69%），不使用旧98%模型。
权重SHA `a69ddcee827749fb9202f9aef11ea45011e433d8b2f0151be2eec3db7dbff9eb`。
实际层序是语言router/expert/global在前、视觉router/expert/global在后；每层后续输入来自上游真实CCD。
`build_lab_package.py --shs-base 原独立仿真ZIP --shs-output 新overlay.zip` 构建Git固定源码包，
`install_shs.py`核验两包与文件SHA后只安装到新目录。`run.py export`必须通过原1000test复评及六边界回放。
`run.py probe/auto`在登录桌面使用持久相位SDK；阶段前后与每50张进行物理变化/重复PCC检查。
`test1000_02`已完成完整1000test六层实采及固定权重评估：修改格0.8830、IoU0.8002333333、F1 0.9115457265、整场景0.6050。
实际使用用户Blink GUI手动保持各层相位，不是上述自动SDK/PCC模式；六层各1000张通过文件SHA/上游链审计，不代表物理响应完美。
同机仿真修改格0.8700、整场景0.6880：实测修改格提高1.30个百分点，整场景降低8.30个百分点，不能声称全面优于仿真。
未微调、未筛选样本；原始CCD及记录仍在师弟电脑`OpenMoji_Lab_SHS_8um/sessions/test1000_02/`。
本地证据集中在 `runs/hardware/shs_single_pc_20260915/sessions/test1000_02/results.json`，SHA256
`9892bef6e171e0878854767d3d9edefb5fec64998d2848248eff830c32957730`；配套六项审计和评估日志已下载。
电脑重启中断了第一次评估；恢复后任务退出码0，成功日志`logs/job_20260915_193031.log`。高速相机已归还，后续只做离线处理。
任务不改变权重、不增加TF/attention；相位沿用物理尺寸重采样、方向及反灰度编码，新硬件拓扑使用新session。

## 对外分享：独立 OURS 原头复现包

打包入口为 `build_lab_package.py`，交付文件存放在本任务 `releases/`；只含 OURS 原头 epoch40 best，
完整 train/test、词嵌入缓存、冻结视觉前端、相位 PT 和源代码，不需要完整 Qwen，也不含硬件控制。
说明模板见 [release_template/COMMAND.md](release_template/COMMAND.md)，唯一复现入口为包根目录的
`python reproduce.py verify / demo / evaluate / train`（四者择一，不是把斜杠一起输入）。
打包后必须在解压出来的独立目录验证完整性、单例和完整1000test，不得只验证源仓库能运行。

## 当前采用：恢复原38.2万参数共享读出头

按用户决定，主版本采用 `routerfill_shared` 与 `qwen_shared`（`shared_readout_variant: standard`）。
直接使用已完成100epoch的原best：光电 `routerfill_shared_s73/best_checkpoint.pt`（epoch40，
修改格87.15%），冻结Qwen `qwen_shared_s73/best_checkpoint.pt`（epoch25，84.20%）。
以上目录均在本任务 `runs/simulation/`，无需重复训练或把原头拼接到slim权重上。
精简版slim/slim_norm仅保留为消融结果，不作为当前主版本，不与原头baseline拼表。
原头保留两组条件卷积，另有前置FiLM及decoder预处理卷积；不把它描述为整个后端只有两次卷积。

复评现有原头best（仓库根目录，Linux缓存多进程读取先执行 `ulimit -n 65536`）：

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile routerfill_shared --phase evaluate --device cuda
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile qwen_shared --phase evaluate --device cuda
```

用于解释推理的真实测试样本：`test_000008`，指令 `Place a flower below the bicycle.`。
源图只有自行车（人类1起始坐标第1行第3列），目标是在第2行第3列新增flower，其余保持；
`routerfill_shared_s73/test_predictions.jsonl`记录该样本整场景正确。训练标签program/target不输入网络，
source_grid只在预测编辑掩码输出后用于保留区域合成。

## 2026-09-09：精简共享电子读出头

保守对照为 `routerfill_slim_norm` / `qwen_slim_norm`：删除相同卷积及前置FiLM，
但保留decoder的GroupNorm+GELU，仅比slim多384参数，合计269,272（比旧头少29.51%）。
使用下方命令时替换profile即可；输出目录同profile加`_s73`。两对均100epoch，保留全部结果，
最终必须成对比较相同头，不能从不同头中挑一组主方法和一组baseline拼表。

`routerfill_slim` 与 `qwen_slim` 使用同一个精简头：381,976 → 268,888 参数（减少29.61%）。
只删去两组条件卷积前的重复FiLM和解码器预处理卷积；保留两组条件卷积、位置线性汇总、
坐标映射、6×6平均降采样、类别/编辑输出以及训练用四操作辅助头。光路、光Router Top2、
同尺度融合alpha>0.4和原噪声配置均不变。没有新增TF/attention或其他分支。

两组均从头训练任务网络100epoch，seed73、batch32、相同v2数据和损失/学习率协议，
不从旧best截取部分权重。Qwen主干完整冻结，复用经过校验的原生特征缓存；双方共享头
初始权重相同、训练后独立。只保留best/last，光电组结束自动生成同best去光对照及相位图。

```bash
# 仓库根目录；服务器只运行已推送的源码。此限制用于多进程特征缓存读取。
ulimit -n 65536
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile routerfill_slim --phase all --device cuda
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile qwen_slim --phase all --device cuda
```

输出分别为 `runs/simulation/routerfill_slim_s73`、`runs/simulation/qwen_slim_s73`，新结果待训练完成。
旧共享头正式100轮结果：标准光电修改格87.15%、IoU0.8327、F1 0.9339、整场景69.00%；
强均衡84.85%、0.8993、0.9478、77.00%；冻结Qwen84.20%、0.7312、0.8757、54.20%。
旧标准光电同best去光降至40.55%（46.60个百分点），不解释为物理贡献百分比。
当前精简只针对末端读出头，不代表光电网络全部电子参数只有26.9万。

## 2026-09-08：语言Router铺满孔径 + 公平共享读出头

当前入口为 `routerfill_shared`（两组卷积、光Router）、`routerfill_shared_balance`（同结构、更强均衡正则）和
`qwen_shared`（完整冻结Qwen、相同读出头）。旧lean已经完成100轮，修改格89.40%，但语言Router固定使用专家0/1，
不满足专家均衡要求。新旧run不混用；详见[新修复及公平对照合同](reports/reproduction/ROUTER_SHARED_HEAD.md)。

```bash
# 主模型 / 相同架构的强均衡候选：分别在可用GPU上运行
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile routerfill_shared --phase all --device cuda
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile routerfill_shared_balance --phase all --device cuda
# 完整冻结Qwen baseline：自动提取原生特征，再用同一个训练器训练共享结构的读出头
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile qwen_shared --phase all --device cuda
```

三个profile都是5000train/1000test、同一v2清单、100epoch、seed73。后端完全复用 `SharedGridReadout`，
参数量381976、初始权重逐位一致（训练后各自独立）。主模型仍无TF/attention；**完整Qwen仅作为baseline执行**。
baseline只增加必要的1024/2048→192线性适配和LayerNorm，无LoRA、额外卷积支路或额外任务prompt。
本轮只测性能；不把训练服务器的耗时/功耗充作5090D实测。

## 当前优化入口：2026-09-08 embedding-only / alpha > 0.4

新 profile 为 `embedding_alpha40`（光 Router Top-2）、`embedding_alpha40_lean`（末端卷积由3组减为2组）和
`embedding_d2nn_alpha40`（普通 D2NN 对照）。**下文98%的历史结果不属于这些新模型。**
新模型文本只做冻结 Qwen 词表查表和正弦位置编码，不使用语言 TF 缓存；视觉只用冻结 Qwen patch/位置前端。
两个模态都各有光 Router → expert → global（D2NN 对照没有 Router）；四个光电融合 alpha 均约束在
`[0.4001, 0.95]`，初始0.60。语言 global 后采用固定位置系数的可学习线性汇总，不使用 mean/max 拼接或 attention。
新模型随机初始化任务网络，不继承旧语言TF教师或旧任务头。每个光阶段只保留一条电子残差和一条光支路；
末端仍有指令条件卷积和类别/编辑双输出头。详见[新架构及运行说明](reports/reproduction/EMBEDDING_ALPHA40.md)。

运行（从本仓库根目录，先 prepare 一次，再按可用显卡分别启动）：

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile embedding_alpha40 --phase prepare --device cpu
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile embedding_alpha40 --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile embedding_alpha40_lean --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile embedding_d2nn_alpha40 --phase all
```

下面是保留供核对的旧协议。

输入是 `224×224` OpenMoji 场景和文本指令，指令任务为 `add / replace / move / remove`；输出是 `6×6` 类别网格和编辑网格，再由固定 OpenMoji 合成器得到目标图像。

论文口径、baseline差异和复现命令统一见 [reports/reproduction/README.md](reports/reproduction/README.md)。
本任务是自行构造的封闭词表网格编辑，不是OpenMoji官方benchmark。当前文本缓存经过完整冻结语言Transformer，
最终保留合成使用真实`source_grid`，不能描述为纯embedding前端、完全无Transformer的RGB+文本端到端系统。
2026-09-08审计发现1对train/test完全重复输入；历史数值保留，后续去重必须使用新split/profile。

**2026-09-08三组完整复评已完成**：[全部指标、去光、相位/冗余审计与通俗数据流说明](reports/reproduction/RESULTS_20260908.md)。
本次修改格准确率：MoE0.9800、D2NN0.9890、Qwen在线0.5390；旧数值仍保留于下文，不覆盖历史。
MoE去光后为0.9850，但整场景从0.8950降至0.8760；四个alpha约5%。
旧版语言TF缓存不满足当前禁TF要求，且CCD归一化含clip/log；这些是待修正旧架构成绩，不是合规新版本成绩。

本目录固定比较三组系统：

1. `main_dc20`：语言、视觉各含光学 Router，均为 4 专家 Top-2；随后各有一张 global phase。融合前做 RMS 同尺度归一化，训练含 20%–30% 相干零级分量与硬/软专家均衡。
2. `d2nn_dc20`：没有 Router；语言和视觉各使用两层普通 D2NN。四张 `224×224` dense 相位与主方法两个模态实际激活的四张专家相位参数量相等。
3. `qwen_pending`：历史待测合同入口，不执行大模型评估。实际冻结Qwen加训练任务头baseline由`baseline_structured_5090d.py`运行；历史结果见下文，复现缺口见复现说明。

主指标为 changed-cell accuracy，并同时报告 foreground category accuracy、edit-grid IoU、object F1、scene exact match 和按四种操作分组的指标。

## 数据协议

- train：5,000 个合成场景，每个操作 1,250 个。
- test：1,000 个不同随机种子的场景，每个操作 250 个。
- train/test 同分布、样本种子不相交；validation 为无。
- epoch 1、每 5 epoch、末轮测试；按最高 test changed-cell accuracy 选择 checkpoint。
- 正式目录只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。

## 运行

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile main_dc20 --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile d2nn_dc20 --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile qwen_pending --phase all
```

大模型 baseline、零样本生成诊断、计时和功耗口径见
[BASELINE_5090D_TODO.md](BASELINE_5090D_TODO.md)。正式 baseline 冻结 Qwen 原生
Vision/Language Transformer，只训练普通结构化任务读出头；旧的自由生成 JSON 结果只作为
zero-shot diagnostic，不写入论文 baseline 行。

RTX 5090 D 正常 baseline 已完成：输入 224×224 图像和完整指令，完整执行冻结的原生
Vision/Language blocks，只训练 1,212,434 参数的结构化任务头。5000 train 训练 50 epoch，
每 5 epoch 测一次完整 1000 test，并按 changed-cell accuracy 选择 epoch 20。正式结果为
changed-cell **0.5475**、foreground category **0.2168**、edit IoU **0.2909**、object F1
**0.1666**、scene exact **0.0160**；第一个 Vision block 到 `6×6` 两个输出的
mean/median/P95 为 **27.166/26.628/30.280 ms/sample**。四任务 changed-cell 分别为
add 0.224、replace 0.336、move 0.650、remove 0.980。该结果没有 LoRA、没有主干微调、
没有自回归生成，也没有为抬数值加入额外 loss 或增强。

## 正式单次结果（seed 73）

- 光 Router Top-2：selected checkpoint 正式复评 changed-cell accuracy 0.9800、
  foreground category 0.9895、edit IoU 0.9350、object F1 0.9837、scene exact match 0.8950。
- 参数匹配 D2NN：changed-cell accuracy 0.9895、foreground category 0.9944、
  edit IoU 0.9813、object F1 0.9949、scene exact match 0.9650。
- 主方法语言 Router 使用 3/4 专家，选择占比 50.00% / 25.00% / 25.00% / 0%；
  视觉 Router 也使用 3/4，选择占比 20.30% / 45.80% / 33.90% / 0%。

因此该结果满足“物理光 Router、Top-2”的结构要求，但不能宣称四专家完全均衡；D2NN 的
主指标高 0.95 个百分点。机器可读结果和可视化见 `reports/dc20_comparison/`；其Qwen pending行为早期快照，不代表后来的baseline未运行。
# 2026-09-15 SHS六层小样本实采状态

师弟电脑`E:\code\guest\2026OpticsMoE\OpenMoji_Lab_SHS_8um`的`pilot01`已完成4条×6层=24张CCD及末端推理。
六层相位前后物理对照通过；关键控制修正是换相位等待期间持续清理相机过渡帧，保留独立相位SDK进程。
4条分别为add/replace/move/remove：修改格准确率0.75，整场景0.25；仅流程验证，不是1000test性能。
固定权重1000test仿真在4060复评为0.8700/0.6880（原归档0.8715/0.6900）。
语言Expert/Global实采p99约9～10/255，保留弱信号风险。随后用户明确要求立即运行完整测试，
已尝试`test1000_01`和重连后的`test1000_02`：1000条×6层，沿用400μs/Gain_X4/240ms振幅等待。
两次均在启动探针得到近黑图（100μs全白ROI p99=6，之前相同探针约83），自动停止，正式CCD均为0张。
独立400μs诊断的整个传感器p99也仅7；用户确认激光仍开。随后GUI相位保持下，相机SDK和振幅SDK黑/白/棋盘格对照正常。
已切换为用户Blink GUI手动保持相位、只由SDK控制振幅和相机。`test1000_02`六层各完成1000/1000并通过SHA/输入链审计；
全部后续输入由上游真实CCD生成，最终1000条推理已完成（结果见本页开头）。正式曝光/ROI/权重未改，失败启动检查作为历史证据保留。
GUI全量与SDK试采首个相同输入CCD的PCC约0.654，仅作为控制方式/时间差异诊断，不宣称两次物理响应相同；数据不混用。
证据统一在`runs/hardware/shs_single_pc_20260915/00_查看这里.md`及其`sessions/pilot01/results.json`。
部署控制源码`1917cda3`，含说明与传输回归测试的完整覆盖包`6ff44e8b`，已同步GitHub。
