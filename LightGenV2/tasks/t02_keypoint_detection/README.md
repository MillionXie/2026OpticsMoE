# T02 关键点检测（LSP）

## 2026-10-08 最新授权：修正物理幅度后在实验室GPU重训，实拍须再次确认

2026-10-09用户已明确授权将当前73.5929%最佳部署光路，不再等待再次确认。
固定新物理合同e15 EMA，SHA b525e613a876b6a4c2543e6433a8bef52d07a8101281c29448c50ea02151bb3e；
本地续训继续至60，不把未完成训练称完训，不在采集中切换后续best。
新增lab_bounded_release只读复用旧TEST1000冻结stem缓存（不复用CCD或旧仿真热图），
精确新PT/settings/phase身份和全1000理想CCD回灌通过后，实验室pilot4再完整三层1000。
新release/session lsp_bounded_e15_20261009，tanh05_uint8在传播前执行，幅度scale1、
CCD除255回灌，不再乘16/256；2000us/GainX4/wait240/原ROI方向不变，暗帧即停。
旧legacy默认合同不变；新合同传播/回灌及原幅度测试11项通过。新光路结果尚未产生。
实际任务LSP_BoundedE15_BridgePilotFull_1009已派发，PT到实验台后SHA再次一致，
全1000缓存SHA通过并实际理想回灌850/1000有增量；不是仅TaskRunning。
任务按回灌成功→pilot4成功→full1000严格退出码串联，失败即停、不重启旧采集。
补充恢复与幅度回灌回归18项通过；源码be73faca1已三端同步。
全1000回灌现已通过（热图maxabs 1.1921e-6），pilot12 CCD/24.034秒完成并释放SDK。
同任务已进入full1000，router13/1000、共21有效CCD（含pilot），p99=22/sat0。
新真实PCK尚未产生；全部完成后必须逐层1000收据审计和原指标回放，不以桥接替代实拍。

00:35补充显式`--continue-from-run`：只允许未完成的bounded_staged，同profile/seed/
batch/workers/资产路径/初始SHA/物理幅度合同及完整原划分，检查history与live-last epoch，
严格加载原架构。新run保留旧best与history并从last下一轮继续原60轮日程，
原run不写不覆盖；manifest记录父run及全部SHA、optimizer/EMA/RNG重置，明确非精确恢复。
不多跑额外60轮、不拿旧e15PCK冒称续训结果；13项日程/幅度/恢复拒绝条件测试通过。
计划在本地4070继续e21–60，独立持久计划任务避免依赖原临时执行session；
仍待原PT实载、实际新epoch/loss确认，用户确认完整新仿真精度前不部署。
上述续训现已实际启动：持久任务`LSP_Bounded_Continue21_Local4070_1009`，Python39864，
新run `lsp_bounded_staged_continue21_local4070_20261009`，已记录权重续训身份并真实
epoch21 batch100/869、loss .11370，4070约2346MiB；不是仅PID/任务Running。
三端已同步训练源码cbfba4c40，原run与e15 best不变，续训尚无新周期TEST指标。

2026-10-09 00:15核查发现本地正式进程已消失、4070降至53MiB，日志停在完整epoch20，
没有final_report或Traceback；不能称60轮完成或仍健康运行。e15开发PCK .7359285714、
e20 .7324285714，best/last及history保存至e20，原始资产/划分保留。
原执行session75115已失效，退出原因尚无证据确定；现有refine入口无resume参数，
不能盲从e0重启或把权重续训冒称含optimizer/EMA/RNG的精确恢复。先保全并诊断恢复合同，
用户确认新完整仿真结果前不部署光路。

本地资产已全部下载并再次读盘SHA通过：005e1ac073132763a38331252f5b72ed7d1eb450bbd1169c37156335e8247a58。
包内11456成员路径/类型检查通过后解包；原PT495b9c2c...28518亦核验一致。
Git9c918fe5b复用现有便携PT读取器，修复Linux PosixPath在Windows反序列化失败；
缓存显式使用Windows扩展长路径，避免preprocessor_config.json因路径长度被误判缺失。
12项幅度/分阶段单元测试及4样本真实梯度smoke通过，正式本地run
`runs/simulation/lsp_bounded_staged_local4070_20261008`已运行，60轮/fullTRAIN10428/
TEST1000/batch12/workers0；训练前新合同PCK .7347142857142858，首轮.7325714，
最新第5轮进行中，不能据初始/早期值称性能改善或正式完成。前4轮约3.7–5.4分钟/轮；
后续光学联合训练阶段可能更慢，不据头部校准阶段保证全程ETA。用户确认新精度前不实拍。

22:15之后用户授权比较并采用更快的本地训练路径。本地RTX4070 Laptop 8GB空闲，
既有`qwen3vl-cifar10`环境使用`python -s`已通过Torch2.11/cu128 CUDA矩阵运算；
默认base环境会误载用户级Torch并报c10.dll错误，不能用于本次训练。
本地数据目录尚无实际TRAIN图片，故仍需同一7,156,828,160字节资产包；服务器到本地
8MiB短测10.0MiB/s，持续下载约8–9MiB/s，预计约12–14分钟，而实验室转发约.37MiB/s。
已实际启动下载至本任务既有`runs/smoke/lsp_bounded_asset_transfer_20261008`，完整SHA
通过后才准备数据。原.7348初始PT已取回，须核SHA；不替换模型为本地已有的Instruct版本。
旧实验室传输残件/进度保留，不为切换执行地点删除数据；正式训练尚未开始。
下一步本地完整依赖、原PT/数据身份、幅度/梯度及小批量显存短测通过后，再运行
同`bounded_staged`配置；先报告新仿真PCK，仍须用户确认后才能部署光路。

本段覆盖下方旧采集进度；首次和独立重复各3000CCD已完成，均保留，不能复活旧任务。
用户先看新仿真PCK再决定部署，本轮不采光路。实验室RTX4060约7.4GB空闲显存；
服务器GPU为其他用户占用，不抢占。实验室原TEST缓存不能替代缺少的TRAIN资产。
新`bounded_staged` profile从精确.7347857 EMA初始化，60轮原分阶段训练：
三层传播输入保零`tanh(abs(E)/.5)*exp(i angle(E))`及round255量化，TRAIN量化STE；
不做旧x16/x256单位回乘。模型拓扑/参数预算保持，但物理计算合同及checkpoint标签改变，
不是旧PT的等价单位换算，也不是相机响应已标定。原CCD读出/融合暂保留，仍需全模型
理想CCD回灌、梯度及小批量显存验证，不能仅以6项幅度测试称部署通过。
源码7a6411a0a已发布并三端同步；服务器新源码幅度/分阶段测试12/12通过。
必要TRAIN图片/标注和冻结前端模型已打包约6.7GiB，私有SHA传输到既有任务
`runs/smoke/lsp_bounded_asset_transfer_20261008`；归档不含源码/凭据/个人照片或旧采集。
原串行传输首测约.57MiB/s，已停止该唯一传输并保留未校验残件，改32请求预取窗口；
新包全量SHA通过后统一准备数据并训练，当前尚无新训练PCK或正式梯度run。

## 2026-10-08 授权实拍：已确认版本，部署合同检查中

首次完整三层实拍已完成，各1000PNG/收据，3000份CCD/BMP/phase/upstream SHA
审计bad0，任务返回0、SDK释放。原TEST1000/14000关节回放PCK .7353571428571428，
同权重原仿真 .7347857142857143；净多8个阈值命中，不称稳定提升，像素误差略增。
已核对用户旧表显示.7353与绑定原报告不一致，用户现确认原仿真.7348。
用户授权一次独立重复：`repeat1000`，任务`LSP_Staged7348_Repeat1000_1008`，
同精确PT、同完整TEST和光学合同，不复用首次任何CCD/pilot，不改变曝光或挑子集。
仅复用冻结stem缓存（输入身份不变），全部输入已到实验室，无传输等待；三层依序
各1000帧。首次结果与重复结果都保留，不以更低重复分数替换首次结果。

用户所指旧表0.7353对应`refinement_20260909/staged_heatmap`第50轮EMA；
原报告精确PCK@0.2为.7347857142857143，不能将展示值0.7353冒称精确证据。
权重SHA495b9c2c4e3df15d3715f1ce8f2faea7cb9156275b31103ec684f4e96a328518。
服务器原PT与本地取回PT均核验相同SHA；原数据、报告和权重没有修改。

新run `runs/hardware/lsp_staged7348_20261008/simulation_reference`在main严格加载
同PT，完整官方TEST1000/14000关节复测PCK .7347857142857143、PCKh .8524285714；
此项是仿真复载，不是实拍。与旧报告PCK完全一致，NME等浮点值另保留新报告。
无重新训练或新增推理层，数据仍为原人体标注框裁剪，不是整图多人检测。

部署前入口`lab_preflight.py`只挂无参数观测hook，沿用原模型/评估器/TEST及PCK定义，
不打开SDK、不改光场或权重。新`contract_audit`完整1000图复测同PCK通过，
`amplitude_audit.json`确认router/expert/global原输入幅度最大
14.0494499/13.5001602/9.7189484，均超出单位区间；router记录含518×518 FFT画布，
expert/global为478×478有效场。不能直接套用0–1的OpenMoji BMP编码，也不能
只在导出端截断或做仿真中没有的逐图peak归一化。

幅度超1的原因是Softplus非负投影及RMS能量归一化不限制峰值，不是权重损坏。
新增`lab_field_units.py`以固定单位比例s=16处理三层：播放幅度a=A/s，
回放物理强度换回原模型单位为I_model=s² I_physical。此为数值单位换算；
真实相机计数到物理强度的响应仍需pilot验证，不能直接把raw CCD计数乘256。
不逐图peak归一化、不截断、不修改相位/alpha/PT、不增加网络层；超出固定范围即停。
三层传播、量化零支持/界限、越界拒绝及理想强度回放单元测试4/4通过。
完整原TEST1000连续换算PCK .7347857142857143，与原值一致；8-bit量化
PCK .7343571428571428，仅下降.042857个百分点。三层物理幅度最大
.8784314/.8392158/.6078432（uint8），均在0–1内。两份amplitude_audit报告
已本地备份，PT SHA不变。这些仍是仿真，不是实拍结果。
`lab_preflight.py --physical-mode uint8 --ideal-replay-check`新增全链三层理想
CCD回放的热图一致性检查；独立`ideal_ccd_replay`完整42批/1000图通过，
三层理想强度回灌后最终热图最大绝对误差为0，PCK仍为.7343571428571428。
这是理想CCD数值桥接通过，尚不能证明真实CCD响应或光路部署正确。

后续必须先验证三层相位和CCD回放，再pilot后
全量官方TEST1000×三层=3000实际CCD。实测上游CCD须驱动后续输入，不混理想场。
2026-10-08已实际完成`pilot4_retry2`：4样本×3层=12 CCD，全部PNG/收据/相位SHA
核验通过，最低p99=22、最大饱和比例0，采集24.793秒，任务返回0且SDK释放。
这是信号和逐层衔接pilot，不是完整TEST的实拍PCK。原未打开SDK的PosixPath及
Git路径兼容失败日志保留，未覆盖有效CCD。
`lab_acquire.py`导出1000个冻结Qwen stem token缓存；同批量缓存模型热图合同通过，
单样本相对batch24的float32差异逐项记录。原光电core/head全量strict加载，
相位回消必须还原非负实幅度，三层理想CCD回放通过才允许设备初始化。
现有实验室主仓库通过已发布main的Git bundle接入该任务；未改旧OpenMoji/ABO目录。
一次性任务`LSP_Staged7348_Full1000_1008`已实际进入全量采集，最新主PID11192/
launcher11776、router60/1000，总68PNG（含其他两层各4 pilot），最新p99=22/sat0。
当前进度以`full1000/progress.json`和实际CCD/收据数量mtime为准，不仅看任务State。
原大包上传被Windows独占锁挡住；已保留残件、失败日志并停止原大包上传。
改为私有逐样本artifact-only发布器，输入每个成员在本机和远端SHA通过后原子发布；
采集再次校验后播放，允许边传边采。每个缺帧等待有10分钟无增量上限。
发布完成由`stem_release_retry2/external_transfer.json`登记，全部1000成员SHA终审。
不再等待大包整体传完才开始；原打包SHA仍保留为来源：
916f929ccdc01f1811e9f3225b7c05db08e9cd4ada1644426a23068759706e19和所有成员SHA，
按router→expert→global各完整1000样本继续。交接最多2小时，异常即停止。
同权重/幅度/曝光/几何的12 pilot CCD保留复用；每帧须重新匹配BMP/phase/upstream SHA，
因此目标3000有效CCD、2988新增采集。2000us/GainX4/wait240、hv反相phase、flip_v CCD，
固定1/255 CCD回灌，不做逐图亮度拉伸；暗p99<15和原严格1%饱和守卫均停。
全量完成还需原PCK定义的物理回放评估，当前没有全量实拍PCK。OpenMoji离线适配
不会占用SDK；二者不能共用数据或篡改对方run。其他窗口任务和原光学合同保持。
本轮不新建分支/worktree/工程副本，源码仅Git同步；权重/缓存走SHA清单。

2026-10-03整理核验：[完整版本地图](reports/reproduction/IDENTITIES_20261003.md)。
下文DC20为早期公平对照，不代表所有后续候选：另保留73.48%低alpha交付、72.83%高alpha蒸馏，
以及独立个人照片pilot。对应PT/划分已查证；后续入口按服务器固定源码收敛到main。
历史文档中的旧分支/目录是运行记录，不再照此自动新建；权重必须匹配各自架构和配置，不能直接换PT。

本任务只处理视觉分支，输出 LSP 的 14 张关键点热图。冻结的
Qwen3-VL-Embedding-2B 仅执行 patch embedding；原生 Vision Transformer
block 不执行。两种正式方法共享数据划分、电子 mixer、同尺度融合、姿态读出头、
10 cm/17 μm 传播、k 空间约束、位移/增益/偏置/读噪声和 20%–30% 相干 0 级分量。

上述说明只针对两种光学仿真方法。论文中的大模型 baseline 是另一条直接路径：输入
224×224 图像后完整执行冻结 Qwen 的所有原生 Vision Transformer blocks，取最后一层
原生视觉 token 恢复为 14×14 空间图，再接 `lsp_pose_opt2.yaml` 已训练的
`DeconvPoseHead`（14→28→56 两级可学习反卷积），输出 14 张 56×56 关键点热图。
它不是只拿 patch embedding，也没有绕过 Qwen Vision。

该直接大模型 baseline 已在 RTX 5090 D 上用完整 1000 张官方 test 重测：PCK@0.2
**0.7217**、PCKh@0.5 **0.8846**、NME **0.2084**。从第一个原生 Vision block 到
14 张 56×56 热图的 mean/median/P95 为 **9.504/9.470/9.632 ms/image**；读出头为
`DeconvPoseHead`，1,102,990 个可训练参数，Qwen 参数全部冻结。
2026-10-06原归档核对：性能评估1000张、batch8；计时另取200张、batch1，
显式预热50次。这不是无预热全量计时，也不是完整二维关键点坐标解码的延迟。
原run目录在已查服务器路径缺失，但纠正归档及原头PT完整保留，头SHA与报告相符；
来源、逐成员SHA及只读检查见
[历史测速绑定](../../../maintenance/storage/T02_BASELINE_TIMING_BINDING_20261006.json)。
不重测、不改原CSV/PT/报告，也不把此旧baseline的速度套给后续光电候选。

## 两个正式 profile

- `main_dc20`：一次光 Router CCD 把样本送给 Top-2/4 个 224×224 专家；
  随后还有一层 478×478 global phase，视觉分支共两次 feature CCD。
- `d2nn_dc20`：无 Router、无空间专家布局；使用两张 224×224 dense phase，
  中间保留 CCD 归一化和电子重载。每张图激活的相位参数量严格等于主方法
  Top-2 专家，即 `2×224²=100,352`。

两支路先分别按每个样本的有效 token/channel RMS 同尺度化，再用
`(1-alpha)E + alpha O` 融合并恢复电子支路 RMS，避免电子数值范围淹没光支路。
训练使用周期性 test（epoch 1、每 5 epoch、最终 epoch），按 PCK@0.2 最大值
选择 best；这是明确的数据使用协议，不把 test 描述成独立封存测试。

## 服务器命令

从仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile main_dc20 --phase all

CUDA_VISIBLE_DEVICES=1 python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile d2nn_dc20 --phase all
```

仅关闭输入、相位与 CCD 三类像素平移扰动，同时保留光 Router、DC20、强度噪声、
phase dropout、k-space 和同尺度融合的 100-epoch 定位误差消融：

```bash
CUDA_VISIBLE_DEVICES=0 python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile main_dc20_no_shift --phase all
```

若从随机公共初始化训练的无位移版本仍明显低于历史模型，可运行兼容 warm start。
它复用旧 0.7131 模型中形状完全一致的二维 mixer、姿态头及 feature/global phase，
但不会加载旧电子 gate；光 Router 相位始终重新初始化并训练，物理光路和 DC20 条件不变：

```bash
CUDA_VISIBLE_DEVICES=1 python -m LightGenV2.tasks.t02_keypoint_detection.run \
  --profile main_dc20_no_shift_warmstart --phase all
```

每个正式 run 只保留：

- `best_checkpoint.pt`：周期性 test PCK@0.2 最佳的 EMA 权重；
- `last_checkpoint.pt`：最终 epoch 的 live 权重；
- `metrics/training_history.csv`：完整曲线；
- `selected_checkpoint_test_evaluation.json` 和逐样本预测；
- `best_visualization/`：由 best 权重直接绘制的相位图与统计。

不要在正式 run 内保存每 5 epoch 的 PT。若以后研究相位演化，应另建明确标为
analysis 的 run。

训练完成后生成汇总表和论文图：

```bash
python -m LightGenV2.tasks.t02_keypoint_detection.report \
  --main LightGenV2/tasks/t02_keypoint_detection/runs/simulation/moe_router_scale_dc20_seed42 \
  --d2nn LightGenV2/tasks/t02_keypoint_detection/runs/simulation/d2nn_matched_dc20_seed42
```

## 正式结果

100 epoch、seed 42 的 DC20 单次复跑已经完成。两组都在 epoch 100 取得最高
周期 test PCK@0.2：光 Router Top-2 主方法为 **57.73%**（PCKh 73.63%，NME
0.3488），参数匹配的普通 D2NN 为 **67.51%**（PCKh 80.54%，NME 0.2736）。

历史目录中还能看到 PCK 0.6024 和 0.7131；它们分别缺少当前 DC20 条件，或使用旧
16 µm/电子 gate/非同尺度融合架构，不能替代当前正式结果。旧 0.7131 运行实际同样是
478×478、4 专家 Top-2，其父工程的 `moe16` 名称不代表运行时架构。完整核查见
[`reports/LSP_METRIC_AUDIT.md`](reports/LSP_METRIC_AUDIT.md)。
因此本次 LSP 协议下，普通 D2NN 明确优于光 Router MoE 9.78 个百分点；不能把
历史不同协议的 71.3% 候选混入本表。结果表、论文图和最佳相位总览见
[`reports/dc20_comparison/RESULTS.md`](reports/dc20_comparison/RESULTS.md)。
