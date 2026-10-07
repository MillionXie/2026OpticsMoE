# T06 视频质量评价

**2026-10-04整理覆盖：**下文保留历史仿真、baseline和测速协议，不代表所有默认命令已适配。
正式Spatial/Temporal实拍版本、各自权重及已发布核心代码见
[当前版本与待收敛边界](CURRENT_VERSION_20261004.md)。主线核心通过57项CPU合同测试、
两份正式PT严格加载；旧Temporal-36默认profile和后续实拍微调入口仍须单独核验。

## 先选版本：当前实拍模型与历史默认不是同一套

| 当前登记版本 | 原固定权重仿真 SRCC | 未适配实拍 SRCC | 权重身份 |
| --- | ---: | ---: | --- |
| Spatial，单视频4帧，六层原1M读出 | .6710968960 | .5786364901 | 95e12397…592828b |
| Temporal，16视频×4帧，同场六层 | .8043868643 | .7977138739 | 5303b574…a243a77c |

精确完整SHA、源数据、原报告、离线适配与只读权重验收命令均从上方
`CURRENT_VERSION_20261004.md`进入。两行不是同一模型；旧测速不套到这两份PT。
以下Temporal-36、旧Spatial-4、9×4等段落保留历史成绩和对照。
**未指定profile的旧默认仍是Temporal-36，不是当前Temporal实拍模型，且预检未通过。**
本轮只澄清入口，不静默改默认profile、原SHA、资产路径或光学合同。

2026-10-06补齐主线测试依赖：`lab_stage_coordinator.desktop_code` 仅返回历史阶段脚本
字符串，不连接机器、不打开SDK、不启动采集；T03测试不再依赖未跟踪的
`lab_manual_stage.py`。同时恢复历史 `stage_config` / `effective_stage_identity` 纯配置
函数及原测试，既有 `lab_bench` 函数未改。来源见
[函数投影身份](stage_coordinator_source_identity_20261006.json)。这不是迁入完整旧控制器，
也不是宣称现有采集入口已经应用逐层曝光；未改变硬件合同或正式模型。
进一步检查主线483份任务Python的2294个内部导入名，发现T03采集入口还缺
`capture_staged`；现从同一历史提交原函数恢复，T03导入及4项采集前置拒绝测试通过。
这些测试不接设备、不写实际会话，原T06的 `capture` 函数未改；不是现场采集验收。


## 历史仿真结论（不替代上方当前实拍版本）

旧独立审阅包在 `LightGenPublic/tasks/t06_lgvq_temporal_consistency`，不是另一个
日常开发工程。原 `teacher_release_final/lgvq_temporal_08044` 使用35个固定field、
558条视频及PT SHA `5303b574b200e14bf943af21c60a246720be453b9cf8847cd93c0eaaa243a77c`，
原仿真SRCC .8043868643。其 `teacher_release_final_v2` 改了8µm传播网格与探测器扰动，
记录 .8022806420，不能按同名PT/继承commit冒称与原版等价。二者仍保留完整原目录、
固定输入与权重，未做新性能评估；训练缓存缺项和重训练边界继续按原说明保留。
版本身份见 [审阅包审计](../../../maintenance/storage/T06_REVIEW_PACKAGE_IDENTITY_20261006.json)，
80份具名包内源码的Git恢复身份见
[历史源码副本](../../../maintenance/storage/T06_REVIEW_SOURCE_VISIBILITY_20261007.json)。
忽略副本不删除文件，不改变封存包的SHA清单；后续开发仍从本任务入口进行。

三个旧审阅包的同一份原依赖说明已集中到
[历史审阅包依赖](reference/temporal_review_requirements_20260922.txt)，原字节SHA为
`10300b8bf6c39ceeebb6345e46075a9090a588cb6305523eae426bd7ae4b9f9c`。
这只记录当时numpy/PyYAML/torch版本，不安装软件、不替换当前环境或正式模型合同；
三份原文件保留，退出Git待提交列表，不再维护重复依赖说明。

误命名的`tasks/06_video_quality_assessment`仅含14份MobileNet历史报告，已原字节归入
[MobileNet旧对照](reports/paper_results/spatial_mobilenetv2_b11_20260910/RESULT.md)。
其33级配置继承链缺失的五份旧配置已按原字段纳入兼容后端；
[配置来源和本地／服务器注释差异](reports/paper_results/spatial_mobilenetv2_b11_20260910/CONFIG_SOURCE.md)
单独说明。原报告、PT和测速不变，没有重训或复测，也不将此历史候选改为默认。
原报告与配置快照纳入main，其余原结果仍保留在该规范任务目录；14份原字节
均可从Git恢复提交`a96c1fc4d7bd6062002cb73955cbc0433229326c`的旧路径恢复。
旧目录已无报告文件，空目录未删除。
这是2026-09-10的SRCC .665159、去光 .561838、PT SHA 5e5020e3…55c4459历史对照，
不是当前Spatial .6710968960/PT 95e12397，也没有补测实拍或套用旧测速。
原s745配置继承及原字段加载已复核；数据/缓存/PT等资产闭包未完整复核，
原配置快照保留出处，不当作已通过新机训练或实拍验收的profile。

历史 Spatial-4 对照归档是 `spatial_single_video4_balanced`：一条视频均匀取 4 帧并排成
2×2，**没有多视频复用**。它使用两套物理光 Router Top-2、六次光传播和 20% 名义
未调制分量，在 558 条 test 视频上达到 SRCC 0.6393、KRCC 0.4642、PLCC 0.6743、
RMSE 8.452、MAE 6.646。入口、checkpoint SHA 和光关闭对照见
[`reports/paper_results/spatial_single_video4_balanced`](reports/paper_results/spatial_single_video4_balanced/README.md)。

历史 Temporal-36 对照版本为 `temporal36_balanced`：一个视频均匀取 36 帧，以 6×6 lane 放进同一个
478×478 有效光场。四专家光学 Top-2 router、六次光传播、20% 名义未调制直流分量、
鲁棒位移/相位/CCD 扰动和目标专属电子读出头保持不变。

正式测试集 558 个视频：SRCC 0.8454、KRCC 0.6394、PLCC 0.8650、RMSE 7.183、
MAE 5.451。平衡候选四专家占比正常，结果详见
[`reports/paper_results/temporal36_balanced`](reports/paper_results/temporal36_balanced/README.md)。

历史正式仿真候选 `temporal_multivideo9x4` 把 9 条互不相关的视频各取 4 帧，以 3×3 视频
tile、每 tile 内 2×2 帧的方式放入同一 478×478 光场。它不是把 Temporal-36 checkpoint
改名，而是新的六次全场相干传播模型，输出形状为 `[B,9]`；其中每个数仍只评价一条视频。
正式单 seed 结果为 SRCC 0.8082、KRCC 0.5999、PLCC 0.8131、RMSE 8.226、MAE 6.109；
视频路由最大专家占比 28.0%，跨样本选择变化率 55.7%，九槽位循环审计平均 SRCC 0.8019。
完整证据见
[`reports/paper_results/temporal_multivideo9x4_contentroute`](reports/paper_results/temporal_multivideo9x4_contentroute/README.md)。
Temporal-36 保留为“单视频 36 帧”的独立基线，二者不能混报。

16×4 吞吐优先模型 `temporal_multivideo16x4` 在同一光场同时处理 16 条视频、每条 4 帧，
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

2026-10-06预检修正：`check_environment --task t06` 仍检查历史Temporal-36，
但不再仅因Torch可用就返回成功。后端/config/PT身份及已列输入路径有问题时返回2，
错误以JSON报告；`--profile <旧兼容profile名称>` 可显式选择，不自动换成另一架构。
本机实查默认PT和缓存缺失，五份后端SHA不匹配，因此当前默认预检失败；
这不是宣布文件被删除，也不是允许更新SHA绕过检查。检查范围不含缓存语义或SDK。
正式Spatial/Temporal实拍版本仍从上方 `CURRENT_VERSION_20261004.md` 查精确入口，
不据旧默认预检或下方历史命令称它们已在本机可运行。

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
  --profile spatial_single_video4_balanced `
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

2026-10-07补齐既有SHS打包器引用的`lab_phase.py`和`hardware/run_lab.py`，
不再依赖本地未跟踪源码。两份服务器原Spatial/Temporal交付ZIP中的对应成员已只读核验；
portable入口原样保留，相位保持器保留后续已有的可选`--release-file`退出控制，
未指定时与原包行为相同。不会自动切层或采CCD，本次未打开设备或重建旧包。
来源和差异见[打包依赖身份](package_dependency_source_import_20261007.json)。
这仅关闭两份源码缺项，不代表旧Temporal默认资产或机器SDK已完整。

同日恢复正式SHS打包分派：显式使用 `--bench shs --target spatial|temporal
--source-root <原资产根目录> --output <新的输出目录>`，调用既有 `lab_bundle`；
不传 `--bench` 仍是旧legacy流程。SHS不接受 `--checkpoint` 静默替换封存权重，
已有输出目录或相邻ZIP在加载模型前拒绝。参数分派及输出保护测试不加载PT/SDK，
本次没有构建新包或覆盖旧包，不代表完整现场复现通过。

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

## 历史 A100 批量与 dataset-once baseline 已收敛（2026-10-06）

### 可选历史 Temporal 读出压缩工具（2026-10-07）

旧服务器`t06_readout_compress_20260912` h512试验目录已完整归档退出日常工作目录：
3577份原文件、13处数据链接身份及全部旧测速保留，四份早期冒烟元数据亦在完整归档中。
原数据／PT链接目标不删，正式Spatial/Temporal及另一个compress_all原结果目录未变。
恢复路径、完整SHA及原服务器Git引用见仓库
[目录恢复收据](../../../maintenance/storage/SERVER_WORKTREE_BALANCE_20261007.json)
的`latest_t06_readout_trial_retirement_20261007`；不是新的模型成绩或本地独立数据备份。

`compress_temporal_readout` 与三份原 h256/h512/h640 配置已从本地保护历史
`ac7e3ecc` 纳入主线，保留独有代码，不把它当作服务器正式运行版本。
核验时训练服务器主目录没有这四份文件，也没有该原历史对象；来源明确记录在
[历史工具身份](temporal_compression_import_20261007.json)。只用于16视频×4帧六层
光电之后的原电子读出头缩减，不能把其配置名或训练缓存成绩替代封存 .8044 PT。
本轮未找到并绑定压缩候选正式成绩，未提取缓存、训练、重评视频或使用GPU/设备。

原九项函数的AST保持一致，仅 extract/train 的首行增加已有输出拒绝；缓存文件、
伴随JSON或训练目录存在时，在读取配置/PT、初始化RNG及CUDA之前失败，保护原结果。
三份配置继承既有formal，原256/512/640宽度及种子不改，默认为可选历史profile，
不修改主入口默认Temporal-36或最终16×4模型。原工具按TEST SRCC选模，是开发指标，
只保存压缩best；它不是新的已验收正式训练协议，不补造历史last或独立泛化结论。
入口帮助：`python -m LightGenV2.tasks.t06_video_quality_assessment.compress_temporal_readout --help`。
9项合成CPU/AST/配置/防覆盖检查通过，不等于真实缓存或完整模型复现。
实际训练服务器main `3a130341` 同样通过这9项检查，CUDA设备隐藏，未读取正式PT/数据
或调用设备；来源收据保留原本地历史身份，不因发布后存在而改称此前服务器运行版。

同一保护历史的八份`temporal_multivideo16x4_readout_rank{R}_s{S}.yaml`亦归主线：
R/S依次为512/173、384/174、320/175、256/176、192/177、128/178、64/179、48/180。
它们继承同一formal并仅声明原末端读出矩阵低秩，供既有压缩工具显式选择；
不加电子支路、不改变六层上游，不修改正式默认或 .8044 PT。服务器此前没有这八份
文件，来源是本地独有历史，不冒称已在服务器验证过训练成绩。原始配置SHA见上方
历史工具身份，八项配置加载/几何/原SHA检查只证明配置合同，未跑真实数据或新性能。
rank48原输出名为`multivideo16x4_readout_rank48_kd_s180`，不为了统一文件名改写原run身份；
名字中的`kd`本身不证明某份历史训练已经完成或具有什么指标。

2026-10-07补回旧A100综合测速导出缺失的15份baseline源码／配置快照，逐文件匹配
其原SHA（含混合CRLF行尾），没有替换现用模型、重测或更新旧SHA清单。
该导出仍有9项文档／渲染图缺失或检查文件变化，不称完整；原README也保持不变。
具体恢复身份见 [测速源码恢复](../../../maintenance/storage/DEMO_TIMING_SOURCE_RESTORATION_20261007.json)。

历史 Spatial s586 报告的28级配置继承链亦已核对，14份此前仅在工作目录／Git归档中的
必要配置纳入主线，内容不变。来源见 [配置闭包](spatial_historical_config_import_20261006.json)。
该候选是既有历史对照，TEST选模属开发指标；不是上方当前实拍Spatial权重，不用旧代码
覆盖后续模型，也没有重新训练／评估。配置加载与逐文件SHA测试不等于完整PT/资产复现。

实际 `2026OpticsMoE_t06_a100_batch` 的两份未提交运行源码现已归入本任务，
不再需要到旁边的独立工程找测速脚本：

- `quality_token_batch_benchmark`：每批不同视频、每视频4帧、448×448；`sweep` 与
  `formal` 分开，558条正式TEST的首批计入，没有正式 warmup。主计时为全部输入已在
  GPU 上、完整Qwen前向至质量分数；Vision block0计时仅为并列的次要诊断。
- `quality_token_dataset_once`：4/9/16帧单视频的既有方案1/2入口，恢复原服务器
  schema 2（同步墙钟为主、CUDA-event并列）。原 schema 1 报告保留，不转换或重写，
  两种时钟与不同批量不能混作一个速度数字。

两入口恢复 GPU 型号与实际功率上限校验，显式使用 CUDA-visible 的物理编号／UUID。
批量入口默认要求 A100，dataset-once 默认要求 RTX5090D；跨GPU必须显式指定
`--expected-gpu`，不以绕过校验或套用575W来生成 A100 报告。新增输出保护会在
查询 GPU 前拒绝已有目标目录，旧结果不覆盖。入口参数可用相应模块的 `--help` 查看。

本次没有加载Qwen、PT或视频，没有测速、训练或触碰GPU。12项无Torch/设备的 CPU
依赖、身份、预处理审计及旧输出保护测试通过；13项批量性能函数、5项dataset-once
非入口定义与封存源码AST相同，已有共享函数和计时器默认行为另核验。
来源、投影范围和SHA见 [历史baseline迁入记录](batch_baseline_source_import_20261006.json)。
它不代表另一机器已完成实际功率/吞吐复测，也不能拿这些历史计时套给正式实拍模型。

## RTX 5090 D 光学 MoE 分段计时

T01–T04 与 T06 的 CCD 后串行电子处理、并行残差、跨层 SLM 场重建、bridge 和任务头已经
在同一块 RTX 5090 D 上按计算图边界正式测量。每个分量 50 次预热、1000 次正式同步调用；
T06 每次调用是一幅同时承载 16 个视频×4帧的物理场。任务级临界路径、能量口径、冻结
Qwen 对照和论文图统一见
[`reports/5090d_moe_and_qwen_baselines_20260907`](reports/5090d_moe_and_qwen_baselines_20260907/README.md)。

该报告明确区分物理光场时间、可被光路覆盖的并行残差、必须串行的 CCD 后处理和任务尾部。
其中 80.388 W 只能计算光学设备能量代理；在没有同步采集电子处理 GPU 功率前，不把它写成
完整光电系统能耗。

## 冻结 CLIP／YOLO 视频质量历史 baseline

旧 CLIP ViT-B/32 与 YOLO11s 的四帧 Temporal／Spatial baseline 运行入口已从其原
运行提交 `95904ecf97bcefe4a7ae76a6eab08a1d7cb217f0` 原样恢复到本任务。
命令、冻结参数与数据合同见 [VISUAL_BACKBONE_BASELINES.md](VISUAL_BACKBONE_BASELINES.md)。
历史结果、选模口径、原PT/缓存SHA和抽取耗时见
[CLIP／YOLO历史记录](reports/reproduction/VISUAL_BACKBONE_LGVQ_20260922.md)；
原中转说明仍保留，不把其2026-09-22成绩或耗时套给当前光电模型。
两模型只拟合五个无偏置质量输出行，共10240参数；按原无VAL划分的TEST SRCC选模，
属于开发指标，不称独立泛化。本轮未重训、未读取视频、未重评性能或测速。
原报告和本地中转源码仍原位保留，来源及CPU合同核验见
[恢复记录](../../../maintenance/storage/T06_VISUAL_BASELINE_IMPORT_20261005.md)。
