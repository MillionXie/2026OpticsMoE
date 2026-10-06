# T06 视频质量评价

**2026-10-04整理覆盖：**下文保留历史仿真、baseline和测速协议，不代表所有默认命令已适配。
正式Spatial/Temporal实拍版本、各自权重及已发布核心代码见
[当前版本与待收敛边界](CURRENT_VERSION_20261004.md)。主线核心通过57项CPU合同测试、
两份正式PT严格加载；旧Temporal-36默认profile和后续实拍微调入口仍须单独核验。

2026-10-06补齐主线测试依赖：`lab_stage_coordinator.desktop_code` 仅返回历史阶段脚本
字符串，不连接机器、不打开SDK、不启动采集；T03测试不再依赖未跟踪的
`lab_manual_stage.py`。同时恢复历史 `stage_config` / `effective_stage_identity` 纯配置
函数及原测试，既有 `lab_bench` 函数未改。来源见
[函数投影身份](stage_coordinator_source_identity_20261006.json)。这不是迁入完整旧控制器，
也不是宣称现有采集入口已经应用逐层曝光；未改变硬件合同或正式模型。


## 历史仿真结论（不替代上方当前实拍版本）

Spatial 的当前正式归档是 `spatial_single_video4_balanced`：一条视频均匀取 4 帧并排成
2×2，**没有多视频复用**。它使用两套物理光 Router Top-2、六次光传播和 20% 名义
未调制分量，在 558 条 test 视频上达到 SRCC 0.6393、KRCC 0.4642、PLCC 0.6743、
RMSE 8.452、MAE 6.646。入口、checkpoint SHA 和光关闭对照见
[`reports/paper_results/spatial_single_video4_balanced`](reports/paper_results/spatial_single_video4_balanced/README.md)。

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
两模型只拟合五个无偏置质量输出行，共10240参数；按原无VAL划分的TEST SRCC选模，
属于开发指标，不称独立泛化。本轮未重训、未读取视频、未重评性能或测速。
原报告和本地中转源码仍原位保留，来源及CPU合同核验见
[恢复记录](../../../maintenance/storage/T06_VISUAL_BASELINE_IMPORT_20261005.md)。
