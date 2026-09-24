# T16 单任务复现入口（2026-09-24）

本页只记录已实际运行的仿真。运行目录位于服务器 `/DATA/DATA1/guest3/t12_cross_modal_20260920/LightGenV2/tasks/t16_zero_phase_ccd_lifelong/runs/simulation/`。每个目录的 `config.json`、`command.txt`、`history.json`、`result.json`、`status.json` 和 `best_checkpoint.pt` 是原始证据；本页的数字以该文件为准。运行环境为 Python 3.11.15、PyTorch 2.6.0+cu124、CUDA 12.4。GPU 用 `CUDA_VISIBLE_DEVICES=<下表中的 GPU UUID>` 精确绑定；其他用户的进程未被操作。源码只从 GitHub 的指定 commit 检出，权重和数据不进 Git。

所有模型均为原项目 0.1 m 角谱传播、原始相位参数全零初始化（物理相位 π）、层间相同 OEO、直达最终 CCD，无额外透镜或傅里叶层。MoE 是中心优先 16 槽几何，A 阶段只开放 4/7/10/13 槽并做 dense soft routing；D2NN 是两层相位。双方各有自己的一层可训练、无偏置 `Linear(784,10)`，无可学习输入 encoder。所有列出的权重均独立训练**单任务**，不代表终身学习矩阵。完整验证集的宏平均召回选权重，测试仅在选模后对所选权重运行一次；明确标 `val-only` 的候选未读取测试划分。

| 数据协议 | 训练／验证／测试样本 | 协议 SHA256 | 数据 manifest SHA256 |
| --- | --- | --- | --- |
| 成对 EuroSAT RGB/SAR，10 类，同地点一对算一条 | 15,998／5,465／5,429 | `8ffff7cf42719bc454ccca3f568c18dba1f056eeb7ada7f51520e749e21ffb48` | `c850e41270f77692a0e2e5def4e882a3a77a517afd9ad243ea3007295f7a1919` |
| CLEVR 图＋颜色形状问句是否匹配，2 类 | 140,000／15,000／15,000 | `5ace5c945743278c8b0df232d7d5e421c54a9e01475282c42a20a2d58800a6f4` | `63e6646b66d3bceee80c5e4b9e9fe75b5070ce4200bb1dac0e3d8dec4e6142ed` |
| Speech Commands 语音＋词是否匹配，2 类 | 12,526／1,686／1,734 | `f97cb20c365729e9424c65686e3db927d65246b645de8053557eac2a222f3489` | `1038d19394848d04ef721684559fac6bd90f799456968d279a4b54e66c9b879c` |
| Physical Concepts 视频＋单条描述是否匹配，2 类，**带符号帧差实光场** | 140,800／29,304／29,896 | `23cd429bf143b2f7bb87ddd16fce15bbf3e59d6cd9793074827647e06d94079d` | `259bf3e1407cccbdfff340d9f8a90a2ef37672764b04387521f38e3d8828d559` |

Physical 视频原始段数是 70,400／14,652／14,948；每段视频产生正确和错误描述各一条。错误描述在每个划分内错排，使十种描述的正负频次相等，且同一视频的正负对共用视频场。**当前帧差有负数**，因此不是纯非负幅度输入；负号等效额外 π 相位。它是这一版明确的输入合同，不可与将正负帧差分区后可能得到的新结果混用。历史 Physical 10 选 1 结果另见任务 README，也不能与这里的二分类数字横比。

| run ID | 源码 commit | 最佳轮 | 验证宏平均召回 | 测试宏平均召回 | 最佳权重 SHA256 |
| --- | --- | ---: | ---: | ---: | --- |
| `eurosat_pair_moe_center_linear_s17_8f28_uuid` | `8f280a5d661d37b43d40eace585c43bd00d819d0` | 23 | 68.02% | 64.61% | `112eca60f47076f05223b21f3bf53c43183bda522fcfc654d60eef9b6dad47a8` |
| `eurosat_pair_d2nn_center_linear_s17_8f28_uuid` | 同上 | 24 | 61.77% | 57.59% | `3c335b47c060cef52a21a7544f6a8160df8d7426c8f82e47402e1dc05dd1006c` |
| `speech_binary_moe_center_linear_s17_efe2_uuid` | `efe2a44c5dd653263cc8907e3e1225a71e4b8d4d` | 9 | 83.75% | 81.89% | `bf87ce744b38feaa7cb6cb57ca30699306f780b60b7cb176b0efb29ac92f5671` |
| `speech_binary_d2nn_center_linear_s17_efe2_uuid` | 同上 | 12 | 64.95% | 67.19% | `3c8e40485ac14c806706c903c1eb5d083dcf9d709ffb7b1efd47d410da89baf2` |
| `physical_binary_moe_center_linear_s17_977_uuid` | `9777aa2258e3f416069c404afd3c0ea5bc22efa8` | 8 | 87.47% | 87.29% | `393bd0f0ade77344f363abfc765181a57a61277104c7f0121711cace2731a1b2` |
| `physical_binary_d2nn_center_linear_s17_977_uuid` | 同上 | 5 | 87.98% | 87.86% | `03b140c4c7beb01e8683f0cfe122bd28e2311c8214b10a87a0d47ceb56d6aee1` |
| `clevr_pairs_moe_pairloss4_valonly_s17_8a28_uuid` | `8a282af30775f2271b74164cde0b1f861c9f1d3a` | 4 | 55.96% | **未测试** | `edf953db17f52c6b89b6bcec7ff4cdf29c38232c3725877c90eb7965771da42c` |
| `clevr_pairs_moe_pairloss4_resume12_valonly_s17_0a1_uuid` | `0a1ef23761caadc52b5203e6ecdfa0a9ee1c9886`，从上一行第 4 轮继续 | 10 | 59.14% | **未测试** | `2a1b203650ff9ac93d0b9edfd1115c5c4ac91e8e7980f0b3b1cff3cf0761e3ac` |
| `clevr_compact_moe_pair4_valonly_s17_fc6_uuid` | `fc6bb8fe7545ae744dc4771c25fa7d4a6542c229`，独立输入编码 | 4 | 52.39% | **未测试** | `c1599bfb2ccc9a0693b33d884454ef58dc675504a2fa9984947fbb8ff71d0318` |
| `clevr_compact_d2nn_ce_valonly_s17_fc6_uuid` | 同上 | 4 | 50.00% | **未测试** | `92ec7704d44007fd86a75c504a05203a8462b96a9f8c16d06a10128ca7a8c523` |
| `clevr_moe_eight_expert_pair4_valonly_s17_66ca_uuid` | `66cae7811b7e7c5894f3346ac03d717883c3e486`，8 槽独立容量诊断 | 8 | 57.86% | **未测试** | `b5b4729425043c3df4eeecd5ea258a7ca375018d2fb3f492bd842c96c85b2777` |
| `clevr_moe_eight_expert_balance1_resume8_valonly_s17_4698_uuid` | `46984d9dfc8f12d9a1def3a1b4ea95f225082c6d`，从同 commit 的 4 轮源 run 继续 | 8 | 58.98% | **未测试** | `a30dae0109fac97b8b8a3b4f9a6b7fea0bb16d703df7f91e9c6b4b862b00f884` |

CLEVR 原交叉熵 MoE/D2NN 完整训练 4 轮后最好验证 50.33%/50.02%，因机会水平停止且未测试；配对损失权重 1.0／4.0 的 MoE 第 4 轮验证分别 54.05%／55.96%。权重 4.0 的 run 保留模型和 Adam 状态延长到第 12 轮后，第 10 轮最佳完整验证 59.14%（负类／正类 66.08%／52.20%），路由槽 7 在全部验证样本上最大，仍未达到准入目标。续训 `last_checkpoint.pt` SHA256 为 `bee13a4326a179cd714531c4843082f62c343dc468f5dae399f766ff5486fedc`，数据协议 SHA 与上表相同；候选全程未读取测试集，不应以验证分数冒充测试结果。Physical 二分类 MoE 已过 70%，但同预算 D2NN 测试高 0.57 个百分点；目前不能声称 MoE 在四任务上全面更好。已保存的验证集“所有相位置零且线性头不变”反事实见对应 run 的 `phase_dependence_val.json`；它只衡量相位敏感性，不是因果精度归因。

另一个 `clevr_compact` 输入协议保留相同图像、问句、标签、图／文各 0.5 的入射功率和完整划分，只把最长 9 个实际 token 行铺到整个文字象限，光路及读出不变。128 对训练样本的固定编码审计确认图像象限一致、正负问句文字不同、总入射功率为 1；文字象限非零行由原编码的 32 行变成 112 行。两模型从零独立训练，MoE 用原配对损失权重 4，D2NN 用普通交叉熵；同样 4 个完整 epoch 后验证为 52.39%／50.00%，低于旧编码 MoE 同轮的 55.96%。因此两条候选停止并释放 GPU，状态为 `stopped_futility_after_full_epoch4`，未读取测试标签。此实验**否定了单靠铺展文字编码即可解决泛化问题**；不能把两种输入协议的权重或精度当成同一训练链。

单任务容量诊断从零开放并训练 8 个中心优先专家（正式 B 阶段则要先学 A、冻结旧四专家并回放旧数据，所以不能混同）。在相同原始 CLEVR 编码、完整划分和配对损失下，8 轮最佳验证 **57.86%**，仍低于 4 专家延长训练的 59.14%；新增槽 2/5/12/15 在所选权重下合计仅得 **1.02%** 路由功率，说明单纯开放参数不能使 router 使用它们。此 run 未测试，也不是终身学习矩阵单元。

对应弱批级路由均衡权重 1.0 的 8 专家候选首轮新增四槽得到 46.29% 功率；第 4 轮后为了释放较慢 GPU，以源 run `clevr_moe_eight_expert_balance1_valonly_s17_4698_uuid/last_checkpoint.pt`（SHA256 `71d4c1eb65255626df10402c3f23b4e98797b50878244859acdbe7bec9cb0f36`）保留模型、Adam 和历史到新 run，原 run 状态是 `stopped_after_epoch4_gpu_handoff`。第 8 轮最佳验证 **58.98%**（负类／正类 66.87%／51.09%），新四槽平均功率合计 **47.63%**，在 **15.27%** 样本上权重最大。它比无约束 8 专家高 1.12 个百分点，但未超过旧 4 专家最佳 59.14%；按槽标准差仍只有约 0.004–0.024，不能仅凭批平均功率称为内容驱动专家分工。两段训练均未触碰测试集。

两个 8 专家最佳 checkpoint 又各自对验证样本索引 0、1（同一张原图、正负两条不同颜色形状问句）生成少量 router/最终 CCD 图。图与逐样本权重 JSON 在各自 run 的 `router_val_pair01/`，不提交原始 CCD 图到 Git。无约束／均衡约束的路由权重在这一对中的 L1 差分别约 `0.04/0.09`；两者都把正负两问判为正类。此图只是一个具体失败样本，不代替上表的完整验证集统计。

完整命令在各 run 的 `command.txt`。从对应源码 commit 的仓库根目录执行时，以下为同义命令；`RUNS` 指上文服务器 `runs/simulation` 绝对目录，`PY` 指 `/home/guest3/miniconda3/envs/xml/bin/python`，`EURO`、`CLEVR`、`SPEECH`、`PHYS` 依次指上表对应的 `protocol.json` 绝对路径。每条命令的 `CUDA_VISIBLE_DEVICES` 必须用当时空闲 GPU 的 UUID，而不是 CUDA 逻辑序号。

```bash
GPU_UUID="<replace-with-a-free-GPU-UUID>"  # 运行前先查 nvidia-smi 并替换
PY=/home/guest3/miniconda3/envs/xml/bin/python
RUNS=/DATA/DATA1/guest3/t12_cross_modal_20260920/LightGenV2/tasks/t16_zero_phase_ccd_lifelong/runs/simulation
EURO=/DATA/DATA1/guest3/demo_reproduction_data/t13_four_modal_features/eurosat_full_s17_v1/protocol.json
CLEVR=/DATA/DATA1/guest3/demo_reproduction_data/clevr_v1_full_ccby4/clevr_full_source_s17_v1/protocol.json
SPEECH=/DATA/DATA1/guest3/demo_reproduction_data/t13_four_modal_hard_v2/speech/protocol.json
PHYS=/DATA/DATA1/guest3/demo_reproduction_data/t13_four_modal_hard_v3_physical/protocol.json
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_eurosat --protocol "$EURO" --architecture moe --out "$RUNS/eurosat_pair_moe_center_linear_s17_8f28_uuid" --epochs 25 --min-epochs 8 --patience 5 --batch 16 --eval-batch 16 --lr 0.001 --seed 17
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_eurosat --protocol "$EURO" --architecture d2nn --out "$RUNS/eurosat_pair_d2nn_center_linear_s17_8f28_uuid" --epochs 25 --min-epochs 8 --patience 5 --batch 16 --eval-batch 16 --lr 0.001 --seed 17
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task speech_binary --protocol "$SPEECH" --architecture moe --out "$RUNS/speech_binary_moe_center_linear_s17_efe2_uuid" --epochs 12 --min-epochs 4 --patience 3 --batch 16 --eval-batch 16 --lr 0.001 --seed 17
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task speech_binary --protocol "$SPEECH" --architecture d2nn --out "$RUNS/speech_binary_d2nn_center_linear_s17_efe2_uuid" --epochs 12 --min-epochs 4 --patience 3 --batch 16 --eval-batch 16 --lr 0.001 --seed 17
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task physical_binary --protocol "$PHYS" --architecture moe --out "$RUNS/physical_binary_moe_center_linear_s17_977_uuid" --epochs 8 --min-epochs 4 --patience 2 --batch 32 --eval-batch 32 --lr 0.001 --seed 17
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task physical_binary --protocol "$PHYS" --architecture d2nn --out "$RUNS/physical_binary_d2nn_center_linear_s17_977_uuid" --epochs 8 --min-epochs 4 --patience 2 --batch 32 --eval-batch 32 --lr 0.001 --seed 17
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task clevr --protocol "$CLEVR" --architecture moe --out "$RUNS/clevr_pairs_moe_pairloss4_valonly_s17_8a28_uuid" --epochs 4 --min-epochs 4 --patience 2 --batch 32 --eval-batch 32 --lr 0.001 --seed 17 --clevr-pairwise-weight 4.0 --skip-test
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task clevr --protocol "$CLEVR" --architecture moe --out "$RUNS/clevr_pairs_moe_pairloss4_resume12_valonly_s17_0a1_uuid" --epochs 12 --min-epochs 4 --patience 3 --batch 32 --eval-batch 32 --lr 0.001 --seed 17 --clevr-pairwise-weight 4.0 --skip-test --resume-checkpoint "$RUNS/clevr_pairs_moe_pairloss4_valonly_s17_8a28_uuid/last_checkpoint.pt"
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task clevr_compact --protocol "$CLEVR" --architecture moe --out "$RUNS/clevr_compact_moe_pair4_valonly_s17_fc6_uuid" --epochs 8 --min-epochs 4 --patience 3 --batch 32 --eval-batch 32 --lr 0.001 --seed 17 --clevr-pairwise-weight 4.0 --skip-test
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task clevr_compact --protocol "$CLEVR" --architecture d2nn --out "$RUNS/clevr_compact_d2nn_ce_valonly_s17_fc6_uuid" --epochs 8 --min-epochs 4 --patience 3 --batch 32 --eval-batch 32 --lr 0.001 --seed 17 --skip-test
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task clevr --protocol "$CLEVR" --architecture moe --moe-active-experts 8 --out "$RUNS/clevr_moe_eight_expert_pair4_valonly_s17_66ca_uuid" --epochs 8 --min-epochs 4 --patience 3 --batch 32 --eval-batch 32 --lr 0.001 --seed 17 --clevr-pairwise-weight 4.0 --skip-test
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task clevr --protocol "$CLEVR" --architecture moe --moe-active-experts 8 --route-balance-weight 1.0 --out "$RUNS/clevr_moe_eight_expert_balance1_valonly_s17_4698_uuid" --epochs 8 --min-epochs 4 --patience 3 --batch 32 --eval-batch 32 --lr 0.001 --seed 17 --clevr-pairwise-weight 4.0 --skip-test
CUDA_VISIBLE_DEVICES="$GPU_UUID" $PY -m LightGenV2.tasks.t16_zero_phase_ccd_lifelong.train_other_tasks --task clevr --protocol "$CLEVR" --architecture moe --moe-active-experts 8 --route-balance-weight 1.0 --out "$RUNS/clevr_moe_eight_expert_balance1_resume8_valonly_s17_4698_uuid" --epochs 8 --min-epochs 4 --patience 3 --batch 32 --eval-batch 32 --lr 0.001 --seed 17 --clevr-pairwise-weight 4.0 --skip-test --resume-checkpoint "$RUNS/clevr_moe_eight_expert_balance1_valonly_s17_4698_uuid/last_checkpoint.pt"
```

若未来重新训练，必须使用**新 run ID**；以上已有目录不能覆盖。执行前核对协议及 manifest 的 SHA256、GPU 空闲状态和 Git commit，运行后再核对 `status.json`、`result.json`、最佳权重 SHA256。只有验证通过并确立任务输入合同时，才能进入下一轮终身学习三矩阵。
