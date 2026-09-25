# 四任务顺序实验的阶段 B 诊断（新链采用无帧差 Physical）

以下全是**完整验证集宏平均召回**，阶段 B 和电子教师候选均未打开测试集，不能填入最终三张测试矩阵。顺序为 A=EuroSAT RGB/SAR 配对十类，B=原图＋原自然语言颜色形状问句匹配二类，C=Speech 音文二类，D=`physical_binary_raw` 原八帧＋描述匹配二类。D 尚未运行；此前带符号帧差 `physical_binary` 的权重与成绩不兼容这条新链。

| 运行与方法 | 选中轮 | A 验证 | B 验证 | 状态 |
| --- | ---: | ---: | ---: | --- |
| 独立 MoE A 起点 `eurosat_pair_moe_center_linear_s17_8f28_uuid` | 23 | 68.02% | — | 历史正式单任务起点 |
| MoE B，CLEVR 单任务专家相位复制到新四槽，回放旧任务，路由均衡 0.3，`lifelong_stage2_moe_warm_clevr_rawphys_s17_1b3f` | 7/8 | 66.62% | 55.37% | 验证候选；旧专家相位精确不变 |
| MoE B，同源相位暖启动、路由均衡 1.0，`lifelong_stage2_moe_warm_balance1_rawphys_s17_1b3f` | 3/4 | 64.72% | 52.53% | 验证候选，低于上一行 |
| MoE B，同源相位暖启动、路由均衡 10.0，`lifelong_stage2_moe_warm_balance10_rawphys_s17_f086` | 3/4 | 58.75% | 54.47% | 验证候选，新增专家收光改善但旧任务受损 |
| D2NN A 起点 `eurosat_pair_d2nn_center_linear_s17_8f28_uuid` | 24 | 61.77% | — | 历史正式单任务起点 |
| D2NN B，**全量 replay**、常规交叉熵，`lifelong_stage2_d2nn_replay_rawphys_s17_1b3f` | 4/4 | 53.11% | 50.17% | 独立新增对照；不替代无 replay 下三角 |

回放的精确定义：每个 epoch 将每个已学任务的全部**训练记录各遍历一次**，各自打乱、批次交错，不使用早期 512 条固定记忆池，也不反复抽样旧记录。B 的训练记录数量为 EuroSAT 15,998、CLEVR 140,000；尽管同时出现的任务损失默认等权，整轮中 CLEVR 的批次约为 EuroSAT 的 8.75 倍，因此“全量回放”不代表**等总梯度权重**。MoE B 旧专家相位冻结，router、共享相位、唯一 `Linear(784,10)` 更新；D2NN 两层相位及其自己唯一 `Linear(784,10)` 更新。推理都不切任务头或掩掉十输出中的某些类。B 阶段按已学任务验证宏平均召回的算术平均选权重。两个模型光路参数量、输入及末端读出结构沿 T16 合同，D2NN 这条 replay 是新增诊断，不把它冒充原指定的 D2NN 无 replay 矩阵。

MoE 相位暖启动只复制原始 CLEVR 单任务 checkpoint 的四个专家相位到 B 新开放的四槽；A 的旧专家、router、共享相位与头均来自 A checkpoint。它没有为图文切换独立读出头。最佳 B 暖启动路由在 CLEVR 验证集上，旧中心槽 13 获得约 22.97% 光功率，并在 75.62% 样本上权重最大；新增四槽合计约 20.33%，说明路由尚未形成稳健的输入相关专家分工。把均衡权重从 0.3 增至 1.0 未提升分类；增至 10.0 时，新四槽的 CLEVR 平均功率合计约 45.90%，但 CLEVR 验证反而比 0.3 候选低 0.90 个百分点，EuroSAT 低 7.87 个百分点。因此“新槽收不到光”确实存在，却不是当前图文性能低的唯一原因；不再盲目加大均衡项。

EuroSAT A 的两个验证候选都不替换原起点：`eurosat_pair_moe_dihedral_cont_s17_1b3f` 从原 A 权重用同一旋转／翻转同时处理四个 RGB/SAR 区块并低学习率续训，最佳 67.58%；`eurosat_pair_moe_lowlr_balance_cont_s17_1b3f` 小学习率＋弱路由均衡，最佳 68.56%，只高原起点约 0.54 个百分点且路由塌缩未解决。两者均 `--skip-test`。

训练期电子教师只用于检查原图＋原问句是否能被一个简易图文网络学会，不参加光学推理，也未给 MoE 蒸馏。晚融合 CNN＋词嵌入的 `clevr_teacher_original_valonly_s17_3fc6` 最佳验证 50.79%；问句条件化图像特征的 `clevr_teacher_conditional_valonly_s17_a4f7` 最佳 50.00%，训练损失约 0.693。条件式教师在仅 16 个同图正负训练配对上 250 步可从 16/32 记到 32/32（`runs/smoke/clevr_teacher_pairs16_steps250_f086`）；这只排除严格无法拟合该小集合，**不能**证明完整数据正确或教师足够强。旧 t09 在不同输入、数据量和光路协议下使用任务监督预训练的约 32k 参数冻结视觉 CNN，不能将其 70% 左右成绩直接计入 T16，也不能未说明便加入当前四阶段。

原始命令、源码 commit、协议／权重 SHA、逐轮值、收光统计和状态保存在上述服务器 `LightGenV2/tasks/t16_zero_phase_ccd_lifelong/runs/simulation/<run ID>/`；小样本教师记录位于 `runs/smoke/`。正式推进闸门：先用训练与验证排查 CLEVR 图文编码和标签并找到实质高于机会水平的 B；不能为填下三角而把当前 55% 候选送入 C/D。用户授权本项目同时最多使用三张 GPU，各候选均独立绑定 GPU UUID，进程结束释放。
