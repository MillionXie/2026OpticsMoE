# T16 多模态光电终身学习（最终协议）

2026-10-02 用户确认保留最终版本和必要对照。本任务不是 T12 文生图或 T13 视频鲁棒训练。
已从实际服务器 `/DATA/DATA1/guest3/t12_cross_modal_20260920` 的干净任务源码
`2f2f6946f38c7d06320eb29314654a6260f3d618` 纳入当前 LightGenV2 工作目录；没有整仓合并旧分支。
来源与逐文件 SHA 见 [source_import_20261002.json](source_import_20261002.json)，已纳入并发布到统一 main；
原服务器运行目录及数据仍保留，不将Git发布等同于三端运行工程已切换。

2026-10-04共享T09视觉前端收拢时支持了明确声明的CNN宽度；T16仍使用原默认16/32/64。
原共享视觉PT严格载入新旧类且合成输入输出逐位相同，29项T16合同测试通过。
依赖清单保留原源码SHA并记录本次兼容来源；没有改变T16模型、权重或原正式结果。

## 正式结果与必要对照

原始三张矩阵及追加对照见[最终报告](reports/FINAL_RESULTS_AND_TRAINING_20260926.md)。
下表均为最终 D 阶段、完整测试划分的四任务**宏平均召回率均值**，不是光路实拍或普通准确率。

| 方案 | 四任务均值 | 保留原因 |
| --- | ---: | --- |
| MoE 原全量 replay | 78.38% | 正式主结果，保留全部 A/B/C/D 下三角 |
| D2NN 无 replay | 48.77% | 遗忘对照，不能用独立模型结果拼接 |
| D2NN 全量 replay | 75.79% | 回放对照 |
| D2NN 每旧任务300条记忆 | 62.02% | 记忆容量对照；每轮重复50次须披露 |
| MoE Physical 排序优化 | 78.62% | 后验探索性改善，不替换原正式矩阵 |

还保留四个独立 D2NN 的固定模型跨任务4×4矩阵。
第二轮路由均衡78.34%未提高均值，原始报告/run保留用于审计，不设为最终最佳。
模型为同一冻结视觉CNN、原角谱传播/OEO、16专家4→8→12→16开放、单一无偏置 Linear(784,10)；
不是纯被动光学。旧专家冻结，router/全局相位/读出持续更新；所有正式链按验证选模。

## 代码和数据

### 名称含t12的四份旧目录，不是文生图

2026-10-08本机无资产的旧 `.worktrees/t12_d2nn_baselines` 已整体收进
`archive/frozen_worktrees_20261006/t12_d2nn_baselines`：5069份原文件移动前后
逐SHA一致，HEAD仍5d1ee1174ba445fb807380f7dd0c28c03a592f9d，无原修改或
非缓存实验产物。源码和缓存保留且Git登记随归档位置更新，不删除必要D2NN对照。
这仅处理本机旧源码入口；下表服务器同名目录内的六份best/last及结果仍原位保护。
现用本机cross_modal的独有改动不移动，不用归档副本覆盖本页main入口。

2026-10-07实际核对下列服务器目录：它们属于本任务的早期多模态终身学习沿革，
不是T12文生图，也不是可以仅凭Git状态干净就删除的代码副本。

| 服务器guest3下的旧目录 | 当初用途 | 当前保留资产 |
| --- | --- | --- |
| `t12_cross_modal_capacity_20260920` | 早期容量／损失权重试验 | 8份best/last及原配置／结果 |
| `t12_cross_modal_video_20260920` | 早期视频任务及划分协议 | 12份best/last及原配置／结果 |
| `t12_d2nn_baselines_20260920` | 早期顺序D2NN对照 | 6份best/last及原配置／结果 |
| `t12_sequential_d2nn_20260920` | 早期顺序D2NN回放对照 | 8份best/last及原配置／结果 |

这些run仍在各目录的旧 `LightGenV2/tasks/t12_cross_modal_lifelong/runs/`，
阶段目录名为sen12ms／clevr／sonyc／video，不能把它们的指标套给本页最终四任务矩阵。
34份PT本次只读取原字节核验SHA，没有载入模型、训练或评估；146份非缓存被忽略资产
以及旧测速均原位保留。精确目录、HEAD及PT SHA见
[源码与历史资产身份](source_import_20261002.json)的`legacy_protocol_asset_protection_20261007`。
这不证明其数据协议与当前最终版相同，也不批准删除这些旧run；本轮没有切换旧目录。

- 正式 MoE/D2NN 顺序链：`train_lifelong_moe.py` / `train_lifelong_d2nn.py`。
- 独立固定模型迁移：`eval_d2nn_fixed_4x4.py`。
- 单任务/前端：`train_eurosat.py`、`train_other_tasks.py`、`pretrain_clevr_vision.py`。
- 必要依赖：旧 `t13_four_modal_lifelong` 的传播/编码、`t14_shared_readout_lifelong` 的图文配对编码、
  `t09_multimodal_matching` 的冻结视觉网络，以及 `demo_check/.../optical_reference/optics.py`。
  本次只带入依赖所需文件，不复制这些旧任务所有试错脚本。
- 数据、原协议JSON、共享冻结视觉PT、各阶段best/last和原始结果仍在服务器，未搬走/删除。
  [复现入口](reports/reproduction/README.md)说明定位与核验要求。
- 当前版本性能计时尚未核定，不能借用其他 LightGen 模型的 A100/相机时间。

本地 CPU 结构/数据合同测试29项通过（PyTorch 2.11.0+cu128，CPU，2026-10-02），
没有重新训练或重新评估测试集；测试通过不等于已将全部数据/权重打包到本地。
