# T04 语义交互（OpenMoji）

## 两条OpenMoji用途必须分开（2026-10-07整理）

本目录是**应用/布局/电子容量开发与对外复现**的入口及共享后端；
[`t04_openmoji_robust_ablation`](../t04_openmoji_robust_ablation/README.md) 是
**实验室robust消融**的入口。共享代码不表示同一数据、结构或权重，也不允许互套性能。
应用线包括不同物体大小、分层/遮挡场景、缩减电子容量、师姐复现及选定版本的光路验证；
robust线比较固定结构下的G2–G5、直接部署及实拍电子微调。

- `reports/layout_design_20260920/PLAN.md`：最初版式样稿，不是训练结果。
- `reports/layered_scene_pilot_20260920/README.md`：大小层次场景的10轮连通试验，
  不是最终指标；对象大小不同，但当时语义锚点输出仍是6×6。
- `reports/layered_scene_focus_changed_iou_20260920/README.md`：历史应用候选，
  仿真修改格.9315、同权重去光.4220、epoch60/PT SHA
  `b8ecf51c8ca8a0f15de8338a75a7778e74bb5ebcbef8a1bc990758d2ab90bc00`。
  这些原报告目前原位保留；它们不证明后续电子缩减/师姐交付最终版已经完成归属核验。
- 9月15日旧SHS包属于更早的 `routerfill_shared` epoch40（仿真.8715），
  原PT SHA `a69ddcee827749fb9202f9aef11ea45011e433d8b2f0151be2eec3db7dbff9eb`；
  不属于后来的G2–G5，也不是新版布局最终版。不能凭旧包名字或旧.9800成绩替代应用线最终身份。

应用线后续实验的指定版已按原交付说明及实际服务器资产确认：**新版分层场景、
电子expansion=.5、DC30＋小CCD噪声，epoch45，仿真修改格.8765、同权重去光.4845**。
PT是 `runs/simulation/layered_dc30_ccdsmall_selected_e45_s73_20260925/selected_checkpoint.pt`，
SHA `03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21`。
本地与实际训练服务器的PT、selection/audit/split记录均逐SHA一致，CPU读取元数据确认
epoch45和电子expansion=.5架构；不是自动最佳epoch70，不称独立TEST泛化。
原模型参数解析、DC/CCD向router/expert/global的传递和expansion架构后缀已恢复；
本机主线模型严格CPU加载该PT全部139项state通过，53项相关CPU检查通过。
训练服务器实际main `4ea21bb6` 也严格CPU加载同SHA的epoch45 PT全部139项state通过，
未执行模型推理、TEST、GPU或设备；这证明两处源码/权重构造兼容，不冒称全流程复现。
仅训练时启用原Router扰动；默认关闭、保持旧调用，主线低秩头和消融保护不回退。
没有重新评估1000条TEST，原87.65%仍引用封存审计，不把严格加载当作精度重测。
该PT的实拍精度本轮未核实；旧应用exp05另一权重的.9365仿真/.671直接实拍/.8375适配
不能套给它。下面旧网格及测速说明均属各自历史版本。

**交付边界：**现有 `build_lab_package.build` / `standalone.py` 明确限定旧标准头epoch40，
不是新版分层epoch45的独立交付器，不能直接替换PT或使用它生成新版包。
当前外部上传暂停；旧包保留，新版完整独立包尚未验收。本轮不自行打包或上传。
本轮只整理入口和身份，不训练、重评、修改现用实验或连接实验室设备。

### 大小层次应用源码与电子缩减配置已归主线

服务器原运行目录 `LightGenV2_worktrees/t04_layered_1bc120428` 的4份渲染/展示源码与
8份应用配置已逐LF SHA核对原提交 `a61a3746d39765991048d95e40812d1028d70c93`。
原 `layered_anchor6_svg_v3` 数据分派现恢复到本任务run入口，光电方法及完整冻结Qwen
baseline各用对应分层数据；评估展示也使用原分层合成器，不再退回旧同尺寸网格图。
来源及保留边界见 [应用源码身份](layered_application_import_20261007.json)。

现有历史应用profile包括 `layered_scene_pilot/formal/focus_changed_iou`、
`layered_scene_qwen_shared`、`layered_scene_electronic_exp1/exp05/exp05_e30`、
`layered_scene_exp05_dc30_ccdsmall`。exp1/exp05只把原电子残差MLP expansion改成1/.5，
不把这些profile描述为后续实验室rank48/rank64消融；exp05_e30旧配置注释中的.8715
仍需正确原PT绑定，不能拿另一个已有30轮PT冒充该成绩。

只恢复原分派，现有防覆盖检查、每种消融的独立输出后缀及训练函数保持。
7项无Torch的数据分派、配置继承、源码身份和AST测试不读取数据/PT，不代表SVG环境、
完整Qwen缓存或师姐独立交付包已经复现；原位数据、运行结果与测速仍保留。
本机单独执行原预览测试的6项因缺 `resvg_py` 失败，尚未完成原SVG预览复现；
所需历史依赖为 `resvg-py==0.5.0`，本轮未安装软件或换用PNG/其他渲染器掩盖差异。
全套324项维护测试通过只覆盖工程合同，不覆盖这些需真实SVG环境的预览测试。

## 2026-10-05 源码收敛边界

本批将实际服务器的语义核心、共享电子头和六份已有配置纳入主线，未修改运行目录、
数据、权重或光学合同。旧 `main_dc20`、`d2nn_dc20`、`qwen_pending` 入口保留；
增加已有的 `embedding_alpha40`、`embedding_alpha40_lean`、`embedding_d2nn_alpha40`、
`routerfill_shared`、`routerfill_shared_balance`、`qwen_shared` 入口。
共享读出配置中的光学方法和冻结 Qwen baseline 使用同结构、同种子初始化的电子头。
新增配置对应 TRAIN5000/TEST1000 的历史开发协议；TEST 选模不称独立泛化。

本批候选通过15项CPU合同检查、4套旧配置各56项共同字段对比和5项AST/默认分派检查。
这不是重新训练、完整Qwen执行或新的实拍精度；下方性能与测速仍属原历史版本，
不能套到共享头或robust新权重。主线入口同步也不表示Windows设备工程已切换。
正在使用的 robust 实验源码仍单独保护，最新未提交源码与恢复边界见
[源码保护记录](../../../maintenance/storage/T04_CURRENT_SOURCE_OVERLAY_20261005.json)；
不要从本页旧历史指标推断其最终效果。复现入口及数据、PT与硬件依赖仍需继续收敛。


输入是 `224×224` OpenMoji 场景和文本指令，指令任务为 `add / replace / move / remove`；输出是 `6×6` 类别网格和编辑网格，再由固定 OpenMoji 合成器得到目标图像。

本目录固定比较三组系统：

1. `main_dc20`：语言、视觉各含光学 Router，均为 4 专家 Top-2；随后各有一张 global phase。融合前做 RMS 同尺度归一化，训练含 20%–30% 相干零级分量与硬/软专家均衡。
2. `d2nn_dc20`：没有 Router；语言和视觉各使用两层普通 D2NN。四张 `224×224` dense 相位与主方法两个模态实际激活的四张专家相位参数量相等。
3. `qwen_pending`：冻结 `Qwen3-VL-2B-Instruct` 大模型 baseline。当前只生成 5090D 待测合同，不在共享训练服务器测性能、速度或功耗。

主指标为 changed-cell accuracy，并同时报告 foreground category accuracy、edit-grid IoU、object F1、scene exact match 和按四种操作分组的指标。

## 数据协议

- train：5,000 个合成场景，每个操作 1,250 个。
- test：1,000 个不同随机种子的场景，每个操作 250 个。
- train/test 同分布、样本种子不相交；validation 为无。
- epoch 1、每 5 epoch、末轮测试；按最高 test changed-cell accuracy 选择 checkpoint。
- 正式目录只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。

## 运行

历史A100同权重消融入口已收回主线（2026-10-06）。只在 `--phase evaluate` 使用
`--fusion-ablation remove_optical` 或 `remove_electronic`；默认 `none`。两模态内部
融合核心使用同一PT，不重训；去光时电子系数恢复为1。纯完整Qwen baseline没有
对应融合核心，拒绝消融选项。报告、逐样本、配置和图库按模式分开，已有评估输出
拒绝覆盖。11项无Torch CPU协议测试通过，训练函数与迁移前主线AST一致；没有新
实拍、测速或完整模型评估。源码身份见 [恢复清单](fusion_ablation_import_20261006.json)。
现用rank64设备工程未修改；本页历史入口与其不是同一部署版本。

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
主指标高 1.00 个百分点。机器可读结果和可视化见 `reports/dc20_comparison/`。
