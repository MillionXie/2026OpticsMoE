# T01 Caltech101 三系统正式对照（单次结果）

日期：2026-09-06  
统一随机种子：42  
状态：三套系统已按同一 split 和检索协议完成；当前仅有一次正式重复，因此不能报告跨种子标准差。

## 1. 结论

| 系统 | Top-1 | Top-3 | MRR | 选中 epoch | 可训练参数 |
|---|---:|---:|---:|---:|---:|
| 光学 Router + Top-2 MoE + 同尺度融合 | **91.0%** | 96.5% | **0.9411** | 20 | 2,782,485 |
| 激活专家相位参数匹配的 dense D2NN | 90.5% | **97.0%** | 0.9382 | 25 | 2,024,461 |
| 冻结 Qwen3-VL-Embedding-2B | **99.5%** | **100.0%** | **0.9975** | 不训练 | 0（冻结参数 2,127,532,032） |

在本次 seed=42 对照中，主方法相对 D2NN 的 Top-1 高 **0.5 个百分点**、MRR 高
0.0029，但 Top-3 低 0.5 个百分点。冻结 Qwen 是未压缩教师基线，在这个固定 10 类子集上
明显更高；学生方法的论文论点应是光电硬件约束下的性能与稀疏路由价值，而不是超过教师。

## 2. 公平性口径

三组使用同一 Caltech101 target-10 划分：训练 2,625 张、gallery 30 张（每类 3 张）、
test 200 张（每类 20 张）。训练系统均为 30 个完整 epoch，每 5 epoch 在 test 上评估，
按 EMA test Top-1 选权重；同分时再比较交叉熵。该口径按当前项目约定执行，属于
`selection_biased=true`，不可表述为独立、无泄漏测试。

D2NN 的匹配对象是“主方法实际激活的专家相位参数”，不是主方法全部参数：

| 相位参数口径 | 主方法 | D2NN |
|---|---:|---:|
| 每个模态实际激活的专家相位 | 2 × 224² = 100,352 | 2 × 224² = 100,352 |
| Vision + Language 激活专家相位 | **200,704** | **200,704** |
| Router 相位 | 100,352 | 0 |
| global 相位 | 456,968（2 × 478²） | 0 |
| 存储的全部光学相位 | 958,728 | 200,704 |
| 每样本实际参与计算的相位值 | 758,024 | 200,704 |
| CCD 捕获次数/样本 | 6 | 4 |

因此，只能称 Baseline A 为“激活专家相位预算匹配”，不能称为总参数量、总光学面积或
采集次数完全匹配。D2NN 保留两个 CCD→电子归一化→重载边界，是 dense O/E/O 对照，
不是相位面连续串联且中间无探测的传统多层 D2NN。

## 3. 主方法内部证据

四个融合点均执行：

```text
rE = stopgrad(RMS(E)); rO = stopgrad(RMS(O))
M  = (1-alpha) * E/rE + alpha * O/rO
F  = rE * M / stopgrad(RMS(M))
```

选中权重的 alpha 分别为：Vision block-1 0.05450、Vision block-2 0.05525、
Language block-1 0.05400、Language block-2 0.05412；对应电子系数约为 0.9455、
0.9448、0.9460、0.9459。融合前四处光/电 RMS 比约为 0.481、0.432、0.443、
0.425，经尺度对齐后均为 1.0，因此 alpha 可直接解释为同尺度下的混合权重。

相位相对初始化的整体 RMS 变化从 epoch 5 的 0.1268 rad 增至 epoch 30 的 0.1975 rad，
说明 mask 确实被训练。Vision Router 在 epoch 30 的四专家硬选择计数较均衡（最少
1,222、最多 1,357）；Language Router 虽然四个专家最终都被使用，但非常集中（最少
1、最多 2,640）。后者是当前模型的明确局限，不能只用软路由熵掩盖。

## 4. 可追溯证据

训练时 Git commit：`692e8534a17518bc0b8ef63fb9110cfd2e5ed989`。

- 主方法 checkpoint SHA256：
  `77946d90079d560ee99c0249950162715916f799a24eff8ddf1beaf1a2e49ace`
- D2NN checkpoint SHA256：
  `492f38f3b9a305adcc84672137d59c2e7281caaf780d790ca6f9c3c34d2028a2`
- warmstart5 Stage-B EMA 源 checkpoint SHA256：
  `6a27f54d8c869cce46150583383a127b0ba47b3d34503f5753aa23974ac1e55d`

服务器 run：

```text
LightGenV2/tasks/t01_object_retrieval/runs/simulation/moe_router_scale_seed42
LightGenV2/tasks/t01_object_retrieval/runs/simulation/d2nn_matched_seed42
LightGenV2/tasks/t01_object_retrieval/runs/simulation/qwen_frozen
```

机器汇总结果位于：

```text
LightGenV2/tasks/t01_object_retrieval/reports/formal_comparison/
```

早期自动生成的 `student_architecture.json` 继承了历史报告中的初始化文字，错误写成
“from scratch”；D2NN 子字段也残留了 MoE 描述。真实初始化由每个 run 的
`initialization_report.json`、严格权重检查及上述源 SHA 证明。此问题只影响说明字段，
不影响计算图、权重或结果；生成逻辑已在后续提交中修正，原始 run 证据不做静默篡改。

## 5. 尚未完成的统计

- 尚未完成 seed 43、44，因此当前差异可能包含随机波动；
- 尚未做统一硬件实测和端到端延迟对照；
- 尚未解决 Language Router 的硬选择集中问题。

在补齐至少三次独立重复前，论文表格应把本页数字标为“single run, seed 42”。
