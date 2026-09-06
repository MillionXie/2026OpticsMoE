# T01 Caltech101 物品检索

本目录只维护论文正式对照的入口、配置和结果索引；历史实现仍由
`experiments/` 提供兼容后端。当前固定比较三套系统，除此之外的试验不得混入正式表。

## 1. 数据与共同协议

- 数据集：Caltech101 中固定 10 类：`airplanes`、`Motorbikes`、`Faces`、
  `Leopards`、`accordion`、`grand_piano`、`scorpion`、`sunflower`、
  `watch`、`yin_yang`。
- 每类 3 张 gallery；每类 20 张 test；余下图像全部用于 train。
- Prompt：`Represent this image for image-to-image object-category retrieval.`
- Qwen 输出取最后一个有效 token 的前 64 个 Matryoshka 维度并做 L2 归一化。
- 指标：Top-1、Top-3、MRR；三个系统使用完全相同的 split manifest、gallery 聚合和检索实现。
- 训练系统每 5 epoch 测试一次，以最高 EMA test Top-1 保存正式权重。该结果明确标为
  `selection_biased=true`；这是本项目按老师要求采用的口径，不是独立无泄漏测试。

## 2. 三套架构

### 主方法：光学 Router + MoE + 同尺度融合

冻结 Qwen3-VL-Embedding-2B 原始参数，但保留其 processor、图像 patch/位置处理、
文本 tokenizer/embedding 和前端特征。Vision 与 Language 各执行两次光学特征阶段：

1. 224×224 输入振幅与一张 224×224 Router 相位同面加载；传播 10 cm；
2. 在固定 478×478 CCD 的四个 59×59 区域积分，得到四专家分数；
3. `power_l2 + straight-through Top-2` 选择两个专家；
4. 在不改变 478 ROI 的 2×2 专家版面完成 expert 光传播；
5. CCD 读出形成第一段光学特征；
6. 同一条路由供全局相位阶段复用，再传播并完成第二次 CCD 读出。

每个融合点的电子特征 `E` 与光学特征 `O` 先按样本、在所有有效 token 和通道上做 RMS 对齐：

```text
rE = stopgrad(RMS(E)); rO = stopgrad(RMS(O))
M  = (1-alpha) * E/rE + alpha * O/rO
F  = rE * M / stopgrad(RMS(M))
```

因此电子支路的显式权重是 `1-alpha`，光支路是 `alpha`，不会仅给光支路乘权重后再被
电子数值范围淹没。四个融合门独立学习，范围 `[0.01, 0.95]`，初值 0.055。
Vision/Language 各增加一次 Router 曝光，因此一条样本共有 6 次物理 CCD 捕获
（2 次 Router + 4 次特征）。

### Baseline A：激活参数量匹配的 dense D2NN

删除 Router 和空间专家分派。Vision 与 Language 各有两个 224×224 dense 相位阶段；
每一阶段后仍保留与主图一致的 CCD→电子归一化→重载边界和同尺度融合，从而只替换
“MoE 稀疏相位”这一因素，不删除两个网络注入位置。

这里是 dense O/E/O D2NN 对照，而不是两相位面之间完全无探测器的连续 D2NN。
如果论文还要比较连续 D2NN，应建立新 profile，不能与本项混称。

参数匹配口径严格限定为专家相位：

| 口径 | 每个模态 | Vision + Language |
|---|---:|---:|
| 主方法实际激活的 Top-2 专家 | 2×224² = 100,352 | 200,704 |
| dense D2NN 两层 | 2×224² = 100,352 | 200,704 |

主方法还含 Router 相位和全局相位，它们会单独报告，不能声称“总参数量相等”。

### Baseline B：冻结 Qwen3-VL-Embedding-2B

不训练、不微调，不接光学学生；直接使用官方冻结 embedding 前端产生 64 维检索特征，
同样在固定 gallery/test 上评估一次。其 trainable parameter 数为 0。

## 3. 正式运行

所有命令从仓库根目录执行。默认 run 写入本任务 `runs/simulation/`；建议正式运行显式
写 `--run-dir`，并保留 seed。

```powershell
python -m LightGenV2.tasks.t01_object_retrieval.run `
  --profile main --phase all --seed 42 `
  --run-dir LightGenV2/tasks/t01_object_retrieval/runs/simulation/moe_router_scale_seed42

python -m LightGenV2.tasks.t01_object_retrieval.run `
  --profile d2nn --phase all --seed 42 `
  --run-dir LightGenV2/tasks/t01_object_retrieval/runs/simulation/d2nn_matched_seed42

python -m LightGenV2.tasks.t01_object_retrieval.run `
  --profile qwen --phase evaluate --seed 42 `
  --run-dir LightGenV2/tasks/t01_object_retrieval/runs/simulation/qwen_frozen
```

汇总三组结果：

```powershell
python -m LightGenV2.tasks.t01_object_retrieval.report `
  --main LightGenV2/tasks/t01_object_retrieval/runs/simulation/moe_router_scale_seed42 LightGenV2/tasks/t01_object_retrieval/runs/simulation/moe_router_scale_seed43 LightGenV2/tasks/t01_object_retrieval/runs/simulation/moe_router_scale_seed44 `
  --d2nn LightGenV2/tasks/t01_object_retrieval/runs/simulation/d2nn_matched_seed42 LightGenV2/tasks/t01_object_retrieval/runs/simulation/d2nn_matched_seed43 LightGenV2/tasks/t01_object_retrieval/runs/simulation/d2nn_matched_seed44 `
  --qwen LightGenV2/tasks/t01_object_retrieval/runs/simulation/qwen_frozen
```

汇总器逐 run 保留原始值，并给主方法和 D2NN 输出三次重复的 mean±sample std；冻结 Qwen
是确定性单次基线，标准差记为 0。

## 4. 每个 run 必须保留的证据

- `config.yaml`：完全解析后的实际参数；
- `run_manifest.json`、`environment.json`：命令、seed、Git commit、环境；
- `parameter_fairness_contract.json`、`student_architecture.json`：参数口径和计算图；
- `train_log.csv`、`metrics/ema_best_observed_test.json`：训练与选权重依据；
- `ema_best_observed_test_checkpoint.pt`：正式学生权重；
- `student_metrics.json`、`retrieval_results.csv`、`confusion_matrix.png`：最终结果；
- `best_optical_artifacts/` 与周期相位快照：相位变化证据。

当前正式单次结果（seed 42）：主方法 Top-1 91.0%，激活专家相位参数匹配 D2NN 90.5%，
冻结 Qwen3-VL-Embedding-2B 99.5%。完整协议、参数口径、alpha、路由集中度、权重 SHA
及限制见 [`reports/FORMAL_RESULTS.md`](reports/FORMAL_RESULTS.md)。在补齐独立重复前必须
标为 single run，不得混用历史目录中协议不同的 81%、83% 或 90.5% 数字。
