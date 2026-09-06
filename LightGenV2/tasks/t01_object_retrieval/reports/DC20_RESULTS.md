# Caltech101：光 Router 与普通 D2NN 的 DC20 正式复跑

## 结论

本次复跑使用固定 Caltech101 十类划分、seed 42 和 200 张检索测试图。主方法的
Router 是物理光学 Router，采用 Top-2/4；普通 D2NN 基线按定义不含 Router。两种
光电系统均使用同尺度凸融合，并在训练中加入相同的未调制直流分量、CCD 噪声、
位置扰动、k 空间限制和 phase-DC 正则。

| 方法 | Router | Top-1 | Top-3 | MRR | 选中 epoch | 物理 CCD 次数/样本 |
|---|---|---:|---:|---:|---:|---:|
| 光 Router + MoE | 光学，Top-2/4 | **90.0%** | 96.5% | **93.44%** | 25 | 6 |
| 参数匹配的普通 D2NN | 无 Router | 89.5% | **97.0%** | 93.49% | 15 | 4 |
| 冻结 Qwen3-VL-Embedding-2B | 不接光学系统 | 99.5% | 100.0% | 99.75% | 不适用 | 0 |

主方法相对 D2NN 的 Top-1 提高 0.5 个百分点。该结果是单个 seed；训练期间按
epoch 1、每 5 epoch 及最终 epoch 周期性测试，并按最高 EMA test Top-1 选权重，
因此明确属于 `selection_biased=true`，不可描述为独立封存测试。

## 公平性与鲁棒条件

- 两个模态实际激活的专家相位参数总数与 D2NN dense phase 参数总数均为
  `200,704`；主方法额外具有 Router 相位与两张 global phase，需单独报告，不能
  宣称两套系统的总参数量相同。
- 主方法 Vision 和 Language 各执行一次光 Router CCD、一次 expert CCD 和一次
  global CCD，共 6 次物理采集。D2NN 每个模态执行两次 feature CCD，共 4 次。
- 训练时振幅 SLM 与相位 SLM 分别注入 20%–30% 强度占比的相干未调制分量；该
  扰动只在训练态启用，表中周期测试是在无随机扰动的确定性仿真条件下完成。
- 共同扰动还包括截断偏置高斯 CCD 噪声、输入/相位/CCD 最大 ±16 pixel 位移、
  0.65° k 空间限制和 phase dropout；phase-DC 正则权重为 0.005。
- 主方法使用负载均衡损失 0.08、importance 损失 0.02、实际进入总 loss 的
  hard-load 损失 0.50，且 Router 的 load-bias 更新率为 0。最终训练 epoch 的
  Vision 四专家选择计数范围为 1302–1340，Language 为 360–2640；两边都不存在
  从未选中的专家，但 Language 仍明显集中，因此结论是“消除未使用专家并改善
  均衡”，不是“四专家完全均匀”。
- 融合先把电子特征与光学特征按样本 RMS 对齐，再计算
  `(1-alpha)E + alpha O`。主方法四个融合点的最终 alpha 为 5.35%–5.51%，融合后
  两支路数值尺度一致，不会因原始电子 RMS 较大而直接淹没光学支路。

## 证据位置

- 机器可读逐 run 数据：`dc20_comparison/comparison.json`
- 平铺表格：`dc20_comparison/comparison.csv`
- 汇总表格：`dc20_comparison/comparison_aggregate.csv`
- 论文图：`dc20_comparison/comparison_top1.png` 和 `.pdf`
- 服务器正式权重：
  `LightGenV2/tasks/t01_object_retrieval/runs/simulation/moe_router_scale_dc20_strict_seed42/`
  与 `d2nn_matched_dc20_seed42/`。每个目录仅保留 `best_checkpoint.pt` 和
  `last_checkpoint.pt`。
