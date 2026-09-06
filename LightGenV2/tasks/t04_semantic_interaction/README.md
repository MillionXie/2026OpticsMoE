# T04 语义交互（OpenMoji）

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

```bash
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile main_dc20 --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile d2nn_dc20 --phase all
python -m LightGenV2.tasks.t04_semantic_interaction.run --profile qwen_pending --phase all
```

大模型待测边界、计时和功耗口径见 [BASELINE_5090D_TODO.md](BASELINE_5090D_TODO.md)。

## 正式单次结果（seed 73）

- 光 Router Top-2：selected checkpoint 正式复评 changed-cell accuracy 0.9800、
  foreground category 0.9895、edit IoU 0.9350、object F1 0.9837、scene exact match 0.8950。
- 参数匹配 D2NN：changed-cell accuracy 0.9895、foreground category 0.9944、
  edit IoU 0.9813、object F1 0.9949、scene exact match 0.9650。
- 主方法语言 Router 使用 3/4 专家，选择占比 50.00% / 25.00% / 25.00% / 0%；
  视觉 Router 也使用 3/4，选择占比 20.30% / 45.80% / 33.90% / 0%。

因此该结果满足“物理光 Router、Top-2”的结构要求，但不能宣称四专家完全均衡；D2NN 的
主指标高 1.00 个百分点。机器可读结果和可视化见 `reports/dc20_comparison/`。
