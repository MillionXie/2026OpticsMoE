# T03 显著性分析（SALICON）

本目录固定比较三组系统，三组共享同一数据清单、输入尺寸和指标实现：

1. `main_dc20`：光学 Router、4 专家、Top-2、Vision 两层光电网络、同尺度凸融合、20%–30% 相干零级分量与位置/读出噪声。
2. `d2nn_dc20`：不含 Router 的普通两层 D2NN；两张 `224×224` 相位等于主方法每样本实际激活的两位专家参数量。
3. `qwen_pending`：冻结 `Qwen3-VL-Embedding-2B` 大模型 baseline。按要求，本机/共享服务器不测性能、速度和功耗，只生成 5090D 待测合同。

输出是 `224×224` 连续显著性概率图。主指标为 `CC`（越大越好），并同时报告 KLD、SIM、NSS、AUC-Judd 和 MAE。

## 数据协议

- train：SALICON 2015r1 官方 train2014，10,000 张。
- public test：官方 val2014，5,000 张；本项目将它作为可复现测试集。
- validation：无。
- 官方 test：真值不公开，不用于本地指标。
- epoch 1、每 5 epoch、末轮测试；以最高 public-test CC 保存 `best_checkpoint.pt`。
- 正式目录只保留 `best_checkpoint.pt` 和 `last_checkpoint.pt`。

## 运行

```bash
python -m LightGenV2.tasks.t03_saliency.run --profile main_dc20 --phase all
python -m LightGenV2.tasks.t03_saliency.run --profile d2nn_dc20 --phase all
python -m LightGenV2.tasks.t03_saliency.run --profile qwen_pending --phase all
```

大模型待测边界、计时和功耗口径见 [BASELINE_5090D_TODO.md](BASELINE_5090D_TODO.md)。
