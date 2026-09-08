# T03 显著性分析（SALICON）

本目录固定比较三组系统，三组共享同一数据清单、输入尺寸和指标实现：

1. `main_dc20`：光学 Router、4 专家、Top-2、Vision 两层光电网络、同尺度凸融合、20%–30% 相干零级分量与位置/读出噪声。
2. `d2nn_dc20`：不含 Router 的普通两层 D2NN；两张 `224×224` 相位等于主方法每样本实际激活的两位专家参数量。
3. 冻结 `Qwen3-VL-Embedding-2B` 视觉主干 + 训练过的显著性头 baseline。历史CC约0.88105；现增加不限定显卡的性能复现入口，速度/功耗仍只在5090D测。`qwen_pending`仅保留旧待测合同入口。

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

## 正式单次结果（seed 42）

- 光 Router Top-2：CC 0.8291、KLD 0.1330、SIM 0.8063、NSS 0.9283、
  AUC-Judd 0.7631、MAE 0.0890。
- 参数匹配 D2NN：CC 0.8346、KLD 0.1296、SIM 0.8092、NSS 0.9344、
  AUC-Judd 0.7643、MAE 0.0884。
- 光 Router 四专家选择占比为 23.54% / 26.80% / 23.38% / 26.28%，有效专家数
  3.985/4，无未使用专家。

本次单 seed 下 D2NN 的 CC 高 0.0055，必须作为真实负差距保留。机器可读指标、相位图和
样例见 `reports/dc20_comparison/`。Frozen Qwen的架构、公平性、固定权重复评、从头训练命令与
本轮训练优化配置集中在 [复现说明](reports/reproduction/README.md)。
