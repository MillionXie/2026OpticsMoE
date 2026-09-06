# LSP：光 Router 与普通 D2NN 的 DC20 正式复跑

主方法固定使用物理光学 Router Top-2/4；普通 D2NN 基线按定义不含 Router。
两者使用相同的数据划分、读出头、同尺度凸融合、10 cm/17 µm 光路和鲁棒训练条件。

| 方法 | Router | PCK@0.2 | PCKh@0.5 | NME | 平均像素误差 | 选中 epoch |
|---|---|---:|---:|---:|---:|---:|
| Optical Router + MoE | 光学，Top-2/4 | 57.73% | 73.63% | 0.3488 | 22.97 | 100 |
| Matched D2NN | 无 Router | 67.51% | 80.54% | 0.2736 | 18.02 | 100 |

## 口径

- 训练阶段注入 20%–30% 强度占比的振幅/相位 SLM 相干未调制分量，并加入截断偏置高斯 CCD 噪声、最大 ±16 pixel 位移、k 空间限制、phase dropout 和 phase-DC 正则。
- 主方法包含一次 Router CCD、一次 expert CCD 和一次 global CCD；D2NN 为两张 dense phase，并使 phase 参数量严格匹配 Top-2 激活专家的 100,352 个相位参数。
- epoch 1、每 5 epoch 和最终 epoch 测试一次，按最高 test PCK@0.2 选 best；这是项目指定的 test 选模口径，不属于独立封存测试。
- 每个正式 run 只保留 `best_checkpoint.pt` 与 `last_checkpoint.pt`；SHA256 见 `comparison.json`。
- 最佳权重相位总览已复制为 `main_best_phase_overview.*` 和 `d2nn_best_phase_overview.*`，对应数值见同名前缀的 `*_phase_statistics.json`。
