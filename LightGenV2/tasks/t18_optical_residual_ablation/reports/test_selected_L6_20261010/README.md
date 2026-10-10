# 六层30%残差：测试集选模开发扫描

2026-10-10用户明确授权直接在test上选PT。本轮没有训练或集成，也未修改数据、光路、残差比例、类别或种子。
**结果为测试集选模的开发成绩，不能称独立泛化或同预算公平消融；原验证选模报告保留。**

最高准确率 **87.80%（367/418）**，宏平均召回88.06%，balanced NLL 0.386227。获选Phase smoothing .05的best_ema，第8轮。
87.80%有三组权重同分，按预先写入manifest的“同分最低test balanced NLL”选择相位平滑组。
比此前验证选定EMA的87.08%高0.72pp；比固定无残差87.56%高0.24pp（仅一个样本净差，不是显著优势）。
89.56%以上需至少375/418=89.71%，当前还差8个正确样本。此值只是已保存权重中的最高值，不能当理论上限。

选定训练设置：lr=0.0003余弦，EMA=0.99，相位平滑=0.05，收光损失=0.2，label smoothing=0.02；九专家、3专家层+3global，router无残差。

| 保存训练臂 | best EMA | last raw | last EMA |
|---|---:|---:|---:|
| 30 epochs | 71.53% | 71.05% | 71.53% |
| 100 epochs, LR .002 | 83.97% | 84.45% | 84.45% |
| 100 epochs, LR .003 | 86.60% | 86.36% | 86.12% |
| Continuation (interrupted) | 86.60% | 86.84% | 86.60% |
| Continuation / migration | 87.56% | 87.80% | 87.56% |
| Continuation round 2 | 86.84% | 87.80% | 87.32% |
| Stronger augmentation | 87.56% | 86.36% | 87.08% |
| EMA .99 | 87.08% | 87.08% | 87.56% |
| Capture loss .05 | 87.56% | 87.56% | 87.56% |
| Phase smoothing .05 | 87.80% | 87.56% | 87.56% |
| Low LR | 87.56% | 87.56% | 87.56% |
| Low LR + smoothing | 86.84% | 87.32% | 87.56% |

扫描12个训练臂、36个状态、33组不同权重。新增27组测试推理，其余复用历史或本轮同状态记录；原中断run明确保留中断身份，不冒充完整训练。
best表示当时验证选中的EMA；last raw表示末轮训练权重；last EMA是同一个last PT中保存的EMA。没有保存逐轮PT，因此不能扫描每一轮。
逐样本ID、标签支持数、argmax、混淆矩阵、正确数及指标一致性均已检查。源码身份校验原已发布Git blob，未改旧PT的sources。

## 权重与证据

- 原checkpoint：`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t18_optical_residual_ablation/runs/simulation/mango_rho03_L6_capture_smooth_20261010/smooth/moe_L6_seed17/best_checkpoint.pt`
- 原checkpoint SHA256：`058b758ce585dde9e9ba092d28feca89465c9444f48be5c087bab5a86ccf565f`
- state key：`model`，state SHA256：`4503adc1a843e648b5fd6238c547472d00fedeff888b9b93c2c0d97bcca2984f`
- 开发选定导出：`LightGenV2/tasks/t18_optical_residual_ablation/runs/simulation/mango_rho03_L6_test_selected_development_20261010/best_checkpoint.pt`
- 导出SHA256：`b825a4f8f5c5addf363cb3c51e388555aa19c696a258f9b6cfd4ed5f7e101d24`
- 源训练版本：`fad46834a`；完整config/sources保留服务器manifest与results。
- manifest SHA256：`ec3acd0f9dce9487d0fa851cea06360b15c4386a45376c801e7496cc4836aaea`
- [36格CSV](test_sweep.csv)、[JSON证据](test_sweep.json)、[比较图](test_sweep.png)、[SVG](test_sweep.svg)、[PDF](test_sweep.pdf)。
- 测试418条，单种子，图像级划分；无误差条。选模反复接触此集合会使最高值乐观，后续独立泛化须用新保留数据。
- 本次仅使用指定物理GPU4，进程完成后退出并释放。无残差和2/4层权重不变。
