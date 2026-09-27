# 时间一致性四组：第一轮正式仿真结果

历史作废轮（schema=3）：用户要求恢复2250训练/test-best并重新训练，不用于部署；该轮权重及派生工程按授权清理，本报告保留最小历史证据。当前协议schema=4见任务README。

run：`bounded_dc30_ccd_v2_s163_uuid2456_dcfix_20260927`。
训练源码：`0a114ee2d9a5a169263d7e48d2fd14b21afa8d57`。
四组均完成100 epoch，seed=163，相同随机初始化、架构和MOS损失；权重以验证集SRCC选择。
数据：实际训练1800、验证450、原测试558视频。原测试有历史选模使用，不能称全新未触碰测试集。
统一评价为8 μm设备网格、相干直流名义系数0.30、CCD pilot noise scale=1；三类像素平移均关闭。
这是回归仿真结果，不是分类准确率或光路实测。噪声参数尚未由实际CCD定标。

| 模型 | 数据划分 | SRCC ↑ | PLCC ↑ | RMSE ↓ | MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| 基础部署 r0_post | train | 0.922045 | 0.928693 | 5.442982 | 4.207130 |
| 基础部署 r0_post | validation | 0.753607 | 0.740257 | 9.242661 | 6.834661 |
| 基础部署 r0_post | test | 0.755502 | 0.758382 | 9.353483 | 7.094135 |
| +CCD r1_ccd_post | train | 0.895042 | 0.902971 | 6.829763 | 5.238056 |
| +CCD r1_ccd_post | validation | 0.759394 | 0.756666 | 9.239942 | 6.959898 |
| +CCD r1_ccd_post | test | 0.767550 | 0.770132 | 9.784726 | 7.384409 |
| +相干DC r2_ccd_dc_post | train | 0.947837 | 0.957194 | 3.960539 | 3.015596 |
| +相干DC r2_ccd_dc_post | validation | 0.763152 | 0.757357 | 9.254848 | 6.747177 |
| +相干DC r2_ccd_dc_post | test | 0.768230 | 0.770451 | 9.063284 | 6.736914 |
| +训练内插值 r3_ccd_dc_intrain | train | 0.879216 | 0.887639 | 6.799563 | 5.292718 |
| +训练内插值 r3_ccd_dc_intrain | validation | 0.759664 | 0.763925 | 9.278728 | 7.180965 |
| +训练内插值 r3_ccd_dc_intrain | test | 0.769354 | 0.782957 | 9.423891 | 7.283479 |

## 可汇报的结论与限制

测试SRCC累计从0.755502升至0.769354（+0.013851），主要增量来自CCD训练（+0.012047）。
后两步SRCC增量分别仅+0.000681和+0.001123，单seed不足以判定稳定收益或统计显著性。
训练内插值组测试PLCC最高，但DC+CCD、训练后插值组RMSE/MAE最低。不能画成所有指标严格单调改善。
CCD训练组虽提升相关性，RMSE/MAE比基线更差：排序鲁棒性与绝对MOS误差不是同一目标。
建议展示四组测试SRCC/PLCC点图与RMSE/MAE小表；没有多seed重复时不编造误差棒。
真正部署结论仍需四组相同曝光、振幅量化/LUT/ROI条件的光路采集验证。

## 可追溯交付

所有指标原始证据在run的comparison.json及各组final_train/final_validation/final_test/evaluation.json。
supervisor标记complete，所有子进程exit_code=0，owned_gpu_pids_remaining为空。
完整run与权重通过SHA256SUMS逐文件核验同步本地；大文件不进入Git。
四份独立工程用build_projects.py从同一源码生成，携带各组新best_checkpoint.pt；旧导师PT仅作teacher_reference。
