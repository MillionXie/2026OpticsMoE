# 2026-09-08 复现结果与优化状态

## 已完成：固定权重复评

新生成全部val2014密度图，完整评估5000张，没有只挑部分样本。

| 指标 | 本次复评 |
|---|---:|
| CC | 0.8810325132 |
| 独立NumPy float64 CC | 0.8810325100 |
| KLD | 0.1034256022 |
| SIM | 0.8327102114 |
| NSS | 0.9902881276 |
| AUC（仓库实现） | 0.7730450296 |
| MAE（仓库实现） | 0.0680666424 |

与历史5090D的0.88105177仅差0.00001926。可以确认**该固定checkpoint在本仓库协议下确实约0.8810**，
不能扩大解释成从头训练已复现、零样本大模型成绩或官方隐藏测试成绩。
实际执行设备RTX3090，PyTorch2.6.0+cu124；本次未测速度/功耗。
机器可读证据见 [原始复评JSON](evidence/baseline_recheck_20260908.json)，含权重/模型/标注SHA256。
原始JSON的sigma简写有歧义：实际执行的代码是19**原图像素**按宽高缩放，不是224图上固定sigma19，
详细标签口径见README。原始证据文件不改写。

服务器证据根目录：
`/DATA/DATA1/guest3/2026OpticsMoE/LightGenV2/tasks/t03_saliency/runs/simulation/baseline_recheck_20260908/`。
其中有逐图 `per_image_cc.csv`、`environment.txt`、配置及新生成的标签缓存。
源代码启动时为8207fa99；运行期间仅本任务命令/文档元信息修订，结果JSON记录结束时7ef8566f。
计算图与指标实现未变。后续复现应固定同一commit直至完成。

## 光电原版与公平性

原版checkpoint实际重新前向得到CC=0.829059，与历史0.8291一致。
相比本次baseline绝对差约0.05197，相对低5.90%。旧D2NN CC=0.83456。
原checkpoint两层alpha分别0.05249681、0.05187522；这是同尺度融合系数，不是准确率或能量贡献比例。
两方数据、密度生成和CC算法一致，但原生Qwen24层视觉主干/读出头与光电两层/读出头均不同；
这属于系统对比，不是总参数量或训练预算完全相同的模块消融。

## 上轮训练状态（新alpha对照见下方链接）

- `baseline_retrain_seed42_20260908`：已完成；新初始化显著性头训练30epoch，Qwen保持冻结，CC=0.87899109，best epoch30。
- `moe_dc20_mean_only_continue_seed42`：去掉CCD对数/上限裁剪，关闭16px错位扰动，原best迁移初始化，100epoch。
- `moe_dc20_mean_only_cc_continue_seed42`：同上，CC loss权重0.5→1.0。

两个新候选保留光学router、Top2、20%–30%随机相干零级分量、同尺度凸融合与专家均衡。
不增加attention/Transformer/VGG或额外支路。只有best/last，不生成周期PT。
由于CCD归一化变更，不能拿旧0.8291冒充新候选的初始/最终成绩。新候选应独立查看epoch0和最佳epoch。
初期退化不能证明最终失败，同样启动训练也不能证明已经改善。
最初mean_only候选epoch0 CC=0.828469，第一轮约0.8219；后续已提升至epoch75的0.84879587，
检查时训练尚未结束。CC加倍候选约0.8476。原0.8291权重未覆盖。
新的同读出头与alpha≥0.4对照见 [实验说明](ALPHA_AND_HEAD_COMPARISON.md)。

日志统一在 `runs/simulation/launch_reproduction_20260908/`：
`baseline_retrain.log`、`mean_only.log`、`mean_only_cc.log`。
完成后以每个run的 `selected_checkpoint_test_evaluation.json` 为最终结果。最新评估入口支持记录alpha；
已启动进程若未加载新增报告字段，可由checkpoint的两项 `hybrid.block*_optical_fusion_logit`
按 `0.01 + 0.94*sigmoid(raw)`读取。
