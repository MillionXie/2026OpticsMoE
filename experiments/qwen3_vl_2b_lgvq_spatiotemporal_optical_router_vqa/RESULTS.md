# LGVQ 空间/时间质量与光路由结果（2026-09-02）

## 1. 结论和实验边界

本实验只预测 `spatial_quality` 与 `temporal_quality`；图文一致性从
manifest、Qwen cache、模型输出头、loss、metric 和 checkpoint 中全部禁用。
在同一套四层光电特征网络中，光路由 O2 得到最高平均 SRCC `0.6446`，高于
最好的电子路由 E1 `0.6378` 和同为 Top-2 的电子路由 E2 `0.6343`。这说明
“四个 CCD 能量区产生专家权重”的仿真方案是可行的，但当前只是单 seed、
test-guided 选模结果，不能解释成无偏泛化估计。

时间质量明显比空间质量容易：所有组的 temporal SRCC 约为 `0.775–0.779`，
而 spatial SRCC 约为 `0.486–0.510`。O2 的空间排序相关性最好，但空间 RMSE
更高，说明其排序能力优于绝对分数标定；后续若用于 MOS 数值报告，应单独做
单调回归/标定，不能只看 SRCC。

> 当前 1024/986 parallel16 光平面是仿真对照，不能直接宣称适配现有
> 518/478 实验光路。硬件版本需要顺序复用 518/478 平面，或重新设计更小
> lane 后重训。

## 2. 固定实验合同

- 数据：2,808 个视频，468 个 prompt group，每组 6 个生成器视频。
- 划分：375 groups / 2,250 train；93 groups / 558 test；validation=0。
- 帧：每个视频取 10%、37%、63%、90% 四帧。
- 冻结 Qwen cache：Vision `[2808,4,196,1024]`；Language `[1,30,2048]`。
- Prompt：`Please evaluate the quality of this video and rate it using one of the following five levels: Excellent, Good, Fair, Poor, or Bad.`
- 训练：50 epoch；epoch 1、每 5 epoch、最终 epoch 测试。
- 选模：最大化 `mean(SRCC_spatial, SRCC_temporal)`；明确接受 test 选模。
- 光电融合：四层独立 `alpha∈[0.01,0.49]`，初值 `0.055`。
- 环境：Python 3.12.3，PyTorch 2.8.0+cu128，Transformers 4.57.6，
  CUDA 12.8，NVIDIA GeForce RTX 5090 D。

## 3. 主结果

以下均为各组周期测试选出的最佳 checkpoint，再对完整 558 test 重评：

| Variant | Router / Top-k | Best epoch | Spatial SRCC | Spatial PLCC | Spatial RMSE | Temporal SRCC | Temporal PLCC | Temporal RMSE | Mean SRCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | electronic / 1 | 10 | 0.4997 | 0.5423 | **9.6809** | 0.7759 | **0.7999** | **8.2807** | 0.6378 |
| E2 | electronic / 2 | 10 | 0.4936 | 0.5354 | 9.7624 | 0.7750 | 0.7927 | 8.5615 | 0.6343 |
| E4 | electronic / 4 | 15 | 0.4857 | 0.5301 | 10.0755 | 0.7759 | 0.7849 | 8.7103 | 0.6308 |
| O2 | optical energy / 2 | 20 | **0.5097** | **0.5636** | 11.5303 | **0.7794** | 0.7931 | 8.4978 | **0.6446** |

O2 相对同 Top-2 的 E2，平均 SRCC 绝对提高 `0.0103`；相对最佳电子组 E1
提高 `0.0068`。差距不大，当前证据支持“光路由可比较且略优”，不支持宣称
显著超过电子路由。E1/E2/E4 没有表现出“激活专家越多越好”，本次单 seed
排序为 E1 > E2 > E4。

## 4. 参数量

| Variant | 总可训练参数 | 特征 phase | Router | 其他电子 |
|---|---:|---:|---:|---:|
| E1/E2/E4 | 4,023,422 | 2,204,200 | 1,576 | 1,817,646 |
| O2 | 4,272,726 | 2,204,200 | 250,880 | 1,817,646 |

O2 比电子 router 多 249,304 个参数，来源是 Vision 四 lane 和 Language 的
光路由 phase，不是额外电子打分头。

## 5. 光电融合与 phase 是否训动

最佳 checkpoint 的四层 alpha 都停在约 `0.0528–0.0544`，说明在允许范围内
模型仍偏好低光占比。融合前 `RMS(M)` 约为 `0.944–0.960`，所以公共后缩放
`F` 实际只乘约 `1.04–1.06`，不是大幅重写特征。

与相同随机种子初始化相比，最佳 checkpoint 的特征 phase 环形差值 RMS 为：

| Variant | Feature phase Δ RMS (rad) | Feature phase std (rad) | Optical-router phase Δ RMS (rad) |
|---|---:|---:|---:|
| E1 | 0.560 | 0.605 | N/A |
| E2 | 0.562 | 0.608 | N/A |
| E4 | 0.653 | 0.693 | N/A |
| O2 | 0.603 | 0.647 | 0.773 |

因此 feature phase 与 O2 router phase 都有实质更新，不是只训练电子头。
O2 的训练 router capture loss 由约 `0.75` 降到约 `0.68`；最佳 checkpoint
重评时，Vision/Language 四探测区合计捕获各自有效平面能量约 `33.4%`。

路由诊断也暴露了限制：O2 Vision 在整个 test 上四专家累计入选份额均为
25%，但 Language 固定集中在前两个专家；电子 E1/E2 也出现固定专家或固定
专家对的塌缩。当前任务分数可用，但若老师关心“输入自适应路由”，下一步应
额外约束每样本路由多样性并绘制逐样本选择矩阵，不能只看总体平衡。

## 6. `M` 后面的 `F` 是什么，能否删除

对每个样本的有效 token/channel 定义：

```text
rE = RMS(E), rO = RMS(O)
En = E/rE, On = O/rO
M  = (1-alpha)En + alpha On
F  = rE M / RMS(M)
```

`M` 决定融合后的方向和光/电相对内容。若
`rho = mean(En*On)`，则：

```text
RMS(M)^2 = (1-alpha)^2 + alpha^2 + 2 alpha (1-alpha) rho
```

即使 E/O 已经同尺度，向量相加仍可能因相关性产生几何缩短或抵消。`F` 只
对整个 `M` 乘同一个正标量，不改变方向和光/电相对系数；它让
`RMS(F)=rE`，保持下一层看到的尺度与原电子残差一致，并保证 `alpha=0` 时
精确退化为 `F=E`。它没有可训练参数，RMS 统计也 detach。

`F` 不是数学上不可删除，但当前后续仍有 residual identity、Linear 和
Softplus 光场编码，并非严格尺度不变，因此不能假定删掉无影响。公平消融也
不能直接令输出为 `M`，因为那会使 `alpha=0` 变成 `E/rE`。应使用：

```text
F_no_post = rE * M
```

它只删除 `/RMS(M)`，同时保留 `alpha=0 -> E`。Full 与 No-post 必须从同一
初始化分别重训；直接拿 Full checkpoint 切公式会造成分布突变，不是公平
性能比较。本轮未执行该专门消融，所以这里只报告作用机制，不捏造删除后的
性能数值。

## 7. 与师姐工程的关系

本工程以我们的四层顺序和融合为主，只参考师姐代码的四帧采样、prompt-group
划分、特征缩小和并行组织。没有继承她的 alignment 目标、hashed prompt、
generator one-hot、SPAQ 强依赖或 Transformer temporal head。师姐服务器副本
缺少其引用的 checkpoint/cache/SPAQ 资产，故目前只能做结构比较，不能提供
可复现的数值 baseline。

## 8. 证据位置

服务器结果根目录：

```text
/root/autodl-tmp/workspace/opticsmoe/experiments/
qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa/runs/
```

每个 `lgvq_e1/e2/e4/o2` 目录包含：

- `training_summary.json`、`metrics_best_observed_test.json`
- `test_metrics.json`、`test_predictions.csv`
- `fusion_diagnostics.json`、`router_diagnostics.json`
- `parameter_breakdown.json`、`resolved_config.json`、`preflight.json`
- `best_observed_test_checkpoint.pt`、`last_checkpoint.pt`

代码、配置和命令见 [ARCHITECTURE.md](ARCHITECTURE.md) 与
[RUN_COMMANDS.md](RUN_COMMANDS.md)。

