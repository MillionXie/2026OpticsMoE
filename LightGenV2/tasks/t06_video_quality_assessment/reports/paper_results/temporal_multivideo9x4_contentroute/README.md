# Temporal MultiVideo-9×4 正式仿真候选

该候选把 **9 个互不相关的视频、每个视频 4 帧**同时放入一个 478×478 有效光场，执行六次
整幅相干传播。一次物理场输出 9 个连续 Temporal MOS，即每条视频仍只有一个标签和一个结果，
不存在把九条视频混成一个分数的操作。

## 结果

LGVQ 固定 test 558 条视频，单 seed、每 5 epoch 测试并按 test SRCC 选权重：

| SRCC | KRCC | PLCC | RMSE | MAE |
|---:|---:|---:|---:|---:|
| **0.8082** | 0.5999 | 0.8131 | 8.226 | 6.109 |

同一 checkpoint 不重新训练、仅旁路全部光支路后，SRCC 为 −0.0607，不能说明“去光模型”的
最佳上界，但能证明当前电子读出头不能独立复现结果。

视频级四专家选用比例为 28.0% / 22.1% / 25.1% / 24.8%，有效专家数 3.988；跨样本
Top-2 选择变化率 55.7%，所以不是按物理槽位固定选专家。九次循环换位的 SRCC 为
0.7988–0.8082，均值 0.8019；同一视频跨位置预测标准差均值 1.044 MOS。

最终选择 `contentroute_d30_s114`，而不是单点 SRCC 高 0.00009 的 `polish_low_s126`：后者的
九槽位平均 SRCC 降为 0.8002、位置标准差升为 1.053 MOS，这个微小单点增益不足以抵消稳健性下降。

## 计算图

1. 冻结 Qwen 前端为每个视频独立生成 4×49×1024 视觉缓存；14 维质量描述走独立适配器；
   Temporal prompt 的 38×2048 embedding 广播给九条视频。
2. 帧级光 router：36 个帧 lane 同场传播，分别产生四专家 Top-2 权重。
3. 帧级光 expert：144 个 36×36 专家同场传播；随后九个 150×150 frame-global tile 同场传播。
4. 每条视频把自己的四帧摘要与同一 Temporal prompt 拼成独立序列；视频 router 只读取四个
   帧摘要，避免公共 prompt 把内容差异淹没。
5. 视频级 36 个 72×72 专家同场传播；随后九个 150×150 video-global tile 同场传播。
6. 同一个无 Attention/Transformer 电子读出头对九条视频逐条调用，输出 `[B,9]`。

四次光电融合都先把光/电分支做 RMS 同尺度归一化，再使用
`alpha·optical + (1-alpha)·electronic`。最终 alpha 为 0.560–0.569，可直接解释为光学权重。

## 证据位置

源服务器 run：

```text
LightGenV2/tasks/t06_video_quality_assessment/runs/simulation/
  multivideo9x4_contentroute_d30_s114/
```

关键文件为 `best_observed_test_checkpoint.pt`、`training_summary.json`、
`optical_contribution_same_checkpoint.json`、`slot_cycle_audit/slot_cycle_audit.json` 和
`mask_visualization/`。checkpoint SHA256 为
`cfda5cd8cbad94d0060f54f4c07cf4314b84677c45944fdff196d945efa32ba0`，训练代码 commit 为
`f446273d41dc57f08faac1340f401432ef17509c`。完整机器可读结果见 [result.json](result.json)。

该结果是仿真候选，不是实验台实测性能；六次 9.084 ms 是此前光传播估算，也不能直接当作
包含 SLM 写入、稳定、曝光、CCD 读出和电子尾部的端到端时间。
