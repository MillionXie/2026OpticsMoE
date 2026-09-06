# LSP 指标口径核查

核查日期：2026-09-06。结论是“LSP 光学模型曾达到约 0.7”这个记忆有来源，但它与当前
正式光 Router 方案不是同一架构，不能直接写在当前主方法一栏。

| 版本 | PCK@0.2 | PCKh@0.5 | NME | 是否符合当前主方法合同 |
|---|---:|---:|---:|---|
| 旧 `vision2_hybrid` | **0.7131** | 0.8375 | 0.2434 | 否 |
| 早期 17 µm 光 Router Top-2 | **0.6024** | 0.7539 | 0.3282 | 部分符合 |
| 当前 DC20 正式光 Router Top-2 | **0.5773** | 0.7363 | 0.3488 | 是 |
| 当前参数匹配 D2NN | **0.6751** | 0.8054 | 0.2736 | baseline |

## 为什么旧版 0.7131 不能替代当前结果

旧 `lsp_pose_vision2_hybrid` 所在父工程的名字虽然带有 `moe16`，但开启
`vision2_hybrid.enabled` 后，实际解析出的模型已经覆盖为 478×478 有效面、4 专家
Top-2；不能把父配置中的 986×986、16 专家 Top-4 当成该次 0.7131 结果的真实架构。
旧版使用 `router.gate.weight/bias` 电子线性 gate，而当前版使用可上光路的相位 Router。
旧版默认像素间距为 16 µm，当前正式硬件合同为 17 µm。

两版的电子二维 mixer **并没有强弱差异**：均为宽度 192、2 层、3×3 depthwise 2-D
卷积、1×1 等价的通道线性混合以及 2 倍扩张 MLP。两版 mixer 参数均为 78,336，
FFN 参数均为 296,064，姿态热图读出头也均为 133,425。因此旧版 0.7131 不能解释为
“用了更强的 mixer”。真正变化来自电子/光 Router、普通/同尺度凸融合、是否加入
20%–30% 未调制分量、扰动强度以及训练配置；这些因素必须分别消融。

证据文件：

```text
experiments/qwen3_vl_embedding_2b_lsp_pose_optical_moe16/
  runs/lsp_pose_vision2_hybrid/metrics/student_inference.json
```

## 为什么还能找到 0.6024

`optical_power_topk2_periodic_test5` 已经是 17 µm、478×478、光 Router Top-2，epoch 100
达到 PCK 0.6024。但它没有采用当前正式 DC20–30% 未调制光条件，因此只适合作为历史
过渡结果，不能与当前 DC20 主结果或其 D2NN baseline 直接替换。

证据文件：

```text
experiments/qwen3_vl_embedding_2b_lsp_pose_optical_router/
  runs/optical_power_topk2_periodic_test5/metrics/periodic_test_epoch_0100.json
```

## 当前论文可用口径

当前公平对照必须使用 `reports/dc20_comparison/RESULTS.md`：光 Router Top-2 的
PCK 为 0.5773、PCKh 为 0.7363；matched D2NN 的 PCK 为 0.6751。若后续希望恢复到
0.60 以上，应在相同 DC20、数据划分和 17 µm 光路下重新训练，而不是回填旧 0.7131。

若要测试更强的电子 mixer，应建立新的显式消融，而不是“移植旧 mixer”：例如保持
宽度 192，将每个光电融合阶段后的电子残差块由 1 个增加为 2 个。主方法与 matched
D2NN 必须使用完全相同的增强 mixer 和姿态头，并补做同一主方法 checkpoint 的
`optical-off` 推理，避免把电子容量提升误记为光学收益。
