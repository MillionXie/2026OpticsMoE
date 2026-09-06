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

旧 `lsp_pose_vision2_hybrid` 使用 8 µm 像素、986×986 有效面、16 专家 Top-4，并由
`router.gate.weight/bias` 电子线性 gate 完成路由；它还有较强的电子二维 mixer。该版本
虽然包含光学分支，但不符合现在固定的 17 µm、478×478、4 专家、物理光 Router Top-2
合同，也没有当前明确的 20%–30% 未调制分量条件。

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

