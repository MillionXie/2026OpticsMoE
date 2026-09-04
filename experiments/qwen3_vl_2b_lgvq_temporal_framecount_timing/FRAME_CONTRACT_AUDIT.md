# LGVQ 工程抽帧与 Qwen 复用核查

## 结论

这里需要区分三个概念：视频抽帧数、光学 lane 数、光路传播/曝光次数。后两者增加时，
并不代表又从视频中抽取了更多帧。

| 工程 | 正式视频帧数 | 时间位置 | 实际复用的 Qwen 部分 |
|---|---:|---|---|
| `qwen3_vl_2b_lgvq_spatiotemporal_optical_router_vqa` | 4 | 10%、37%、63%、90% | Vision 只到 patch+position，2×2 无参数均值到 196 token；固定 prompt 经过完整冻结 Language 后缓存 |
| `qwen3_vl_2b_lgvq_o2_109_highalpha_vqa` | 4 | 10%、37%、63%、90% | 完整 24 层 Qwen Vision tower + learned main merger；固定 prompt 经过完整冻结 Language 后缓存 |
| `qwen3_vl_2b_lgvq_single_metric_o2_16frame_54` Temporal 主线 | 16 | 10% 到 90% 等间隔 | 只保留 Qwen patch+position 输入前端和词 embedding，不执行 Vision/Language Transformer |
| 同一 `single_metric` 工程后加入的 compact 分支 | 9 | 10%、20%……90% | 与 16 帧主线相同的 Qwen 输入前端；3×3 lane |

因此：

- `16frame_54` 名称对应 16 帧、4×4 lane、每个 Vision 专家 54×54；
- 该目录里的 `temporal9_compact*.yaml` 是后来加入的另一套 9 帧实验，不是 16 帧模型内部隐藏的布局；
- 3×3 compact 就是 9 个视频帧；router 和四层 feature 光路的额外 pass 不计入抽帧；
- 三个工程不能直接用各自的端到端时间回答“完整 Qwen 随帧数如何变化”，因为它们执行的 Qwen 深度不同。

## 本速度实验为什么统一测 4/9/16 帧

为了只改变抽帧数，本工程固定使用同一份完整 `Qwen3-VL-2B-Instruct`、同一个 Temporal
prompt、同一个 `Linear(2048,1)` 输出边界，只改变抽取帧数。这能把模型复用深度、光路
lane 和读出头差异从速度对比中排除。

4/9/16 帧的采样位置严格沿用上述工程。解码也保留原工程实现：每个目标位置都执行一次
`CAP_PROP_POS_FRAMES` 和 `read()`；没有采用后来写过的顺序解码加速。

## 源视频实际有多少帧

对 manifest 中全部 2,808 个 MP4 用 OpenCV 实测，并非每段都约 96 帧：

| 源文件帧数 | 全集视频数 | train | test |
|---:|---:|---:|---:|
| 8 | 468 | 375 | 93 |
| 16 | 936 | 750 | 186 |
| 24 | 468 | 375 | 93 |
| 36 | 468 | 375 | 93 |
| 96 | 468 | 375 | 93 |

全集与 test 的平均源文件帧数都是 32.667，中位数为 20。16 帧配置表示执行 16 次
时间位置采样并向 Qwen 提供 16 张图；对于只有 8 或 16 帧的源文件，四舍五入后的帧号
可能重复。这是原工程的既定采样合同，本速度测试不擅自去重或改变它。
