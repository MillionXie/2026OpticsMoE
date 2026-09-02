# LGVQ 视频质量评价：O2-109 高光占比实验

本工程只回答一个问题：在不改变现有 `17 um、10 cm、518×518` 光路的前提下，固定使用 **Top-2 全光路由**，光学分支的融合占比最多能提高到多少，同时仍保持可接受的视频质量评价性能。

本工程不是新的 Router 大矩阵消融：

- 只保留光路由，Vision 和 Language 都由 CCD 四区域能量产生路由分数；
- 固定 Top-2，不再训练或比较 Top-1、Top-4；
- 不再运行电子 Router；
- 不再运行 LSP，当前唯一正式任务是 LGVQ 的 Spatial/Temporal 视频质量评价；
- 图文一致性标签和输出被禁用。

## 一句话结构

四帧视频经冻结的完整 `Qwen3-VL-2B-Instruct` Vision 主干与 learned main merger 得到 `4×196×2048` 特征，固定提示词经完整冻结 Language 主干得到 `L×2048` 特征；两者投影到 192 维后依次经过 Vision expert、Vision global、Language expert、Language global 四个光电融合层，最后由独立的 Spatial 与 Temporal 电子读出头输出两个 MOS。

Qwen Vision 的 main merger 保留；DeepStack 不使用。后续学生网络不增加 Transformer attention。

## 固定硬件合同

| 项目 | 固定值 |
|---|---:|
| 仿真面阵 | `518×518` |
| 有效光场 | 居中的 `478×478` |
| 四帧并行 lane | 每个 `232×232`，lane 间隔 14 px |
| 单专家相位/振幅 tile | `109×109` |
| 专家 pitch | 123 px，专家间隔 14 px |
| 像素尺寸 | `17 um` |
| 波长 | `532 nm` |
| 传播距离 | `10 cm` |
| Router | Optical Top-2，power-L2 权重，corrected STE |

`109×109` 专家被直接放入现有 478 有效区域，不扩大 SLM、ROI 或传播面。因此本工程修复了旧视频实验超过实际硬件尺寸的问题。

## alpha 梯度的含义

这些配置不是不同 Router，而是同一个 O2 光路由模型的光占比压力测试：

| 配置 | 每层 alpha 范围 | 初值 | 目的 |
|---|---:|---:|---|
| `alpha20.yaml` | `[0.20, 0.90]` | 0.30 | 优先保证性能的下界 |
| `alpha35.yaml` | `[0.35, 0.90]` | 0.42 | 中等高光占比 |
| `alpha50.yaml` | `[0.50, 0.90]` | 0.57 | 强制至少一半归一化混合系数来自光分支 |
| `alpha65.yaml` | `[0.65, 0.90]` | 0.70 | 在 0.50 达标后继续搜索更高光贡献上限 |

最终选择规则是：先满足 [RESULTS.md](RESULTS.md) 中的性能目标，再选择 alpha 下界最高的版本。alpha 是 RMS 配平后两分支的混合系数，不等同于未经归一化的原始光功率比例；正式报告必须同时给出四层最终 alpha、电子 RMS、光学 RMS 和融合后 RMS。

## 数据与评估

- LGVQ：2,250 个训练视频，0 validation，558 个固定 test 视频；
- 每个视频取 10%、37%、63%、90% 四帧；
- 固定提示词：`Please evaluate ... Excellent, Good, Fair, Poor, or Bad.`；
- 每 5 epoch 测一次 test，epoch 1 和最后一轮也测试；
- 按 `mean(Spatial SRCC, Temporal SRCC)` 保存 `best_observed_test_checkpoint.pt`；
- 另存十项指标全部达到老师参考线的 `best_reference_compliant_checkpoint.pt`；
- 完整报告 SRCC、KRCC、PLCC、RMSE、MAE；
- 用户已明确接受使用周期 test 选 checkpoint，本工程不划分 validation。

架构细节见 [ARCHITECTURE.md](ARCHITECTURE.md)，服务器命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)，指标与填表规则见 [RESULTS.md](RESULTS.md)。

