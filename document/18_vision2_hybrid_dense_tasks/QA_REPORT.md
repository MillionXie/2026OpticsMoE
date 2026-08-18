# 数据与图片质量检查

检查时间：2026-08-18（Asia/Shanghai）。

## 数据完整性

| 任务 | 期望 epoch | 实际记录 | 连续性 | NaN/Inf | 选模一致性 |
|---|---:|---:|---|---|---|
| SALICON | 60 | 60 | 1–60 连续 | 无 | `validation_cc` 最大为 epoch 59 |
| ISIC 2016 | 100 | 100 | 1–100 连续 | 无 | `test_metrics.checkpoint_epoch=96`，与 train loss 最小值一致 |
| LSP | 150 | 150 | 1–150 连续 | 无 | train loss 最小为 epoch 148；test PCK 观察峰值为 epoch 149 |
| Caltech101-10 | 60 | 60 | 1–60 连续 | 无 | total loss 最小为 epoch 54；test Top-1 观察峰值首次为 epoch 42 |

Caltech101 另通过脚本核验：`metrics_best_train_loss.epoch=54` 与 history 最小 total loss 一致，`test_was_not_used_for_selection=true`；全部 epoch 的 `phase_learning_rate=0` 且 `phase_delta_run_rms_rad=0`。`SOURCE_MANIFEST.csv` 记录了 29 个冻结证据文件的服务器绝对路径、字节数和 SHA-256，可用于检查传输后是否发生变化。

## 图片物理尺寸

四张 PNG/TIFF 均为：

- 像素：`4322 × 1275`；
- 分辨率：`600 × 600 dpi`；
- 按文件 DPI 换算：约 `182.97 × 53.98 mm`；
- 目标设计尺寸：`183 × 54 mm`。

因此图片高度为约 5.4 cm，满足 4–6 cm 的要求。

## 字体与矢量检查

- 绘图脚本要求系统真实存在 Arial；缺失时直接失败，不允许静默 fallback。
- 四个 SVG 分别保留 34–39 处 Arial 文字样式，`svg.fonttype=none`，可继续编辑文字。
- 四个 PDF 均嵌入 `ArialMT` 和 `Arial-BoldMT` 子集，`pdf.fonttype=42`。
- 基础字、坐标、tick、legend 和脚注均为 7 pt；面板号为 8 pt。

## 人工视觉检查

- 四张 600 dpi PNG 均已逐张查看；曲线、坐标、图例、面板号和脚注未发现裁切。
- ISIC 的 panel b 数字均在坐标范围内。
- LSP 的 epoch 148 选模线与 epoch 149 红色诊断峰值被明确区分。
- Caltech101 的 epoch 54 选模线与 epoch 42 红色诊断峰值被明确区分；panel b 数值未超出坐标范围。
- 当前版为初步定量图，没有加入样例图、phase mask 或 CCD 图。Caltech101 明确为 simulation-only，没有混入服务器上的 hardware session。

## 尚未完成的论文级检查

- 还没有多 seed 误差统计；
- 还没有匹配参数量电子 baseline；
- 还没有硬件逐层替换结果；
- 还没有对 LSP expert selection frequency 做全局占用审计；
- 还没有按论文最终版面与其他主图统一面板顺序和图注编号。
- Caltech101 当前快照的 phase LR 为 0；还没有纳入可学习 phase 的独立复现实验。
- Caltech101 的 router entropy/importance 显示单一 expert 强主导；当前只记录诊断，不把它解释为负载均衡的 MoE4。
