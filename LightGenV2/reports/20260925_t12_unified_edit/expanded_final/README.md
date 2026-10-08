# ABO 扩展版：灯、桌、枕头统一编辑

这是本轮选定的两份权重。旧的两类商品及试验性 v1/v2 预览仍留在上级目录；这里仅放最终候选和对应证据。大模型从修正过的 true-256 透明度掩码版本继续训练，小模型保留两层 Qwen 式语言 block，并从已有 13.7M → 9.5M 权重逐步迁移、局部续训。

## 功能与数据

同一份权重根据输入 RGB 商品图和文字执行三种编辑：只换背景、只换同类目标商品、同时换商品与背景。三种模式不是三份权重。训练的配对目标使用 ABO CleanRender 真实透明掩码和程序化房间合成；推理不输入掩码，也不贴回原图像素。光学和电子残差在瓶颈并行，单次生成，无 GAN、无扩散循环。

灯、桌、枕头每类训练/验证/测试各 48/8/8 个互斥商品身份。总计 1,728/192/192 张源视角图；每张派生 12 条编辑配对，因此是 20,736/2,304/2,304 对，而非这么多独立商品。每类有四款文字可指定的目标，共 12 款。数据清单记录 CC BY 4.0 及来源信息，见 [数据摘要](expanded_dataset_summary.json)。

| 候选 | 图像尺寸 | 计入参数 | Qwen 头 | 光学融合 α | 测试指标 |
| --- | ---: | ---: | --- | ---: | --- |
| 大模型 v2 | 256² | 142,544,637 | 两层、宽 640 | 0.510 | latent MSE：背景 0.0704 / 商品 0.0128 / 同时 0.0356 |
| 小模型 v3 | 128² | 9,495,704 | 两层、宽 512 | 0.496 | RGB MSE：背景 0.00107 / 商品 0.00090 / 同时 0.00152 |

以上参数口径按既定约定不计共享、冻结的 Qwen token embedding（311,164,928 参数）；大模型计入 VAE encoder、decoder 和光电 UNet，小模型为直接 RGB 输出、不含 VAE。两种 MSE 处于不同空间、不同分辨率，**不能互相比大小**。小模型改动把同数据上的商品 MSE 从 0.00117 降到 0.00090，同时编辑从 0.00186 降到 0.00152；放宽残差输出并加强编辑区域损失，没有增加参数。

## 看图与权重

- [大模型：三类、三种模式](expanded_large_v2_best.jpg)
- [大模型：每类四款目标](expanded_large_v2_four_designs.jpg)
- [小模型：三类、三种模式](expanded_small9m_v3_balanced_test.jpg)
- [小模型：每类四款目标及文字](expanded_small9m_v3_labeled_four_designs.jpg)
- 大模型单份打包权重 `expanded_large_v2_unified_editor.pt`：本地同目录，服务器 `/DATA/DATA1/guest3/t12_assets/runs/abo_unified_expanded_143m_true256_v2/unified_editor.pt`。不提交到 Git（138 MB）。
- 小模型单份权重 `expanded_small9m_v3_best_model.pt`：本地同目录，服务器 `/DATA/DATA1/guest3/t12_assets/runs/abo_unified_expanded_qwenmini_9m_v3/best_model.pt`。不提交到 Git（19 MB）。
- [大模型训练与测试记录](expanded_large_v2_training_summary.json)、[小模型训练记录](expanded_small9m_v3_training_summary.json)、[小模型分模式测试记录](expanded_small9m_v3_metrics.json)

限制：目标商品仍是固定的 12 款目录，背景也来自参数化房间组合；这能验证图文条件下的完整重绘，不足以宣称开放词汇、任意新商品生成。小模型在某些枕头形状切换上仍有淡残影；大模型效果明显更干净。固定目录且目标重复的配对集不适合用一个 FID 数字宣称开放集生成质量，本轮未报 FID。也未重测相对 Qwen+decoder baseline 的延迟，不能由参数量推断加速。
