# 分层故事图：Ours 与 Qwen baseline 的同输入对照

直接查看 `figures/story_ours_qwen_comparison.png`；同名 PDF 适合排版，SVG 保留可编辑文字。
每个操作另有一张 `a_add/b_replace/c_move/d_remove_comparison` 的 PNG/PDF/SVG；
四列的独立 896×896 PNG 和 `case_metrics.csv` 方便画图同学逐图摆放、标指标。

本次沿用之前的四个故事 ID，没有按 Qwen 失败重新挑选样本：

| 操作 | sample_id | 指令 | Ours 修改格准确率 / 编辑IoU | Qwen 修改格准确率 / 编辑IoU |
|---|---|---|---|---|
| 新增 | test_000588 | Add the ball to the left of the house. | 1.00 / 1.00 | 0.00 / 0.00 |
| 替换 | test_000097 | Replace the bicycle with a ball. | 1.00 / 1.00 | 0.00 / 0.00 |
| 移动 | test_000766 | Relocate the dog so it is to the left of the car. | 0.50 / 0.50 | 0.50 / 0.50 |
| 删除 | test_000839 | Erase the cat. | 1.00 / 1.00 | 0.00 / 0.00 |

Qwen 在新增、替换、删除三个例子中未执行编辑，保留了原场景。移动例子中两个模型
均移除了旧位置的狗，但没有在新位置放置狗；这是真实的部分失败，图中没有修补。
这些是具名的定性故事展示，不用四例均值替代全部1000条的结果。

## 模型身份与全量开发指标

- Ours：用户指定的分层场景 DC30 + 小 CCD 噪声版本，epoch45，
  `layered_dc30_ccdsmall_selected_e45_s73_20260925/selected_checkpoint.pt`；
  SHA256 `03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21`。
  全量1000条修改格准确率 **0.8765**，编辑IoU **0.8914167**。
- Qwen：`layered_scene_qwen_shared_s73_e100_20260920/best_checkpoint.pt`，epoch30；
  SHA256 `0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8`。
  全量1000条修改格准确率 **0.8120**。完整冻结Qwen3-VL-2B-Instruct视觉/语言主干，
  训练维度适配器与同结构共享读出头。本次复用已保存的完整主干特征，重新运行固定权重
  的适配器和读出头；未重新训练，也未对四个案例重新执行大模型主干。
- 相同TEST清单 SHA256：`93aa528c61fb7da116dc0bb7e749f7f3e525d1140050982e33604e3e72d22fca`。
  两模型的输入网格、目标网格、指令逐例完全相同。TEST参与选模，所列为开发指标。

这里的Qwen不是自由生成图片或调用在线聊天接口；两个系统都预测语义锚点/编辑网格，
再用相同OpenMoji分层合成器生成图片，合成阶段不修正模型的预测。

## 数据来源与可重复绘制

原始证据留在任务 `runs/smoke/story_comparison_20261013/`：

- `inference_records.json`：两个固定PT本次CPU推理的预测网格、编辑网格、逐例指标、
  PT与源码SHA、冻结Qwen缓存身份。推理源码来自原已验证工作树 commit
  `a61a3746d39765991048d95e40812d1028d70c93`，未改动其源码。
- `samples/<sample_id>/source.png,target.png` 与 `download_manifest.json`：原224px输入/目标，
  下载后的文件SHA。四例新推理的全部逐例指标与原封存记录误差小于1e-6。

所有展示列采用相同类别大小、锚点、绘制深度与遮挡规则。Source/Target重新渲染的
224px图与原数据图逐像素一致；896px版只提高SVG采样分辨率，布局不变。错误预测不移动、
不补全、不删去；越界预测按原画布裁切并记录，本次四例没有越界对象。

在已有绘图环境、安装本任务 `requirements-layered.txt` 的 resvg-py==0.5.0 后，从仓库根运行：

```powershell
python -m LightGenV2.tasks.t04_semantic_interaction.story_comparison `
  --evidence LightGenV2/tasks/t04_semantic_interaction/runs/smoke/story_comparison_20261013/inference_records.json `
  --output LightGenV2/tasks/t04_semantic_interaction/reports/story_comparison_20261013/figures_reproduction
```

此命令只读取保存的真实推理记录并绘图，无需加载模型或使用GPU。输出目录必须不存在，
防止覆盖已经排版的图。`figure_manifest.json` 记录证据、绘图脚本、SVG与导出文件SHA。
预测证据、PNG/PDF/SVG和CSV作为本地结果保留，源码和本文档进入Git。

## 可用图注

**Representative text-guided edits in layered OpenMoji scenes.** Identical RGB scenes
and instructions are supplied to both systems. Predicted semantic anchors and edit
masks are rendered using the same proportional-size compositor. Ours performs the
addition, replacement and removal in these previously selected examples, whereas
the frozen-Qwen baseline preserves the original objects. Both systems only partially
complete the relocation example. Changed-cell accuracy and edit-mask IoU are shown
for each output. These qualitative examples do not replace the 1000-case development
evaluation (Ours: 0.8765; Qwen baseline: 0.8120).
