# 完整分层场景对照：TRAIN + TEST，Ours + Qwen

本目录是给画图同学的完整素材，不是挑选后的成功案例。包含 **TRAIN 5,000 条、TEST
1,000 条**，每条都有原始输入、目标、Ours预测、Qwen预测、文本指令和逐样本指标。
没有重新训练、选新权重、用目标修补预测，也没有删除失败案例。

## 最简单的使用方式

1. 打开 `pdf/01_test.pdf` 浏览测试图；训练图见 `pdf/02_train.pdf`；
   `pdf/00_all_train_test.pdf` 包含全部两部分。
2. 记下图上的六位序号，例如 **000148**。每页四条，四列依次是输入、目标、Ours、Qwen。
3. 双击 `00_browse.html`，输入序号、原 sample_id 或指令。可选择 train/test 和操作类型。
   浏览器直接打开即可，不需要模型、Python、GPU或联网。列表每次最多显示100条，
   搜索/筛选可找到全部6,000条；这只是避免浏览器同时加载24,000张图，不是删数据。
4. 点击图可以打开对应 `paper_896` 的无标注896px图，适合论文排版。224px原图在上一级。
   指标完整值、网格和图片SHA见同一文件夹的 `case.json`，不要从PDF的三位小数反推精确值。
5. 整个文件夹可以直接复制给画图同学；不要只复制HTML，否则相对图片链接会失效。

`00_index.json` 包含全量6,000条索引、文件夹路径、完整册/分册页码、所有指标、预测网格、
输入/目标网格、模型身份和图像SHA。后续AI可直接读取，不需要从PDF识别图片。

## 文件如何对应

```text
full_comparison_20261013/
  README.md
  00_browse.html                       可搜索的本地浏览页
  00_index.json                        全量原始指标和路径索引
  pdf/
    00_all_train_test.pdf              完整册：1,513页
    01_test.pdf                        TEST册：257页
    02_train.pdf                       TRAIN册：1,257页
  01_test/
    01_add/000148_test_000588/
      000148_01_input.png              原始224px输入，文件字节不变
      000148_02_target.png             原始224px目标，文件字节不变
      000148_03_ours.png               本次固定Ours权重真实预测
      000148_04_qwen.png               本次固定Qwen权重真实预测
      case.json                       指令、网格、指标、SHA、页码
      paper_896/                      同名4张无标注896px排版图
    02_replace/
    03_move/
    04_remove/
  02_train/                           相同四类/逐样本目录结构
  pdf_manifest.json                   PDF SHA、页数、样本数
  qa/                                 抽查页预览及核查记录
```

每条8张PNG，共 **24,000张224px图 + 24,000张896px图**。每条独立文件夹，不把不同指令的
操作混在一起。文件名带同一六位序号，复制到其他排版目录后也能追溯。

| 划分 | 操作 | 条数 | 六位序号范围 | 完整册图页 |
|---|---|---:|---|---|
| TEST | add 新增 | 250 | 000001–000250 | 3–65 |
| TEST | replace 替换 | 250 | 000251–000500 | 67–129 |
| TEST | move 移动 | 250 | 000501–000750 | 131–193 |
| TEST | remove 删除 | 250 | 000751–001000 | 195–257 |
| TRAIN | add 新增 | 1250 | 001001–002250 | 259–571 |
| TRAIN | replace 替换 | 1250 | 002251–003500 | 573–885 |
| TRAIN | move 移动 | 1250 | 003501–004750 | 887–1199 |
| TRAIN | remove 删除 | 1250 | 004751–006000 | 1201–1513 |

每类内部按原始 sample_id 排序，不按分数排序。PDF有八类操作书签、分节统计、可搜索序号。
PDF页码是从封面算起的实际页数，与文件夹 `case.json` 的页码一致。

此前四个故事样例仍可找到：

| 操作 | 六位序号 | 原 sample_id | 完整册页码 |
|---|---|---|---:|
| 新增 | 000148 | test_000588 | 39 |
| 替换 | 000275 | test_000097 | 73 |
| 移动 | 000692 | test_000766 | 178 |
| 删除 | 000960 | test_000839 | 247 |

## 指标是什么

- **Changed-cell accuracy / CCA**：指令要求改变的语义锚点格中，类别预测正确的比例。
  不是整个36格的准确率，也不是像素相似度。一次移动通常涉及旧位置和新位置两格，
  因此单图可能是0、0.5或1。
- **Edit-grid IoU**：模型预测编辑掩码与真实编辑掩码的交并比，包含错误编辑。
  即使一个预测编辑后的类别碰巧与原类别相同，该格仍按照实际预测编辑掩码参与IoU。
- `case.json` 还包括所有格准确率、前景类别准确率、保持格准确率、对象F1、
  整场景完全正确率和操作类别准确率。所有比例用0–1值，百分比需乘100。

全量固定权重结果（各条样本指标的平均值）：

| 划分 | 模型 | CCA | Edit-grid IoU | 对象F1 | 场景完全正确率 |
|---|---|---:|---:|---:|---:|
| TEST，1000条 | Ours | 0.876500 | 0.891417 | 0.962149 | 0.781000 |
| TEST，1000条 | Qwen | 0.812000 | 0.743833 | 0.935835 | 0.634000 |
| TRAIN，5000条 | Ours | 0.957400 | 0.965467 | 0.986488 | 0.916400 |
| TRAIN，5000条 | Qwen | 0.998700 | 0.933467 | 0.987203 | 0.894200 |

TRAIN只反映训练拟合，不能与TEST混合当成泛化指标。两个版本的TEST参与了选模，
所列TEST属于**开发指标**，不能在论文中写成未参与选模的独立测试。

## 图片与模型身份

- **Ours**：用户选定的分层场景版本，epoch45，PT SHA256：
  `03cb861c3ac344556601eb3eb6d7d1a22b77a54d2e7e68e85d77ee30fb09eb21`。
  每块SLM固定30%未调制强度，训练时加小幅截断偏置高斯CCD噪声；
  本次为固定PT的正常评估，训练扰动关闭，不是额外加噪声后的评估图。
- **Qwen**：epoch30共享读出头baseline，PT SHA256：
  `0be775796d5f64df3fab6359eb354ec6666dfc29329484a620b619a6f7ec7cf8`。
  完整冻结Qwen3-VL-2B-Instruct视觉/语言主干，训练适配器和同结构共享读出头。
  复用已封存的完整主干特征，重新运行固定权重适配器/读出头；未重训主干，
  也没有用在线聊天接口生成图片。
- 两者都预测语义锚点/编辑网格，再经过相同的OpenMoji比例大小/分层遮挡合成器出图。
  Qwen图不是模型自由生成的照片。896px版重新提高SVG采样分辨率，不改变位置、尺寸比例、
  深度顺序或遮挡，不是把224px图插值锐化。原始224px输入和目标保留原PNG字节，
  12,000张重渲染输入/目标均逐像素核对；错误预测不修复。预测越界只按原画布裁切，
  `clipped_predictions`记录越界对象。
- 原运行源码来自已验证工作树 commit
  `a61a3746d39765991048d95e40812d1028d70c93`；没有为出图改动该工作树。
  两PT、清单、缓存、推理源码SHA都记录在索引 `provenance` 中。
- TRAIN清单SHA：`452c0d4cede80ee8f838a853566b2bc52204d1c3c588fd2c6b4f7c99fac31fff`；
  TEST清单SHA：`93aa528c61fb7da116dc0bb7e749f7f3e525d1140050982e33604e3e72d22fca`。

### 数值精度核查的唯一差异

使用A100、原32条batch和BF16评估口径，两个模型的**1000条逐例CCA与IoU完全复现**。
Ours全部逐样本指标复现。Qwen `test_000486` 的对象F1从原封存0.5变为本次
0.5714285714；该条CCA/IoU及其余主要指标未变。其原因范围是固定权重数值边界，
不同CPU/混合精度的诊断记录也显示少数边界预测会变化，不能混用精度来逐条挑结果。
本包统一使用A100 BF16，不按样本切换精度，保留真实新网格及其重新计算的F1；
差异明细在 `archived_secondary_differences` 中。Qwen全量F1因此与旧报告相差约0.0000714，
表中和逐图JSON都使用本次真实值，不把旧指标冒充本次预测。

## 重新出图（给后续AI）

原始全量证据位于同一任务的
`runs/simulation/full_comparison_export_20261013_final/`：两模型各两份TRAIN/TEST JSONL、
原图归档、下载SHA清单及 `provenance.json`。证据/图片/PDF不进入Git。
本次图片/PDF保存在本地本目录，没有上传师姐服务器；Git只同步源码和说明。
本目录`.gitignore`仅忽略上表列出的生成素材和索引，不删除文件，不忽略README、
绘图源码或新的根目录说明。绘图入口会为以后新生成的同类目录自动建立相同局部规则；
若已有自定义规则则拒绝覆盖。代码三端同步不表示大图册也自动传到服务器。

从仓库根目录执行，输出必须是新的目录。绘图环境需要本任务的Torch/Pillow/numpy、
`resvg-py==0.5.0`；PDF步骤额外需要reportlab和pypdf，不加载模型、不使用GPU。

```powershell
python -m LightGenV2.tasks.t04_semantic_interaction.comparison_catalogue images `
  --evidence LightGenV2/tasks/t04_semantic_interaction/runs/simulation/full_comparison_export_20261013_final `
  --output LightGenV2/tasks/t04_semantic_interaction/reports/full_comparison_reproduction `
  --workers 4

python -m LightGenV2.tasks.t04_semantic_interaction.comparison_catalogue pdf `
  --output LightGenV2/tasks/t04_semantic_interaction/reports/full_comparison_reproduction
```

只有恢复自己中断的绘图任务时才加 `images --resume`，程序先校验已完成图片SHA与样本身份；
不能用来覆盖别人修改过的排版素材。最终图册全量核查6,000个编号、48,000张PNG和PDF编号覆盖，
另抽查PDF分节首尾/中间页。OpenMoji素材使用时请保留项目已有署名/版权说明。
